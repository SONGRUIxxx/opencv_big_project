/**
 * @file main.cpp
 * @brief Fish-eye video motion region extraction and target tracking.
 *
 * Compares two routes:
 *   Route A: Process directly in fisheye image domain
 *   Route B: Undistort to perspective, then process
 *
 * Unified pipeline (from prj4 integration):
 *   ECC alignment → Frame diff + Farneback → Median-flow residual → Fusion → Filter → Evaluate
 *
 * Also retains original three methods (frame_diff, farneback, lk) as optional.
 *
 * Evaluates with IoU, Precision, Recall, F1-score,
 * including center vs. edge region analysis.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "fisheye_undistort.hpp"
#include "motion_extraction.hpp"
#include "tracking.hpp"
#include "evaluation.hpp"
#include "visualization.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

// ---- Configuration ----
struct Config {
    std::string dataRoot = "../homework2";
    std::string outputRoot = "./output";
    int maxSamples = -1;          // -1 = process all
    int visualLimit = 20;        // save detailed visuals for first N
    std::string fusionMode = "clean-and";
    std::string alignMode = "ecc";
    double focalScale = 0.27;
    bool showProgress = true;
};

// ---- Utility ----
static std::vector<std::string> getSampleIds(const std::string& rgbDir) {
    std::vector<std::string> ids;
    for (const auto& entry : fs::directory_iterator(rgbDir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.size() >= 8 && fname.substr(fname.size() - 7) == "_FV.png") {
            ids.push_back(fname.substr(0, fname.size() - 7));
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

static cv::Mat loadGray(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
        std::cerr << "ERROR: Cannot load image: " << path << std::endl;
    return img;
}

static cv::Mat loadBGR(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty())
        std::cerr << "ERROR: Cannot load image: " << path << std::endl;
    return img;
}

static cv::Mat loadGT(const std::string& path) {
    cv::Mat gt = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (gt.empty())
        std::cerr << "ERROR: Cannot load GT: " << path << std::endl;
    return gt;
}

static cv::Mat binaryMaskFromGt(const cv::Mat& gt) {
    cv::Mat binary;
    cv::threshold(gt, binary, 0, 255, cv::THRESH_BINARY);
    return binary;
}

// ---- Zone masks (center / edge) ----
static cv::Mat zoneMask(const cv::Size& size, double cx, double cy, double maxR,
                        bool centerZone) {
    cv::Mat mask(size, CV_8UC1, cv::Scalar(0));
    double threshold = centerZone ? 0.42 * maxR : 0.58 * maxR;
    for (int y = 0; y < size.height; ++y) {
        uchar* row = mask.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            double r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if ((centerZone && r <= threshold) || (!centerZone && r >= threshold))
                row[x] = 255;
        }
    }
    return mask;
}

// ---- Output helpers ----
static void createOutputDirs(const fs::path& root) {
    std::vector<std::string> dirs = {"fisheye", "undistorted", "flow", "masks", "boxes", "compare"};
    fs::create_directories(root);
    for (const auto& d : dirs)
        fs::create_directories(root / d);
}

static cv::Mat resizePanel(const cv::Mat& img, const cv::Size& size, const std::string& label) {
    cv::Mat panel;
    if (img.channels() == 1)
        cv::cvtColor(img, panel, cv::COLOR_GRAY2BGR);
    else
        panel = img.clone();
    cv::resize(panel, panel, size, 0.0, 0.0, cv::INTER_AREA);
    cv::rectangle(panel, {0, 0}, {panel.cols, 28}, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(panel, label, {8, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return panel;
}

static cv::Mat makeSummary(const cv::Mat& prev, const cv::Mat& curr,
                           const cv::Mat& gt, const UnifiedMotionResult& det,
                           const cv::Mat& overlay, const cv::Mat& tracked,
                           const EvalMetrics& metrics, const std::string& routeLabel) {
    const cv::Size panelSize(480, 362);

    cv::Mat gtVis;
    cv::normalize(gt, gtVis, 0, 255, cv::NORM_MINMAX);

    std::ostringstream metricLabel;
    metricLabel << std::fixed << std::setprecision(3)
                << routeLabel << " IoU=" << metrics.iou
                << " P=" << metrics.precision << " R=" << metrics.recall
                << " F1=" << metrics.f1;

    std::vector<cv::Mat> row1 = {
        resizePanel(prev, panelSize, "Previous frame"),
        resizePanel(curr, panelSize, "Current frame"),
        resizePanel(gtVis, panelSize, "Ground truth mask (>0)")
    };
    std::vector<cv::Mat> row2 = {
        resizePanel(det.motionMask, panelSize, "Predicted moving mask"),
        resizePanel(overlay, panelSize, metricLabel.str()),
        resizePanel(tracked, panelSize, "Boxes + short-term flow arrows")
    };
    std::vector<cv::Mat> row3 = {
        resizePanel(det.diffGray, panelSize, "Frame difference"),
        resizePanel(det.flowResidualMask, panelSize, "Median-flow residual mask"),
        resizePanel(det.flowVis, panelSize, "Farneback optical flow")
    };

    cv::Mat r1, r2, r3;
    cv::hconcat(row1, r1);
    cv::hconcat(row2, r2);
    cv::hconcat(row3, r3);

    cv::Mat summary;
    cv::vconcat(std::vector<cv::Mat>{r1, r2, r3}, summary);
    return summary;
}

static cv::Mat drawMaskOverlay(const cv::Mat& bgr, const cv::Mat& pred, const cv::Mat& gt) {
    cv::Mat out = bgr.clone();
    cv::Mat predBin, gtBin;
    cv::threshold(pred, predBin, 0, 255, cv::THRESH_BINARY);
    cv::threshold(gt, gtBin, 0, 255, cv::THRESH_BINARY);

    cv::Mat tp, fp, fn;
    cv::bitwise_and(predBin, gtBin, tp);
    cv::bitwise_and(predBin, cv::Scalar(255) - gtBin, fp);
    cv::bitwise_and(cv::Scalar(255) - predBin, gtBin, fn);

    out.setTo(cv::Scalar(0, 180, 0), tp);     // TP: green
    out.setTo(cv::Scalar(0, 0, 255), fp);      // FP: red
    out.setTo(cv::Scalar(255, 0, 0), fn);      // FN: blue
    cv::addWeighted(bgr, 0.55, out, 0.45, 0.0, out);
    return out;
}

static cv::Mat drawBoxesAndTracks(const cv::Mat& bgr,
                                   const std::vector<cv::Rect>& boxes,
                                   const cv::Mat& flow) {
    cv::Mat out = bgr.clone();
    for (size_t i = 0; i < boxes.size(); ++i) {
        const cv::Rect& box = boxes[i];
        cv::rectangle(out, box, cv::Scalar(0, 255, 255), 2);
        const cv::Point2f center(box.x + box.width * 0.5F, box.y + box.height * 0.5F);

        // Compute median flow in this box for arrow
        cv::Rect clipped = box & cv::Rect(0, 0, flow.cols, flow.rows);
        std::vector<float> xs, ys;
        for (int y = clipped.y; y < clipped.y + clipped.height; y += 2)
            for (int x = clipped.x; x < clipped.x + clipped.width; x += 2) {
                const cv::Point2f& f = flow.at<cv::Point2f>(y, x);
                if (std::isfinite(f.x) && std::abs(f.x) < 100.0F &&
                    std::isfinite(f.y) && std::abs(f.y) < 100.0F) {
                    xs.push_back(f.x); ys.push_back(f.y);
                }
            }
        if (!xs.empty()) {
            auto mx = xs.begin() + xs.size() / 2;
            auto my = ys.begin() + ys.size() / 2;
            std::nth_element(xs.begin(), mx, xs.end());
            std::nth_element(ys.begin(), my, ys.end());
            cv::Point2f v(*mx, *my);
            cv::arrowedLine(out, center - v * 4.0F, center,
                            cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.25);
        }
    }
    return out;
}

static std::vector<cv::Rect> boxesFromMask(const cv::Mat& mask, int minArea) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> boxes;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < minArea) continue;
        cv::Rect box = cv::boundingRect(c);
        if (box.width < 4 || box.height < 4) continue;
        boxes.push_back(box);
    }
    std::sort(boxes.begin(), boxes.end(),
              [](const cv::Rect& a, const cv::Rect& b) { return a.area() > b.area(); });
    return boxes;
}

// ---- Main ----
int main(int argc, char** argv) {
    Config cfg;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto needVal = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--data-root" && i + 1 < argc) {
            cfg.dataRoot = argv[++i];
        } else if (arg == "--output-root" && i + 1 < argc) {
            cfg.outputRoot = argv[++i];
        } else if (arg == "--max-samples" && i + 1 < argc) {
            cfg.maxSamples = std::stoi(argv[++i]);
        } else if (arg == "--visual-limit" && i + 1 < argc) {
            cfg.visualLimit = std::stoi(argv[++i]);
        } else if (arg == "--fusion") {
            cfg.fusionMode = needVal(arg);
        } else if (arg == "--align") {
            cfg.alignMode = needVal(arg);
        } else if (arg == "--focal-scale" && i + 1 < argc) {
            cfg.focalScale = std::stod(argv[++i]);
        } else if (arg == "--no-progress") {
            cfg.showProgress = false;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: fisheye_motion_tracking [OPTIONS]\n"
                      << "  --data-root PATH      Dataset root (default: ../homework2)\n"
                      << "  --output-root PATH    Output directory (default: ./output)\n"
                      << "  --max-samples N        Process at most N samples (-1 = all)\n"
                      << "  --visual-limit N       Save detailed vis for first N (default 20)\n"
                      << "  --fusion MODE          clean-and | raw-and | clean-or (default: clean-and)\n"
                      << "  --align MODE           ecc | feature | none (default: ecc)\n"
                      << "  --focal-scale X        Undistort focal scale (default: 0.27)\n"
                      << "  --no-progress          Suppress progress output\n";
            return 0;
        }
    }

    // Resolve paths
    fs::path dataPath = fs::absolute(cfg.dataRoot);
    fs::path outputPath = fs::absolute(cfg.outputRoot);
    fs::path rgbDir   = dataPath / "rgb_images";
    fs::path prevDir  = dataPath / "previous_images";
    fs::path calibDir = dataPath / "calibration_data";
    fs::path gtDir    = dataPath / "motion_annotation" / "GroudTruth";

    createOutputDirs(outputPath);

    // Get samples
    auto sampleIds = getSampleIds(rgbDir.string());
    if (cfg.maxSamples > 0 && cfg.maxSamples < static_cast<int>(sampleIds.size()))
        sampleIds.resize(cfg.maxSamples);

    std::cout << "==============================================================\n"
              << "  Fish-eye Motion Extraction & Tracking (Unified Pipeline)\n"
              << "==============================================================\n"
              << "Data root:     " << dataPath.string() << "\n"
              << "Output root:   " << outputPath.string() << "\n"
              << "Samples:       " << sampleIds.size() << "\n"
              << "Alignment:     " << cfg.alignMode << "\n"
              << "Fusion:        " << cfg.fusionMode << "\n"
              << "Visual limit:  " << cfg.visualLimit << "\n"
              << "==============================================================\n\n";

    // ---- Initialize ----
    MotionExtractor extractor;
    extractor.fusionMode = cfg.fusionMode;
    extractor.alignMode = cfg.alignMode;
    MotionTracker tracker;

    // Route B: build undistorter from first sample's calibration
    std::unique_ptr<FisheyeUndistorter> undistorter;
    if (!sampleIds.empty()) {
        fs::path firstCalib = calibDir / (sampleIds[0] + "_FV.json");
        if (fs::exists(firstCalib)) {
            std::cout << "Building fisheye undistorter... ";
            undistorter = std::make_unique<FisheyeUndistorter>(firstCalib.string(), 120.0, 1024);
            std::cout << "Done. Output: "
                      << undistorter->getOutputSize().width << "x"
                      << undistorter->getOutputSize().height << "\n";
        }
    }

    // Route A evaluator
    double cxA, cyA, maxRA;
    {
        fs::path firstCalib = calibDir / (sampleIds[0] + "_FV.json");
        FisheyeUndistorter tmp(firstCalib.string());
        cxA = tmp.getPrincipalPoint().x;
        cyA = tmp.getPrincipalPoint().y;
        maxRA = tmp.getMaxRadius();
    }
    Evaluator evalA(cxA, cyA, maxRA, 0.42, 0.58);

    // Route B evaluator
    std::unique_ptr<Evaluator> evalB;
    if (undistorter) {
        cv::Size outSz = undistorter->getOutputSize();
        double cxB = outSz.width / 2.0;
        double cyB = outSz.height / 2.0;
        double maxRB = std::sqrt(cxB * cxB + cyB * cyB);
        evalB = std::make_unique<Evaluator>(cxB, cyB, maxRB, 0.42, 0.58);
    }

    // ---- Collect all metrics ----
    std::vector<EvalMetrics> allMetrics;
    int processed = 0;
    auto startTime = std::chrono::steady_clock::now();

    for (size_t idx = 0; idx < sampleIds.size(); idx++) {
        const auto& sid = sampleIds[idx];

        fs::path rgbPath   = rgbDir   / (sid + "_FV.png");
        fs::path prevPath  = prevDir  / (sid + "_FV_prev.png");
        fs::path calibPath = calibDir / (sid + "_FV.json");
        fs::path gtPath    = gtDir    / (sid + "_FV.png");

        cv::Mat bgrCurr = loadBGR(rgbPath.string());
        cv::Mat grayCurr = loadGray(rgbPath.string());
        cv::Mat grayPrev = loadGray(prevPath.string());
        cv::Mat gtRaw = loadGT(gtPath.string());

        if (bgrCurr.empty() || grayCurr.empty() || grayPrev.empty()) {
            std::cerr << "Skipping sample " << sid << " (missing data)\n";
            continue;
        }

        cv::Mat gt = binaryMaskFromGt(gtRaw);

        // ---- Route A: Fisheye domain ----
        {
            cv::Mat validA = Evaluator::validImageMask(bgrCurr);

            // Unified motion detection
            auto det = extractor.detectMotion(grayPrev, grayCurr, validA);

            // Additional object-like filtering
            det.motionMask = MotionTracker::filterObjectLikeComponents(
                det.motionMask, extractor.minArea);

            // Extract boxes from final mask
            auto boxes = boxesFromMask(det.motionMask, extractor.minArea);

            // Overlay + tracking visualization
            cv::Mat overlayA = drawMaskOverlay(bgrCurr, det.motionMask, gt);
            cv::Mat trackedA = drawBoxesAndTracks(bgrCurr, boxes, det.flow);

            // Evaluate
            auto metricsA = evalA.evaluate(det.motionMask, gt, sid, "A", "unified");
            for (auto& m : metricsA) allMetrics.push_back(m);

            int numBoxes = static_cast<int>(boxes.size());

            // Save visuals
            if (static_cast<int>(idx) < cfg.visualLimit) {
                cv::imwrite((outputPath / "masks" / (sid + "_A_pred.png")).string(), det.motionMask);
                cv::imwrite((outputPath / "masks" / (sid + "_A_gt.png")).string(), gt);
                cv::imwrite((outputPath / "flow"   / (sid + "_A_flow.png")).string(), det.flowVis);
                cv::imwrite((outputPath / "boxes"  / (sid + "_A_boxes.png")).string(), trackedA);
                cv::imwrite((outputPath / "fisheye" / (sid + "_overlay.png")).string(), overlayA);

                // Summary (9-panel)
                if (!metricsA.empty()) {
                    cv::Mat summary = makeSummary(grayPrev, bgrCurr, gt, det,
                                                  overlayA, trackedA, metricsA[0], "Fish-A");
                    cv::imwrite((outputPath / "compare" / (sid + "_A_summary.jpg")).string(), summary);
                }
            }

            if (cfg.showProgress && (processed + 1) % 10 == 0) {
                std::cout << "  [" << (processed + 1) << "/" << sampleIds.size()
                          << "] A-" << sid << " boxes=" << numBoxes
                          << " F1=" << std::fixed << std::setprecision(4)
                          << (metricsA.empty() ? 0.0 : metricsA[0].f1) << "\n";
            }
        }

        // ---- Route B: Undistorted domain ----
        if (undistorter) {
            cv::Mat bgrCurrB = undistorter->undistort(bgrCurr);
            cv::Mat grayCurrB, grayPrevB;
            cv::cvtColor(bgrCurrB, grayCurrB, cv::COLOR_BGR2GRAY);

            cv::Mat bgrPrev = loadBGR(prevPath.string());
            cv::Mat bgrPrevB = undistorter->undistort(bgrPrev);
            cv::cvtColor(bgrPrevB, grayPrevB, cv::COLOR_BGR2GRAY);

            cv::Mat gtB = undistorter->undistortMask(gt);

            cv::Mat validB = Evaluator::validImageMask(bgrCurrB);

            auto detB = extractor.detectMotion(grayPrevB, grayCurrB, validB);
            detB.motionMask = MotionTracker::filterObjectLikeComponents(
                detB.motionMask, extractor.minArea);

            auto boxesB = boxesFromMask(detB.motionMask, extractor.minArea);
            cv::Mat overlayB = drawMaskOverlay(bgrCurrB, detB.motionMask, gtB);
            cv::Mat trackedB = drawBoxesAndTracks(bgrCurrB, boxesB, detB.flow);

            auto metricsB = evalB->evaluate(detB.motionMask, gtB, sid, "B", "unified");
            for (auto& m : metricsB) allMetrics.push_back(m);

            if (static_cast<int>(idx) < cfg.visualLimit) {
                cv::imwrite((outputPath / "masks" / (sid + "_B_pred.png")).string(), detB.motionMask);
                cv::imwrite((outputPath / "undistorted" / (sid + "_current.png")).string(), bgrCurrB);
                cv::imwrite((outputPath / "undistorted" / (sid + "_overlay.png")).string(), overlayB);
                cv::imwrite((outputPath / "undistorted" / (sid + "_boxes.png")).string(), trackedB);

                if (!metricsB.empty()) {
                    cv::Mat summaryB = makeSummary(grayPrevB, bgrCurrB, gtB, detB,
                                                   overlayB, trackedB, metricsB[0], "Undist-B");
                    cv::imwrite((outputPath / "compare" / (sid + "_B_summary.jpg")).string(), summaryB);
                }
            }

            if (cfg.showProgress && (processed + 1) % 10 == 0) {
                std::cout << "  [" << (processed + 1) << "/" << sampleIds.size()
                          << "] B-" << sid << " F1=" << std::fixed << std::setprecision(4)
                          << (metricsB.empty() ? 0.0 : metricsB[0].f1) << "\n";
            }
        }

        processed++;
    }

    // ---- Final summary ----
    auto endTime = std::chrono::steady_clock::now();
    auto totalSec = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();

    std::cout << "\n==============================================================\n"
              << "  Processing Complete\n"
              << "==============================================================\n"
              << "Samples:    " << processed << "\n"
              << "Time:       " << totalSec << " s\n"
              << "Metrics:    " << allMetrics.size() << "\n";

    // Write CSV
    std::string csvPath = (outputPath / "metrics.csv").string();

    // Write prj4-style metrics CSV
    {
        std::ofstream out(csvPath);
        out << std::fixed << std::setprecision(6);
        out << "id,route,method,region,tp,fp,fn,tn,iou,precision,recall,f1\n";
        for (const auto& m : allMetrics) {
            out << m.sampleId << "," << m.route << "," << m.method << "," << m.region << ","
                << m.tp << "," << m.fp << "," << m.fn << "," << m.tn << ","
                << m.iou << "," << m.precision << "," << m.recall << "," << m.f1 << "\n";
        }
    }
    std::cout << "Metrics CSV: " << csvPath << "\n";

    // Print average metrics by route + region
    auto printAvg = [&](const std::string& route, const std::string& region) {
        double iou = 0, prec = 0, rec = 0, f1 = 0;
        int count = 0;
        for (const auto& m : allMetrics) {
            if (m.route == route && m.region == region) {
                iou += m.iou; prec += m.precision; rec += m.recall; f1 += m.f1;
                count++;
            }
        }
        if (count > 0) {
            iou /= count; prec /= count; rec /= count; f1 /= count;
            std::cout << std::fixed << std::setprecision(4)
                      << "  " << route << "/" << region
                      << ": IoU=" << iou << " Prec=" << prec
                      << " Rec=" << rec << " F1=" << f1
                      << " (n=" << count << ")\n";
        }
    };

    std::cout << "\nAverage metrics:\n";
    for (const auto& r : {"A", "B"}) {
        for (const auto& reg : {"full", "center", "edge"}) {
            printAvg(r, reg);
        }
    }

    // Generate aggregate charts
    if (!allMetrics.empty()) {
        std::string chartDir = (outputPath / "comparison_charts").string();
        fs::create_directories(chartDir);
        Visualizer::createComparisonCharts(chartDir, allMetrics);
        Visualizer::createCenterVsEdgeChart(chartDir, allMetrics);
        std::cout << "Charts: " << chartDir << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}

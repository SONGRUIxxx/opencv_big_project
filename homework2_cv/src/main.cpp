/**
 * @file main.cpp
 * @brief Fish-eye video motion region extraction and target tracking.
 *
 * Compares two routes:
 *   Route A: Process directly in fisheye image domain
 *   Route B: Undistort to perspective, then process
 *
 * Uses three motion extraction methods:
 *   1. Frame differencing
 *   2. Farneback dense optical flow
 *   3. Lucas-Kanade sparse optical flow
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
    std::string dataRoot;         // e.g., "../homework2"
    std::string outputRoot;       // e.g., "./output"
    int maxSamples = -1;          // -1 = process all, N = process first N
    bool enableRouteA = true;
    bool enableRouteB = true;
    bool enableFrameDiff = true;
    bool enableFarneback = true;
    bool enableLK = true;
    bool showProgress = true;
};

// ---- Utility ----
static std::vector<std::string> getSampleIds(const std::string& rgbDir) {
    std::vector<std::string> ids;
    for (const auto& entry : fs::directory_iterator(rgbDir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        // Expect format: {ID}_FV.png  (e.g., 00000_FV.png)
        if (fname.size() >= 8 && fname.substr(fname.size() - 7) == "_FV.png") {
            std::string id = fname.substr(0, fname.size() - 7);
            ids.push_back(id);
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

static cv::Mat loadGray(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "ERROR: Cannot load image: " << path << std::endl;
    }
    return img;
}

static cv::Mat loadBGR(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "ERROR: Cannot load image: " << path << std::endl;
    }
    return img;
}

static cv::Mat loadGT(const std::string& path) {
    cv::Mat gt = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (gt.empty()) {
        std::cerr << "ERROR: Cannot load GT: " << path << std::endl;
    }
    return gt;
}

// ---- Process one sample with one method ----
struct SampleResult {
    std::string sampleId;
    std::string route;
    std::string method;

    // Motion results
    cv::Mat motionMask;
    double motionRatio = 0.0;

    // Tracking
    std::vector<TrackedRegion> trackedRegions;

    // Evaluation (full, center, edge)
    std::vector<EvalMetrics> metrics;

    // Optional visualizations
    cv::Mat motionOverlayImg;    // image with green overlay
    cv::Mat flowVisImg;          // HSV flow visualization
    cv::Mat flowLegendImg;       // color wheel legend
    cv::Mat lkVisImg;            // LK sparse visualization
    cv::Mat trackingVisImg;      // tracking visualization
    cv::Mat gtComparisonImg;     // GT comparison
};

static SampleResult processSample(const std::string& sampleId,
                                   const Config& cfg,
                                   const cv::Mat& grayPrev,
                                   const cv::Mat& grayCurr,
                                   const cv::Mat& bgrCurr,
                                   const cv::Mat& gtMask,
                                   const MotionExtractor& extractor,
                                   MotionTracker& tracker,
                                   const Evaluator* evaluator,
                                   const std::string& route,
                                   const std::string& method)
{
    SampleResult result;
    result.sampleId = sampleId;
    result.route = route;
    result.method = method;

    // ---- Apply motion extraction method ----
    cv::Mat flowForTracking;  // store Farneback flow for reuse in tracking

    if (method == "frame_diff") {
        auto fd = extractor.frameDifference(grayPrev, grayCurr);
        result.motionMask = fd.motionMask;
        result.motionRatio = fd.motionRatio;
    } else if (method == "farneback") {
        auto fb = extractor.farnebackFlow(grayPrev, grayCurr);
        result.motionMask = fb.motionMask;
        result.motionRatio = fb.motionRatio;
        result.flowVisImg = fb.flowVis;
        result.flowLegendImg = fb.flowVisLegend;
        flowForTracking = fb.flow;  // reuse for tracking, avoid recompute
    } else if (method == "lk") {
        auto lk = extractor.lucasKanadeSparse(grayPrev, grayCurr);
        result.lkVisImg = lk.visualization;
        // LK doesn't produce a dense motion mask – create an empty one
        result.motionMask = cv::Mat::zeros(grayPrev.size(), CV_8UC1);
        result.motionRatio = 0.0;
    }

    // ---- Tracking (skip for LK which is already sparse) ----
    cv::Mat trackingImg = bgrCurr.clone();
    if (method != "lk" && !result.motionMask.empty()) {
        result.trackedRegions = tracker.extractRegions(result.motionMask, flowForTracking);
        MotionTracker::drawTracking(trackingImg, result.trackedRegions);
    }
    result.trackingVisImg = trackingImg;

    // ---- Create motion overlay ----
    if (method != "lk" && !result.motionMask.empty()) {
        result.motionOverlayImg = bgrCurr.clone();
        Visualizer::addOverlay(result.motionOverlayImg, result.motionMask,
                               cv::Scalar(0, 255, 0), 0.4);
    } else if (method == "lk") {
        result.motionOverlayImg = result.lkVisImg;
    }

    // ---- Evaluation ----
    if (evaluator && !gtMask.empty() && method != "lk") {
        result.metrics = evaluator->evaluate(result.motionMask, gtMask,
                                              sampleId, route, method);
    }

    return result;
}


// ---- Main ----
int main(int argc, char** argv) {
    // Parse arguments
    Config cfg;
    cfg.dataRoot = "../homework2";
    cfg.outputRoot = "./output";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data-root" && i + 1 < argc) {
            cfg.dataRoot = argv[++i];
        } else if (arg == "--output-root" && i + 1 < argc) {
            cfg.outputRoot = argv[++i];
        } else if (arg == "--max-samples" && i + 1 < argc) {
            cfg.maxSamples = std::stoi(argv[++i]);
        } else if (arg == "--no-route-a") {
            cfg.enableRouteA = false;
        } else if (arg == "--no-route-b") {
            cfg.enableRouteB = false;
        } else if (arg == "--no-frame-diff") {
            cfg.enableFrameDiff = false;
        } else if (arg == "--no-farneback") {
            cfg.enableFarneback = false;
        } else if (arg == "--no-lk") {
            cfg.enableLK = false;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: fisheye_motion_tracking [OPTIONS]\n"
                      << "  --data-root PATH     Dataset root (default: ../homework2)\n"
                      << "  --output-root PATH   Output directory (default: ./output)\n"
                      << "  --max-samples N      Process at most N samples\n"
                      << "  --no-route-a         Disable Route A (fisheye domain)\n"
                      << "  --no-route-b         Disable Route B (rectified domain)\n"
                      << "  --no-frame-diff      Disable frame difference\n"
                      << "  --no-farneback       Disable Farneback optical flow\n"
                      << "  --no-lk              Disable Lucas-Kanade sparse flow\n";
            return 0;
        }
    }

    // Resolve paths (fs::path handles / vs \ on all platforms)
    fs::path dataPath = fs::absolute(cfg.dataRoot);
    fs::path outputPath = fs::absolute(cfg.outputRoot);

    fs::path rgbDir   = dataPath / "rgb_images";
    fs::path prevDir  = dataPath / "previous_images";
    fs::path calibDir = dataPath / "calibration_data";
    fs::path gtDir    = dataPath / "motion_annotation" / "GroudTruth";

    // Ensure output directories exist
    fs::create_directories(outputPath / "visualizations");
    fs::create_directories(outputPath / "comparison_charts");

    // Get sample IDs
    auto sampleIds = getSampleIds(rgbDir.string());
    if (cfg.maxSamples > 0 && cfg.maxSamples < static_cast<int>(sampleIds.size())) {
        sampleIds.resize(cfg.maxSamples);
    }

    std::cout << "==============================================================\n";
    std::cout << "  Fish-eye Motion Extraction & Tracking\n";
    std::cout << "==============================================================\n";
    std::cout << "Data root:    " << dataPath.string() << "\n";
    std::cout << "Output root:  " << outputPath.string() << "\n";
    std::cout << "Samples:      " << sampleIds.size() << "\n";
    std::cout << "Route A:      " << (cfg.enableRouteA ? "ON" : "OFF") << "\n";
    std::cout << "Route B:      " << (cfg.enableRouteB ? "ON" : "OFF") << "\n";
    std::cout << "Frame Diff:   " << (cfg.enableFrameDiff ? "ON" : "OFF") << "\n";
    std::cout << "Farneback:    " << (cfg.enableFarneback ? "ON" : "OFF") << "\n";
    std::cout << "Lucas-Kanade: " << (cfg.enableLK ? "ON" : "OFF") << "\n";
    std::cout << "==============================================================\n\n";

    // ---- Initialize ----
    MotionExtractor extractor;   // uses default parameters from report
    MotionTracker tracker;       // uses default parameters from report

    // Route B: build undistorter once (all samples share same calibration)
    std::unique_ptr<FisheyeUndistorter> undistorter;
    std::unique_ptr<Evaluator> evalB;
    if (cfg.enableRouteB) {
        fs::path firstCalib = calibDir / (sampleIds[0] + "_FV.json");
        if (fs::exists(firstCalib)) {
            std::cout << "Building fisheye undistorter... ";
            undistorter = std::make_unique<FisheyeUndistorter>(firstCalib.string(), 120.0, 1024);
            std::cout << "Done. Output size: "
                      << undistorter->getOutputSize().width << "x"
                      << undistorter->getOutputSize().height << "\n";

            // Evaluator for Route B (needs output image size, principal point at image center)
            cv::Size outSz = undistorter->getOutputSize();
            double cxB = outSz.width / 2.0;
            double cyB = outSz.height / 2.0;
            double maxRB = std::sqrt(cxB * cxB + cyB * cyB);
            evalB = std::make_unique<Evaluator>(cxB, cyB, maxRB, 0.40, 0.60);
        }
    }

    // Route A evaluator: principal point from fisheye image
    std::unique_ptr<Evaluator> evalA;
    if (cfg.enableRouteA) {
        fs::path firstCalib = calibDir / (sampleIds[0] + "_FV.json");
        FisheyeUndistorter tmp(firstCalib.string());  // just to get params
        double cxA = tmp.getPrincipalPoint().x;
        double cyA = tmp.getPrincipalPoint().y;
        double maxRA = tmp.getMaxRadius();
        evalA = std::make_unique<Evaluator>(cxA, cyA, maxRA, 0.40, 0.60);
    }

    // ---- Process all samples ----
    std::vector<EvalMetrics> allMetrics;
    int processedCount = 0;
    auto startTime = std::chrono::steady_clock::now();

    for (size_t idx = 0; idx < sampleIds.size(); idx++) {
        const auto& sid = sampleIds[idx];

        // Load data (fs::path handles cross-platform separators)
        fs::path rgbPath   = rgbDir   / (sid + "_FV.png");
        fs::path prevPath  = prevDir  / (sid + "_FV_prev.png");
        fs::path calibPath = calibDir / (sid + "_FV.json");
        fs::path gtPath    = gtDir    / (sid + "_FV.png");

        cv::Mat bgrCurr = loadBGR(rgbPath.string());
        cv::Mat grayCurr = loadGray(rgbPath.string());
        cv::Mat grayPrev = loadGray(prevPath.string());
        cv::Mat gtMask = loadGT(gtPath.string());

        if (bgrCurr.empty() || grayCurr.empty() || grayPrev.empty()) {
            std::cerr << "Skipping sample " << sid << " (missing data)\n";
            continue;
        }

        // Output directory for this sample's visualizations
        std::string sampleVisDir = (outputPath / "visualizations" / sid).string();
        Visualizer::setSampleOutputDir(sampleVisDir);

        // ---- Route A: Process in fisheye domain ----
        if (cfg.enableRouteA) {
            if (cfg.enableFrameDiff) {
                auto res = processSample(sid, cfg, grayPrev, grayCurr,
                                          bgrCurr, gtMask,
                                          extractor, tracker, evalA.get(),
                                          "A", "frame_diff");

                if (!res.motionOverlayImg.empty())
                    Visualizer::saveMotionOverlay(sid, "A", "frame_diff",
                                                   bgrCurr, res.motionMask);
                if (!res.trackingVisImg.empty())
                    Visualizer::saveTrackingVis(sid, "A", res.trackingVisImg);
                if (!gtMask.empty() && !res.motionMask.empty()) {
                    auto fullMetrics = res.metrics.empty() ? EvalMetrics{} : res.metrics[0];
                    Visualizer::saveGTComparison(sid, "A", "frame_diff",
                                                  bgrCurr, res.motionMask, gtMask, fullMetrics);
                }
                for (auto& m : res.metrics) allMetrics.push_back(m);
            }

            if (cfg.enableFarneback) {
                auto res = processSample(sid, cfg, grayPrev, grayCurr,
                                          bgrCurr, gtMask,
                                          extractor, tracker, evalA.get(),
                                          "A", "farneback");

                if (!res.flowVisImg.empty())
                    Visualizer::saveFlowVis(sid, "A", res.flowVisImg, res.flowLegendImg);
                if (!res.motionOverlayImg.empty())
                    Visualizer::saveMotionOverlay(sid, "A", "farneback",
                                                   bgrCurr, res.motionMask);
                if (!res.trackingVisImg.empty())
                    Visualizer::saveTrackingVis(sid, "A", res.trackingVisImg);
                if (!gtMask.empty() && !res.motionMask.empty()) {
                    auto fullMetrics = res.metrics.empty() ? EvalMetrics{} : res.metrics[0];
                    Visualizer::saveGTComparison(sid, "A", "farneback",
                                                  bgrCurr, res.motionMask, gtMask, fullMetrics);
                }
                for (auto& m : res.metrics) allMetrics.push_back(m);
            }

            if (cfg.enableLK) {
                auto res = processSample(sid, cfg, grayPrev, grayCurr,
                                          bgrCurr, gtMask,
                                          extractor, tracker, evalA.get(),
                                          "A", "lk");

                if (!res.lkVisImg.empty())
                    Visualizer::saveLKVis(sid, "A", res.lkVisImg);
            }
        }

        // ---- Route B: Undistort then process ----
        if (cfg.enableRouteB && undistorter) {
            // Undistort images
            cv::Mat bgrCurrB = undistorter->undistort(bgrCurr);
            cv::Mat grayCurrB, grayPrevB;
            cv::cvtColor(bgrCurrB, grayCurrB, cv::COLOR_BGR2GRAY);

            cv::Mat bgrPrev = loadBGR(prevPath.string());
            cv::Mat bgrPrevB = undistorter->undistort(bgrPrev);
            cv::cvtColor(bgrPrevB, grayPrevB, cv::COLOR_BGR2GRAY);

            // Undistort the GT mask (nearest-neighbor to preserve binary)
            cv::Mat gtMaskB = undistorter->undistortMask(gtMask);

            // Save undistort comparison (for first 5 samples only, to save space)
            if (idx < 5) {
                Visualizer::saveUndistortComparison(sid, bgrCurr, bgrCurrB);
            }

            if (cfg.enableFrameDiff) {
                auto res = processSample(sid, cfg, grayPrevB, grayCurrB,
                                          bgrCurrB, gtMaskB,
                                          extractor, tracker, evalB.get(),
                                          "B", "frame_diff");

                if (!res.motionOverlayImg.empty())
                    Visualizer::saveMotionOverlay(sid, "B", "frame_diff",
                                                   bgrCurrB, res.motionMask);
                if (!res.trackingVisImg.empty())
                    Visualizer::saveTrackingVis(sid, "B", res.trackingVisImg);
                if (!gtMaskB.empty() && !res.motionMask.empty()) {
                    auto fullMetrics = res.metrics.empty() ? EvalMetrics{} : res.metrics[0];
                    Visualizer::saveGTComparison(sid, "B", "frame_diff",
                                                  bgrCurrB, res.motionMask, gtMaskB, fullMetrics);
                }
                for (auto& m : res.metrics) allMetrics.push_back(m);
            }

            if (cfg.enableFarneback) {
                auto res = processSample(sid, cfg, grayPrevB, grayCurrB,
                                          bgrCurrB, gtMaskB,
                                          extractor, tracker, evalB.get(),
                                          "B", "farneback");

                if (!res.flowVisImg.empty())
                    Visualizer::saveFlowVis(sid, "B", res.flowVisImg, res.flowLegendImg);
                if (!res.motionOverlayImg.empty())
                    Visualizer::saveMotionOverlay(sid, "B", "farneback",
                                                   bgrCurrB, res.motionMask);
                if (!res.trackingVisImg.empty())
                    Visualizer::saveTrackingVis(sid, "B", res.trackingVisImg);
                if (!gtMaskB.empty() && !res.motionMask.empty()) {
                    auto fullMetrics = res.metrics.empty() ? EvalMetrics{} : res.metrics[0];
                    Visualizer::saveGTComparison(sid, "B", "farneback",
                                                  bgrCurrB, res.motionMask, gtMaskB, fullMetrics);
                }
                for (auto& m : res.metrics) allMetrics.push_back(m);
            }

            if (cfg.enableLK) {
                auto res = processSample(sid, cfg, grayPrevB, grayCurrB,
                                          bgrCurrB, gtMaskB,
                                          extractor, tracker, evalB.get(),
                                          "B", "lk");

                if (!res.lkVisImg.empty())
                    Visualizer::saveLKVis(sid, "B", res.lkVisImg);
            }
        }

        processedCount++;

        // Progress
        if (cfg.showProgress && processedCount % 10 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            double rate = static_cast<double>(processedCount) / std::max(1.0, static_cast<double>(elapsed));
            std::cout << "[" << processedCount << "/" << sampleIds.size()
                      << "] " << sid << " ("
                      << std::fixed << std::setprecision(1) << rate << " samples/s)\n";
        }
    }

    // ---- Final summary ----
    auto endTime = std::chrono::steady_clock::now();
    auto totalSec = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();

    std::cout << "\n==============================================================\n";
    std::cout << "  Processing Complete\n";
    std::cout << "==============================================================\n";
    std::cout << "Samples processed: " << processedCount << "\n";
    std::cout << "Total time:        " << totalSec << " s\n";
    std::cout << "Metrics collected: " << allMetrics.size() << "\n";

    // Write results CSV
    std::string csvPath = (outputPath / "results_summary.csv").string();
    Evaluator::writeResultsCSV(csvPath, allMetrics);
    std::cout << "Results CSV:       " << csvPath << "\n";

    // Create aggregate charts
    std::string chartDir = (outputPath / "comparison_charts").string();
    Visualizer::createComparisonCharts(chartDir, allMetrics);
    Visualizer::createCenterVsEdgeChart(chartDir, allMetrics);
    std::cout << "Charts saved to:   " << chartDir << "\n";

    // Print summary table (mean metrics across all samples)
    auto printMean = [&](const std::string& method, const std::string& route, const std::string& region) {
        double iou = 0, prec = 0, rec = 0, f1 = 0;
        int count = 0;
        for (const auto& m : allMetrics) {
            if (m.method == method && m.route == route && m.region == region) {
                iou += m.iou;
                prec += m.precision;
                rec += m.recall;
                f1 += m.f1;
                count++;
            }
        }
        if (count > 0) {
            iou /= count; prec /= count; rec /= count; f1 /= count;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << std::setw(12) << route << " | "
                      << std::setw(8) << region << " | "
                      << std::setw(8) << iou << " | "
                      << std::setw(8) << prec << " | "
                      << std::setw(8) << rec << " | "
                      << std::setw(8) << f1 << "\n";
        }
    };

    std::cout << "\n--- Average Metrics (Frame Difference) ---\n";
    std::cout << "       Route |   Region |     IoU |   Prec |   Recall |       F1\n";
    std::cout << "-------------+----------+---------+--------+----------+---------\n";
    for (const auto& r : {"A", "B"}) {
        for (const auto& reg : {"full", "center", "edge"}) {
            printMean("frame_diff", r, reg);
        }
    }

    std::cout << "\n--- Average Metrics (Farneback) ---\n";
    std::cout << "       Route |   Region |     IoU |   Prec |   Recall |       F1\n";
    std::cout << "-------------+----------+---------+--------+----------+---------\n";
    for (const auto& r : {"A", "B"}) {
        for (const auto& reg : {"full", "center", "edge"}) {
            printMean("farneback", r, reg);
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}

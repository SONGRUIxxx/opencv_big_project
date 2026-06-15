#include "visualization.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cstdio>
#include <map>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

std::string Visualizer::sampleOutputDir_;

void Visualizer::setSampleOutputDir(const std::string& dir) {
    sampleOutputDir_ = dir;
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
}

// =====================================================================
//  Single-image helper
// =====================================================================

static cv::Mat resizeKeepAspect(const cv::Mat& src, int targetW) {
    double ratio = static_cast<double>(targetW) / src.cols;
    int h = static_cast<int>(src.rows * ratio);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(targetW, h));
    return resized;
}

// =====================================================================
//  Undistort comparison
// =====================================================================

void Visualizer::saveUndistortComparison(const std::string& sampleId,
                                          const cv::Mat& fisheye,
                                          const cv::Mat& perspective,
                                          const std::string& suffix) {
    // Resize both to same height for side-by-side
    int outH = std::min(fisheye.rows, 600);
    double scaleF = static_cast<double>(outH) / fisheye.rows;
    int wF = static_cast<int>(fisheye.cols * scaleF);

    double scaleP = static_cast<double>(outH) / perspective.rows;
    int wP = static_cast<int>(perspective.cols * scaleP);

    cv::Mat fishR, perspR;
    cv::resize(fisheye, fishR, cv::Size(wF, outH));
    cv::resize(perspective, perspR, cv::Size(wP, outH));

    // Side-by-side concatenation
    cv::Mat concat;
    cv::hconcat(fishR, perspR, concat);

    // Labels
    cv::putText(concat, "Original Fisheye", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(concat, "Rectified (HFOV=120)", cv::Point(wF + 10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    std::string fname = sampleOutputDir_ + "/" + sampleId + "_undistorted_comp.png";
    if (!suffix.empty()) {
        fname = sampleOutputDir_ + "/" + sampleId + "_" + suffix + "_undistorted_comp.png";
    }
    cv::imwrite(fname, concat);
}

// =====================================================================
//  Motion overlay
// =====================================================================

void Visualizer::addOverlay(cv::Mat& image, const cv::Mat& mask,
                             const cv::Scalar& color, double alpha) {
    for (int y = 0; y < image.rows; y++) {
        const uchar* mRow = mask.ptr<uchar>(y);
        cv::Vec3b* imgRow = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; x++) {
            if (mRow[x] > 127) {
                imgRow[x][0] = static_cast<uchar>(
                    imgRow[x][0] * (1.0 - alpha) + color[0] * alpha);
                imgRow[x][1] = static_cast<uchar>(
                    imgRow[x][1] * (1.0 - alpha) + color[1] * alpha);
                imgRow[x][2] = static_cast<uchar>(
                    imgRow[x][2] * (1.0 - alpha) + color[2] * alpha);
            }
        }
    }
}

void Visualizer::saveMotionOverlay(const std::string& sampleId,
                                    const std::string& route,
                                    const std::string& method,
                                    const cv::Mat& image,
                                    const cv::Mat& motionMask) {
    cv::Mat overlay = image.clone();
    addOverlay(overlay, motionMask, cv::Scalar(0, 255, 0), 0.4);

    cv::putText(overlay, "Route " + route + " - " + method,
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 255), 2);

    std::string fname = sampleOutputDir_ + "/" + sampleId +
                        "_route_" + route + "_" + method + ".png";
    cv::imwrite(fname, overlay);
}

// =====================================================================
//  Flow visualization
// =====================================================================

void Visualizer::saveFlowVis(const std::string& sampleId,
                              const std::string& route,
                              const cv::Mat& flowVis,
                              const cv::Mat& legend) {
    // Place flow image and legend side-by-side
    cv::Mat combined;
    if (!legend.empty()) {
        // Resize legend to match flowVis height
        double scale = static_cast<double>(flowVis.rows) / legend.rows;
        int lw = static_cast<int>(legend.cols * scale);
        cv::Mat legendR;
        cv::resize(legend, legendR, cv::Size(lw, flowVis.rows));

        cv::Mat temp;
        cv::hconcat(flowVis, legendR, temp);

        // Add title
        cv::Mat titled(temp.rows + 30, temp.cols, temp.type(), cv::Scalar(30, 30, 30));
        temp.copyTo(titled(cv::Rect(0, 30, temp.cols, temp.rows)));
        cv::putText(titled, "Route " + route + " - Farneback Optical Flow",
                    cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255), 2);
        combined = titled;
    } else {
        combined = flowVis;
    }

    std::string fname = sampleOutputDir_ + "/" + sampleId +
                        "_route_" + route + "_farneback_flow.png";
    cv::imwrite(fname, combined);
}

// =====================================================================
//  LK visualization
// =====================================================================

void Visualizer::saveLKVis(const std::string& sampleId,
                            const std::string& route,
                            const cv::Mat& lkVis) {
    cv::Mat out = lkVis.clone();
    cv::putText(out, "Route " + route + " - Lucas-Kanade Sparse Flow",
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 255), 2);

    std::string fname = sampleOutputDir_ + "/" + sampleId +
                        "_route_" + route + "_lk_sparse.png";
    cv::imwrite(fname, out);
}

// =====================================================================
//  Tracking visualization
// =====================================================================

void Visualizer::saveTrackingVis(const std::string& sampleId,
                                  const std::string& route,
                                  const cv::Mat& trackingImg) {
    cv::Mat out = trackingImg.clone();
    cv::putText(out, "Route " + route + " - Tracking",
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 255), 2);

    std::string fname = sampleOutputDir_ + "/" + sampleId +
                        "_route_" + route + "_tracking.png";
    cv::imwrite(fname, out);
}

// =====================================================================
//  Ground truth comparison
// =====================================================================

void Visualizer::saveGTComparison(const std::string& sampleId,
                                   const std::string& route,
                                   const std::string& method,
                                   const cv::Mat& image,
                                   const cv::Mat& predMask,
                                   const cv::Mat& gtMask,
                                   const EvalMetrics& fullMetrics) {
    // Create three panels: Predicted | Ground Truth | Error Map
    int panelH = std::min(image.rows, 400);
    double scale = static_cast<double>(panelH) / image.rows;
    int panelW = static_cast<int>(image.cols * scale);

    auto resizePanel = [&](const cv::Mat& src) {
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(panelW, panelH));
        return dst;
    };

    // Panel 1: Predicted (motion overlay on image)
    cv::Mat predPanel = resizePanel(image);
    cv::Mat predMaskR;
    cv::resize(predMask, predMaskR, cv::Size(panelW, panelH), 0, 0, cv::INTER_NEAREST);
    addOverlay(predPanel, predMaskR, cv::Scalar(0, 255, 0), 0.4);
    cv::putText(predPanel, "Predicted", cv::Point(5, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // Panel 2: Ground Truth (white mask on black background)
    cv::Mat gtPanel(panelH, panelW, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat gtMaskR;
    cv::resize(gtMask, gtMaskR, cv::Size(panelW, panelH), 0, 0, cv::INTER_NEAREST);
    for (int y = 0; y < panelH; y++) {
        for (int x = 0; x < panelW; x++) {
            if (gtMaskR.at<uchar>(y, x) > 127) {
                gtPanel.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            }
        }
    }
    cv::putText(gtPanel, "Ground Truth", cv::Point(5, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // Panel 3: Error map
    cv::Mat errMap = Evaluator::makeErrorMap(predMask, gtMask);
    cv::Mat errPanel = resizePanel(errMap);
    cv::putText(errPanel, "Error Map (G=TP,R=FP,B=FN)", cv::Point(5, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    // Concatenate three panels
    cv::Mat threePanels;
    cv::hconcat(predPanel, gtPanel, threePanels);
    cv::hconcat(threePanels, errPanel, threePanels);

    // Add metrics text at bottom
    int textH = 60;
    cv::Mat result(panelH + textH, threePanels.cols, CV_8UC3, cv::Scalar(20, 20, 20));
    threePanels.copyTo(result(cv::Rect(0, 0, threePanels.cols, threePanels.rows)));

    char buf[512];
    snprintf(buf, sizeof(buf),
             "Route %s %s | IoU=%.4f Prec=%.4f Rec=%.4f F1=%.4f | TP=%lld FP=%lld FN=%lld",
             route.c_str(), method.c_str(),
             fullMetrics.iou, fullMetrics.precision, fullMetrics.recall, fullMetrics.f1,
             fullMetrics.tp, fullMetrics.fp, fullMetrics.fn);
    cv::putText(result, buf, cv::Point(5, panelH + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);

    // Title
    cv::putText(result, "Route " + route + " (" + method + ") - Motion Detection Evaluation",
                cv::Point(5, panelH + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

    std::string fname = sampleOutputDir_ + "/" + sampleId +
                        "_route_" + route + "_" + method + "_gt_comparison.png";
    cv::imwrite(fname, result);
}


// =====================================================================
//  Aggregate charts
// =====================================================================

void Visualizer::drawBar(cv::Mat& canvas, int x, int y, int w, int h,
                          const cv::Scalar& color, const std::string& label,
                          double value, int maxH, int barW) {
    int barH = std::max(1, static_cast<int>(value / 0.1 * maxH));
    cv::rectangle(canvas,
                  cv::Point(x, y + maxH - barH),
                  cv::Point(x + barW, y + maxH),
                  color, -1);

    // Value label on top of bar
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", value);
    cv::putText(canvas, buf,
                cv::Point(x, y + maxH - barH - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);

    // X-axis label
    cv::putText(canvas, label,
                cv::Point(x, y + maxH + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);
}

void Visualizer::createComparisonCharts(const std::string& outputDir,
                                         const std::vector<EvalMetrics>& allMetrics) {
    if (allMetrics.empty()) return;

    // Aggregate: method x region x route -> mean IoU / F1
    struct Key {
        std::string method, region, route;
        bool operator<(const Key& o) const {
            if (method != o.method) return method < o.method;
            if (region != o.region) return region < o.region;
            return route < o.route;
        }
    };
    std::map<Key, std::vector<double>> iouMap, f1Map;

    for (const auto& m : allMetrics) {
        Key k{m.method, m.region, m.route};
        iouMap[k].push_back(m.iou);
        f1Map[k].push_back(m.f1);
    }

    // Group by method and region
    std::map<std::pair<std::string, std::string>, double> avgIouA, avgIouB, avgF1A, avgF1B;

    for (auto& [key, vals] : iouMap) {
        double avg = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        if (key.route == "A") avgIouA[{key.method, key.region}] = avg;
        else avgIouB[{key.method, key.region}] = avg;
    }
    for (auto& [key, vals] : f1Map) {
        double avg = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        if (key.route == "A") avgF1A[{key.method, key.region}] = avg;
        else avgF1B[{key.method, key.region}] = avg;
    }

    // Build sorted list of categories
    std::vector<std::string> methods = {"FD", "FB"};
    std::vector<std::string> regions = {"full", "center", "edge"};
    std::vector<std::string> catLabels;
    for (const auto& m : methods) {
        for (const auto& r : regions) {
            std::string rShort = (r == "full") ? "Full" : (r == "center") ? "Center" : "Edge";
            catLabels.push_back(m + "-" + rShort);
        }
    }

    // Draw IoU chart
    {
        int w = 1000, h = 500;
        cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(40, 40, 40));
        int margin = 80, barAreaW = w - 2 * margin, barAreaH = h - 120;
        int nCats = static_cast<int>(catLabels.size());
        int groupW = barAreaW / nCats;
        int barW = groupW / 4;

        cv::putText(canvas, "Route A vs Route B - IoU Comparison",
                    cv::Point(margin, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(255, 255, 255), 2);

        for (int i = 0; i < nCats; i++) {
            int x = margin + i * groupW;
            std::string m = (i < 3) ? "frame_diff" : "farneback";
            std::string r = regions[i % 3];

            double vA = avgIouA[{m, r}];
            double vB = avgIouB[{m, r}];

            drawBar(canvas, x, margin, barW, vA, cv::Scalar(0, 200, 0),
                    "", vA, barAreaH, barW);
            drawBar(canvas, x + barW + 2, margin, barW, vB, cv::Scalar(0, 140, 255),
                    catLabels[i], vB, barAreaH, barW);
        }

        // Legend
        cv::rectangle(canvas, cv::Point(w - 250, h - 50), cv::Point(w - 230, h - 35),
                      cv::Scalar(0, 200, 0), -1);
        cv::putText(canvas, "Route A (fisheye)", cv::Point(w - 225, h - 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        cv::rectangle(canvas, cv::Point(w - 250, h - 25), cv::Point(w - 230, h - 10),
                      cv::Scalar(0, 140, 255), -1);
        cv::putText(canvas, "Route B (rectified)", cv::Point(w - 225, h - 7),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        cv::imwrite(outputDir + "/chart_iou_comparison.png", canvas);
    }

    // Draw F1 chart (same layout)
    {
        int w = 1000, h = 500;
        cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(40, 40, 40));
        int margin = 80, barAreaW = w - 2 * margin, barAreaH = h - 120;
        int nCats = static_cast<int>(catLabels.size());
        int groupW = barAreaW / nCats;
        int barW = groupW / 4;

        cv::putText(canvas, "Route A vs Route B - F1 Comparison",
                    cv::Point(margin, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(255, 255, 255), 2);

        for (int i = 0; i < nCats; i++) {
            int x = margin + i * groupW;
            std::string m = (i < 3) ? "frame_diff" : "farneback";
            std::string r = regions[i % 3];

            double vA = avgF1A[{m, r}];
            double vB = avgF1B[{m, r}];

            drawBar(canvas, x, margin, barW, vA, cv::Scalar(0, 200, 0),
                    "", vA, barAreaH, barW);
            drawBar(canvas, x + barW + 2, margin, barW, vB, cv::Scalar(0, 140, 255),
                    catLabels[i], vB, barAreaH, barW);
        }

        // Legend
        cv::rectangle(canvas, cv::Point(w - 250, h - 50), cv::Point(w - 230, h - 35),
                      cv::Scalar(0, 200, 0), -1);
        cv::putText(canvas, "Route A (fisheye)", cv::Point(w - 225, h - 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        cv::rectangle(canvas, cv::Point(w - 250, h - 25), cv::Point(w - 230, h - 10),
                      cv::Scalar(0, 140, 255), -1);
        cv::putText(canvas, "Route B (rectified)", cv::Point(w - 225, h - 7),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        cv::imwrite(outputDir + "/chart_f1_comparison.png", canvas);
    }
}

void Visualizer::createCenterVsEdgeChart(const std::string& outputDir,
                                          const std::vector<EvalMetrics>& allMetrics) {
    if (allMetrics.empty()) return;

    // Collect center vs edge IoU pairs for each sample (using frame_diff + Route A)
    std::vector<std::pair<double, double>> points;

    for (size_t i = 0; i < allMetrics.size(); i++) {
        const auto& m = allMetrics[i];
        if (m.method == "frame_diff" && m.route == "A" && m.region == "center") {
            // Find corresponding edge metric
            for (size_t j = 0; j < allMetrics.size(); j++) {
                const auto& e = allMetrics[j];
                if (e.sampleId == m.sampleId && e.method == m.method &&
                    e.route == m.route && e.region == "edge") {
                    points.push_back({m.iou, e.iou});
                    break;
                }
            }
        }
    }

    if (points.empty()) return;

    // Determine range
    double maxVal = 0.0;
    for (const auto& p : points) {
        maxVal = std::max(maxVal, std::max(p.first, p.second));
    }
    maxVal = std::max(maxVal, 0.1);  // at least 0.1

    int size = 600, margin = 70;
    int plotSize = size - 2 * margin;
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(40, 40, 40));

    // Axes
    cv::line(canvas, cv::Point(margin, size - margin), cv::Point(size - margin, size - margin),
             cv::Scalar(200, 200, 200), 1);
    cv::line(canvas, cv::Point(margin, size - margin), cv::Point(margin, margin),
             cv::Scalar(200, 200, 200), 1);

    // Diagonal y=x (dashed)
    cv::line(canvas, cv::Point(margin, size - margin), cv::Point(size - margin, margin),
             cv::Scalar(100, 100, 100), 1, cv::LINE_AA);

    // Plot points
    for (const auto& p : points) {
        int px = margin + static_cast<int>(p.first / maxVal * plotSize);
        int py = size - margin - static_cast<int>(p.second / maxVal * plotSize);
        cv::circle(canvas, cv::Point(px, py), 2, cv::Scalar(0, 255, 0), -1);
    }

    // Labels
    cv::putText(canvas, "Center IoU", cv::Point(size / 2 - 40, size - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    cv::putText(canvas, "Edge IoU", cv::Point(10, size / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);

    cv::putText(canvas, "y=x (parity line)", cv::Point(size - margin - 150, margin + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

    cv::putText(canvas, "Center vs Edge - IoU Scatter (FrameDiff, Route A)",
                cv::Point(margin, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(255, 255, 255), 2);

    // Tick marks
    for (int i = 0; i <= 4; i++) {
        double val = maxVal * i / 4.0;
        int x = margin + static_cast<int>((double)i / 4 * plotSize);
        int y = size - margin - static_cast<int>((double)i / 4 * plotSize);
        char buf[16];
        snprintf(buf, sizeof(buf), "%.2f", val);

        cv::putText(canvas, buf, cv::Point(x - 15, size - margin + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
        cv::putText(canvas, buf, cv::Point(margin - 45, y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
    }

    cv::imwrite(outputDir + "/chart_center_vs_edge.png", canvas);
}

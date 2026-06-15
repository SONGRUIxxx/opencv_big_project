#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "evaluation.hpp"

/**
 * @brief Generates all visualization outputs for the experiment.
 */
class Visualizer {
public:
    /**
     * @brief Set output directory for this sample's visualizations.
     */
    static void setSampleOutputDir(const std::string& dir);

    // ---- Single-image outputs ----

    /** Save undistorted comparison: left=fisheye, right=perspective */
    static void saveUndistortComparison(const std::string& sampleId,
                                         const cv::Mat& fisheye,
                                         const cv::Mat& perspective,
                                         const std::string& suffix = "");

    /** Save motion overlay: original image + green semi-transparent motion mask */
    static void saveMotionOverlay(const std::string& sampleId,
                                   const std::string& route,
                                   const std::string& method,
                                   const cv::Mat& image,
                                   const cv::Mat& motionMask);

    /** Save optical flow visualization with legend */
    static void saveFlowVis(const std::string& sampleId,
                             const std::string& route,
                             const cv::Mat& flowVis,
                             const cv::Mat& legend);

    /** Save sparse LK visualization */
    static void saveLKVis(const std::string& sampleId,
                           const std::string& route,
                           const cv::Mat& lkVis);

    /** Save tracking visualization */
    static void saveTrackingVis(const std::string& sampleId,
                                 const std::string& route,
                                 const cv::Mat& trackingImg);

    /** Save ground-truth comparison: pred | gt | error map */
    static void saveGTComparison(const std::string& sampleId,
                                  const std::string& route,
                                  const std::string& method,
                                  const cv::Mat& image,
                                  const cv::Mat& predMask,
                                  const cv::Mat& gtMask,
                                  const EvalMetrics& fullMetrics);

    // ---- Aggregate chart outputs ----

    /** Create Route A vs Route B bar charts (IoU, F1) */
    static void createComparisonCharts(
        const std::string& outputDir,
        const std::vector<EvalMetrics>& allMetrics);

    /** Create center-vs-edge IoU scatter plot */
    static void createCenterVsEdgeChart(
        const std::string& outputDir,
        const std::vector<EvalMetrics>& allMetrics);

    /** Helper: add semi-transparent colored overlay to image */
    static void addOverlay(cv::Mat& image, const cv::Mat& mask,
                           const cv::Scalar& color, double alpha = 0.4);

private:
    static std::string sampleOutputDir_;

    /** Helper: draw a single bar in a chart */
    static void drawBar(cv::Mat& canvas, int x, int y, int w, int h,
                        const cv::Scalar& color, const std::string& label,
                        double value, int maxH, int barW);
};

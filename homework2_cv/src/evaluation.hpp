#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <fstream>

/**
 * @brief Per-sample evaluation metrics.
 */
struct EvalMetrics {
    std::string sampleId;
    std::string route;       // "A" or "B"
    std::string method;      // "frame_diff", "farneback", "lk"
    std::string region;      // "full", "center", "edge"

    long long tp = 0;
    long long fp = 0;
    long long fn = 0;
    long long tn = 0;

    double iou = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

/**
 * @brief Evaluator for motion detection results.
 *
 * Supports center/edge region analysis based on distance from principal point.
 */
class Evaluator {
public:
    /**
     * @param cx Principal point x (fisheye image space)
     * @param cy Principal point y
     * @param maxR Maximum radius from principal point to corner
     * @param centerThresh Center region: distance < centerThresh * maxR
     * @param edgeThresh   Edge region:   distance > edgeThresh * maxR
     */
    Evaluator(double cx, double cy, double maxR,
              double centerThresh = 0.40,
              double edgeThresh = 0.60);

    /**
     * @brief Compute metrics for a single (predicted, ground-truth) mask pair.
     *
     * @param predMask  Predicted binary motion mask (CV_8UC1, 0/255)
     * @param gtMask    Ground truth binary motion mask (CV_8UC1, 0/255)
     * @param sampleId  Sample identifier
     * @param route     "A" or "B"
     * @param method    Algorithm name
     * @return Vector of 3 EvalMetrics: full, center, edge
     */
    std::vector<EvalMetrics> evaluate(const cv::Mat& predMask,
                                       const cv::Mat& gtMask,
                                       const std::string& sampleId,
                                       const std::string& route,
                                       const std::string& method) const;

    /**
     * @brief Create an error map (color-coded TP/FP/FN).
     *        Green=TP, Red=FP, Blue=FN, Black=TN
     */
    static cv::Mat makeErrorMap(const cv::Mat& predMask, const cv::Mat& gtMask);

    /**
     * @brief Write all metrics to a CSV file.
     */
    static void writeResultsCSV(const std::string& path,
                                const std::vector<EvalMetrics>& allMetrics);

private:
    double cx_, cy_, maxR_;
    double centerThresh_, edgeThresh_;

    /** Generate a per-pixel mask for center or edge region. */
    cv::Mat regionMask(const cv::Size& size, const std::string& region) const;

    /** Compute TP/FP/FN/TN and derived metrics for a masked region. */
    EvalMetrics computeRegion(const cv::Mat& predMask,
                              const cv::Mat& gtMask,
                              const cv::Mat& rMask,
                              const std::string& sampleId,
                              const std::string& route,
                              const std::string& method,
                              const std::string& region) const;
};

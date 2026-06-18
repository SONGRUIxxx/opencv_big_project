#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <string>

/**
 * @brief Result of frame differencing.
 */
struct FrameDiffResult {
    cv::Mat motionMask;      // binary mask (255=motion, 0=static), CV_8UC1
    double motionRatio;      // fraction of pixels classified as motion
    cv::Mat diffGray;        // raw absolute difference (for visualization)
};

/**
 * @brief Result of Farneback dense optical flow.
 */
struct FarnebackResult {
    cv::Mat flow;            // CV_32FC2, (vx, vy) per pixel
    cv::Mat motionMask;      // binary mask where |flow| > threshold
    cv::Mat flowVis;         // HSV color-coded flow visualization (BGR)
    cv::Mat flowVisLegend;   // color wheel legend (BGR)
    double motionRatio;
};

/**
 * @brief Result of Lucas-Kanade sparse optical flow.
 */
struct LKResult {
    std::vector<cv::Point2f> ptsPrev;     // feature points in previous frame
    std::vector<cv::Point2f> ptsCurr;     // tracked points in current frame
    std::vector<uchar> status;            // tracking status (1=success)
    std::vector<float> err;               // tracking error
    cv::Mat visualization;                // BGR image with arrows
};

/**
 * @brief Result of unified motion detection pipeline (ECC + fusion).
 */
struct UnifiedMotionResult {
    cv::Mat motionMask;         // final binary mask
    cv::Mat flowVis;            // HSV flow visualization
    cv::Mat flowVisLegend;      // color wheel legend
    cv::Mat flow;               // raw Farneback flow (CV_32FC2)
    cv::Mat diffGray;           // raw frame difference
    cv::Mat flowResidualMask;   // median-flow residual mask
    cv::Mat diffMask;           // cleaned frame-diff mask
    double motionRatio = 0.0;
};

/**
 * @brief Motion extraction algorithms operating on a pair of grayscale images.
 */
class MotionExtractor {
public:
    // ---- Parameters (configurable) ----

    // Frame difference
    int fdBlurKernel = 5;
    int fdThreshold = 25;
    int fdOpenKernel = 3;
    int fdCloseKernel = 5;

    // Farneback optical flow
    double fbPyrScale = 0.5;
    int fbLevels = 3;
    int fbWinSize = 15;
    int fbIterations = 3;
    int fbPolyN = 5;
    double fbPolySigma = 1.2;
    double fbMagThreshold = 2.0;  // minimum flow magnitude for motion mask

    // Lucas-Kanade
    int lkMaxCorners = 500;
    double lkQualityLevel = 0.01;
    double lkMinDistance = 10.0;
    int lkWinSize = 21;
    int lkMaxLevel = 3;
    double lkMinDisplacement = 2.0;

    // ---- ECC global motion compensation ----
    int eccMaxIter = 60;
    double eccEps = 1e-5;

    // ---- Median-flow residual ----
    double mfResidualThreshold = 4.0;  // threshold for flow residual magnitude

    // ---- Alignment mode ----
    std::string alignMode = "ecc";  // "ecc", "feature", "none"

    // ---- Fusion strategy ----
    std::string fusionMode = "clean-and";  // "clean-and", "raw-and", "clean-or"

    // ---- Post-processing ----
    int morphSize = 5;
    int minArea = 600;

    // ---- Frame Difference ----
    FrameDiffResult frameDifference(const cv::Mat& grayPrev,
                                    const cv::Mat& grayCurr) const;

    // ---- Farneback Dense Optical Flow ----
    FarnebackResult farnebackFlow(const cv::Mat& grayPrev,
                                  const cv::Mat& grayCurr) const;

    // ---- Lucas-Kanade Sparse Optical Flow ----
    LKResult lucasKanadeSparse(const cv::Mat& grayPrev,
                               const cv::Mat& grayCurr) const;

    // ---- ECC Alignment ----
    /** Align curr to prev using ECC affine transform, return aligned curr. */
    cv::Mat alignECC(const cv::Mat& prevGray, const cv::Mat& currGray) const;

    /** Align curr to prev using ORB feature matching + RANSAC affine, return aligned curr. */
    cv::Mat alignFeature(const cv::Mat& prevGray, const cv::Mat& currGray) const;

    /** Align curr to prev using specified mode ("ecc", "feature", "none"). */
    cv::Mat alignCurrentToPrevious(const cv::Mat& prevGray, const cv::Mat& currGray,
                                   const std::string& mode) const;

    // ---- Median Flow Residual ----
    /** Compute global median flow vector (for ego-motion estimation). */
    static cv::Point2f medianFlow(const cv::Mat& flow, const cv::Mat& valid = cv::Mat());

    /** Compute flow residual mask by subtracting global flow. */
    cv::Mat flowResidualMask(const cv::Mat& flow, const cv::Mat& valid = cv::Mat()) const;

    // ---- Mask Cleanup ----
    /** Morphological cleanup + min-area filtering. */
    cv::Mat cleanupMask(const cv::Mat& mask) const;

    // ---- Fusion ----
    /** Combine frame-diff mask and flow-residual mask according to fusionMode. */
    cv::Mat fuseMasks(const cv::Mat& diffMask, const cv::Mat& flowMask,
                      const cv::Mat& diffRaw, const cv::Mat& flowRaw,
                      const cv::Mat& valid = cv::Mat()) const;

    // ---- Unified Pipeline ----
    /** Run the unified motion detection pipeline (ECC + fusion). */
    UnifiedMotionResult detectMotion(const cv::Mat& prevGray, const cv::Mat& currGray,
                                     const cv::Mat& valid = cv::Mat()) const;

private:
    /** Create a color wheel legend for flow visualization. */
    static cv::Mat makeColorWheel(int size = 100);
};

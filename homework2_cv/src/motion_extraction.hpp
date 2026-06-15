#pragma once

#include <opencv2/core.hpp>
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

    // ---- Frame Difference ----
    FrameDiffResult frameDifference(const cv::Mat& grayPrev,
                                    const cv::Mat& grayCurr) const;

    // ---- Farneback Dense Optical Flow ----
    FarnebackResult farnebackFlow(const cv::Mat& grayPrev,
                                  const cv::Mat& grayCurr) const;

    // ---- Lucas-Kanade Sparse Optical Flow ----
    LKResult lucasKanadeSparse(const cv::Mat& grayPrev,
                               const cv::Mat& grayCurr) const;

private:
    /** Create a color wheel legend for flow visualization. */
    static cv::Mat makeColorWheel(int size = 100);
};

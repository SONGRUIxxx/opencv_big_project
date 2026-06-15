#include "motion_extraction.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =====================================================================
//  Frame Difference
// =====================================================================

FrameDiffResult MotionExtractor::frameDifference(const cv::Mat& grayPrev,
                                                  const cv::Mat& grayCurr) const
{
    FrameDiffResult result;

    // Step 1: Gaussian blur to suppress sensor noise
    cv::Mat blurredPrev, blurredCurr;
    cv::GaussianBlur(grayPrev, blurredPrev, cv::Size(fdBlurKernel, fdBlurKernel), 0);
    cv::GaussianBlur(grayCurr, blurredCurr, cv::Size(fdBlurKernel, fdBlurKernel), 0);

    // Step 2: Absolute difference
    cv::Mat diff;
    cv::absdiff(blurredPrev, blurredCurr, diff);
    result.diffGray = diff.clone();

    // Step 3: Fixed threshold
    cv::Mat binMask;
    cv::threshold(diff, binMask, fdThreshold, 255, cv::THRESH_BINARY);
    binMask.convertTo(binMask, CV_8UC1);

    // Step 4: Morphological opening (remove isolated noise)
    cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                    cv::Size(fdOpenKernel, fdOpenKernel));
    cv::morphologyEx(binMask, binMask, cv::MORPH_OPEN, kernelOpen);

    // Step 5: Morphological closing (fill holes in motion regions)
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                     cv::Size(fdCloseKernel, fdCloseKernel));
    cv::morphologyEx(binMask, binMask, cv::MORPH_CLOSE, kernelClose);

    result.motionMask = binMask;
    result.motionRatio = static_cast<double>(cv::countNonZero(binMask)) /
                         (binMask.rows * binMask.cols);

    return result;
}


// =====================================================================
//  Farneback Dense Optical Flow
// =====================================================================

FarnebackResult MotionExtractor::farnebackFlow(const cv::Mat& grayPrev,
                                                const cv::Mat& grayCurr) const
{
    FarnebackResult result;

    // Compute dense optical flow
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(grayPrev, grayCurr, flow,
                                 fbPyrScale, fbLevels, fbWinSize,
                                 fbIterations, fbPolyN, fbPolySigma, 0);
    result.flow = flow;

    // Compute magnitude and create motion mask
    std::vector<cv::Mat> channels(2);
    cv::split(flow, channels);
    cv::Mat mag, ang;
    cv::cartToPolar(channels[0], channels[1], mag, ang, true);

    // Threshold: pixels with flow magnitude > threshold are "moving"
    cv::Mat motionMask;
    cv::threshold(mag, motionMask, fbMagThreshold, 255, cv::THRESH_BINARY);
    motionMask.convertTo(motionMask, CV_8UC1);
    result.motionMask = motionMask;
    result.motionRatio = static_cast<double>(cv::countNonZero(motionMask)) /
                         (motionMask.rows * motionMask.cols);

    // Color-code flow for visualization (HSV color wheel)
    // Hue = direction, Saturation = 255, Value = magnitude (clamped)
    cv::Mat hsv(mag.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            float m = mag.at<float>(y, x);
            if (m >= fbMagThreshold) {
                float a = ang.at<float>(y, x);  // degrees, 0..360
                hsv.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(a / 2.0);  // H: 0..180
                hsv.at<cv::Vec3b>(y, x)[1] = 255;                          // S: full
                // V: clamp magnitude to [0, 255] for visualization
                float v = std::min(m * 10.0f, 255.0f);
                hsv.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(v);
            }
        }
    }
    cv::cvtColor(hsv, result.flowVis, cv::COLOR_HSV2BGR);

    // Create color wheel legend
    result.flowVisLegend = makeColorWheel(120);

    return result;
}


// =====================================================================
//  Lucas-Kanade Sparse Optical Flow
// =====================================================================

LKResult MotionExtractor::lucasKanadeSparse(const cv::Mat& grayPrev,
                                             const cv::Mat& grayCurr) const
{
    LKResult result;

    // Detect good features to track (Shi-Tomasi corners) in previous frame
    cv::goodFeaturesToTrack(grayPrev, result.ptsPrev, lkMaxCorners,
                            lkQualityLevel, lkMinDistance);
    if (result.ptsPrev.empty()) {
        return result;
    }

    // Lucas-Kanade tracking
    result.ptsCurr.resize(result.ptsPrev.size());
    result.status.resize(result.ptsPrev.size());
    result.err.resize(result.ptsPrev.size());

    cv::calcOpticalFlowPyrLK(grayPrev, grayCurr,
                             result.ptsPrev, result.ptsCurr,
                             result.status, result.err,
                             cv::Size(lkWinSize, lkWinSize),
                             lkMaxLevel);

    // Draw visualization on a color version of the previous frame
    // We create a BGR canvas (the caller can overlay on actual image if needed)
    cv::Mat vis(grayPrev.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::cvtColor(grayPrev, vis, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < result.ptsPrev.size(); i++) {
        if (result.status[i] == 0) continue;

        double dx = result.ptsCurr[i].x - result.ptsPrev[i].x;
        double dy = result.ptsCurr[i].y - result.ptsPrev[i].y;
        double dist = std::sqrt(dx * dx + dy * dy);

        if (dist < lkMinDisplacement) continue;  // filter static points

        // Draw feature point (green dot)
        cv::circle(vis, result.ptsPrev[i], 3, cv::Scalar(0, 255, 0), -1);

        // Draw motion arrow (yellow, scaled 3x for visibility)
        cv::Point2f tip(result.ptsPrev[i].x + static_cast<float>(dx * 3.0),
                        result.ptsPrev[i].y + static_cast<float>(dy * 3.0));
        cv::arrowedLine(vis, result.ptsPrev[i], tip,
                        cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0, 0.2);
    }

    result.visualization = vis;
    return result;
}


// =====================================================================
//  Color Wheel Legend for Optical Flow
// =====================================================================

cv::Mat MotionExtractor::makeColorWheel(int size) {
    cv::Mat wheel(size, size, CV_8UC3);
    int center = size / 2;
    int radius = size / 2 - 5;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            double dx = x - center;
            double dy = y - center;
            double dist = std::sqrt(dx * dx + dy * dy);

            if (dist <= radius) {
                double angle = std::atan2(dy, dx) * 180.0 / M_PI + 180.0; // 0..360
                uchar hue = static_cast<uchar>(angle / 2.0);  // 0..180
                uchar sat = 255;
                uchar val = static_cast<uchar>(std::min(dist / radius * 255.0, 255.0));

                cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, sat, val));
                cv::Mat bgr;
                cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
                wheel.at<cv::Vec3b>(y, x) = bgr.at<cv::Vec3b>(0, 0);
            } else {
                wheel.at<cv::Vec3b>(y, x) = cv::Vec3b(30, 30, 30);
            }
        }
    }

    // Add direction labels
    std::vector<std::pair<std::string, cv::Point>> labels = {
        {"R", {center + radius - 10, center + 5}},
        {"L", {center - radius + 2, center + 5}},
        {"D", {center - 10, center + radius - 5}},
        {"U", {center - 10, center - radius + 12}},
    };
    for (const auto& lbl : labels) {
        cv::putText(wheel, lbl.first, lbl.second,
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }

    return wheel;
}

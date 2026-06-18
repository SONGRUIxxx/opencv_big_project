#include "motion_extraction.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <cmath>
#include <algorithm>

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
//  ECC Global Motion Compensation
// =====================================================================

cv::Mat MotionExtractor::alignECC(const cv::Mat& prevGray, const cv::Mat& currGray) const
{
    cv::Mat prevSmall, currSmall;
    constexpr double scale = 0.5;
    cv::resize(prevGray, prevSmall, {}, scale, scale, cv::INTER_AREA);
    cv::resize(currGray, currSmall, {}, scale, scale, cv::INTER_AREA);

    cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
    const cv::TermCriteria criteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS, eccMaxIter, eccEps);

    try {
        cv::findTransformECC(prevSmall, currSmall, warp, cv::MOTION_AFFINE, criteria);
    } catch (const cv::Exception&) {
        return currGray.clone();
    }

    cv::Mat warpFull = warp.clone();
    warpFull.at<float>(0, 2) = warp.at<float>(0, 2) / static_cast<float>(scale);
    warpFull.at<float>(1, 2) = warp.at<float>(1, 2) / static_cast<float>(scale);

    cv::Mat aligned;
    cv::warpAffine(currGray, aligned, warpFull, currGray.size(),
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_REPLICATE);
    return aligned;
}

cv::Mat MotionExtractor::alignFeature(const cv::Mat& prevGray, const cv::Mat& currGray) const
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1600);
    std::vector<cv::KeyPoint> prevKpts, currKpts;
    cv::Mat prevDesc, currDesc;

    orb->detectAndCompute(prevGray, cv::noArray(), prevKpts, prevDesc);
    orb->detectAndCompute(currGray, cv::noArray(), currKpts, currDesc);

    if (prevDesc.empty() || currDesc.empty() || prevKpts.size() < 12 || currKpts.size() < 12)
        return currGray.clone();

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(prevDesc, currDesc, matches);
    if (matches.size() < 12)
        return currGray.clone();

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
    if (matches.size() > 220) matches.resize(220);

    std::vector<cv::Point2f> prevPts, currPts;
    for (const auto& m : matches) {
        prevPts.push_back(prevKpts[m.queryIdx].pt);
        currPts.push_back(currKpts[m.trainIdx].pt);
    }

    cv::Mat inliers;
    cv::Mat affine = cv::estimateAffinePartial2D(
        currPts, prevPts, inliers, cv::RANSAC, 3.0, 2000, 0.99, 10);
    if (affine.empty()) return currGray.clone();

    int inlierCount = inliers.empty() ? 0 : cv::countNonZero(inliers);
    if (inlierCount < 10) return currGray.clone();

    cv::Mat aligned;
    cv::warpAffine(currGray, aligned, affine, currGray.size(),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return aligned;
}

cv::Mat MotionExtractor::alignCurrentToPrevious(const cv::Mat& prevGray, const cv::Mat& currGray,
                                                  const std::string& mode) const
{
    if (mode == "none") return currGray.clone();
    if (mode == "feature") return alignFeature(prevGray, currGray);
    return alignECC(prevGray, currGray);  // default: ecc
}

// =====================================================================
//  Median Flow Residual
// =====================================================================

cv::Point2f MotionExtractor::medianFlow(const cv::Mat& flow, const cv::Mat& valid)
{
    std::vector<float> xs, ys;
    xs.reserve(static_cast<size_t>(flow.total() / 10));
    ys.reserve(static_cast<size_t>(flow.total() / 10));

    for (int y = 0; y < flow.rows; y += 3) {
        const cv::Point2f* flowRow = flow.ptr<cv::Point2f>(y);
        const uchar* validRow = valid.empty() ? nullptr : valid.ptr<uchar>(y);
        for (int x = 0; x < flow.cols; x += 3) {
            if (validRow && validRow[x] == 0) continue;
            const cv::Point2f v = flowRow[x];
            if (std::isfinite(v.x) && std::isfinite(v.y) &&
                std::abs(v.x) < 100.0F && std::abs(v.y) < 100.0F) {
                xs.push_back(v.x);
                ys.push_back(v.y);
            }
        }
    }

    if (xs.empty()) return {0.0F, 0.0F};

    auto mx = xs.begin() + static_cast<long>(xs.size() / 2);
    auto my = ys.begin() + static_cast<long>(ys.size() / 2);
    std::nth_element(xs.begin(), mx, xs.end());
    std::nth_element(ys.begin(), my, ys.end());
    return {*mx, *my};
}

cv::Mat MotionExtractor::flowResidualMask(const cv::Mat& flow, const cv::Mat& valid) const
{
    const cv::Point2f global = medianFlow(flow, valid);
    cv::Mat residual(flow.size(), CV_32FC1, cv::Scalar(0));
    for (int y = 0; y < flow.rows; ++y) {
        const cv::Point2f* frow = flow.ptr<cv::Point2f>(y);
        float* rrow = residual.ptr<float>(y);
        for (int x = 0; x < flow.cols; ++x) {
            const cv::Point2f v = frow[x] - global;
            rrow[x] = std::sqrt(v.x * v.x + v.y * v.y);
        }
    }

    cv::Mat mask;
    cv::threshold(residual, mask, mfResidualThreshold, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
    if (!valid.empty())
        cv::bitwise_and(mask, valid, mask);
    return mask;
}

// =====================================================================
//  Mask Cleanup
// =====================================================================

cv::Mat MotionExtractor::cleanupMask(const cv::Mat& mask) const
{
    int ksize = std::max(3, morphSize);
    if (ksize % 2 == 0) ++ksize;
    cv::Mat cleaned = mask.clone();
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {ksize, ksize});
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel, {-1, -1}, 2);
    cv::dilate(cleaned, cleaned, kernel, {-1, -1}, 1);

    // Min-area filtering via connected components
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8);
    cv::Mat filtered(cleaned.size(), CV_8UC1, cv::Scalar(0));
    for (int i = 1; i < n; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minArea)
            filtered.setTo(255, labels == i);
    }
    return filtered;
}

// =====================================================================
//  Fusion
// =====================================================================

cv::Mat MotionExtractor::fuseMasks(const cv::Mat& diffMask, const cv::Mat& flowMask,
                                     const cv::Mat& diffRaw, const cv::Mat& flowRaw,
                                     const cv::Mat& valid) const
{
    cv::Mat combined;
    if (fusionMode == "raw-and") {
        cv::bitwise_and(diffRaw, flowRaw, combined);
    } else if (fusionMode == "clean-or") {
        cv::bitwise_or(diffMask, flowMask, combined);
    } else {
        // Default clean-and: intersection of cleaned masks
        cv::bitwise_and(diffMask, flowMask, combined);
    }
    if (!valid.empty())
        cv::bitwise_and(combined, valid, combined);
    return combined;
}

// =====================================================================
//  Unified Motion Detection Pipeline
// =====================================================================

UnifiedMotionResult
MotionExtractor::detectMotion(const cv::Mat& prevGray, const cv::Mat& currGray,
                               const cv::Mat& valid) const
{
    UnifiedMotionResult result;

    // Step 1: Gaussian denoising
    cv::Mat prevBlur, currBlur;
    cv::GaussianBlur(prevGray, prevBlur, {5, 5}, 0.0);
    cv::GaussianBlur(currGray, currBlur, {5, 5}, 0.0);

    // Step 2: Global motion compensation (ECC / feature / none)
    cv::Mat currAligned = alignCurrentToPrevious(prevBlur, currBlur, alignMode);

    // Step 3: Frame difference
    cv::absdiff(prevBlur, currAligned, result.diffGray);
    cv::Mat diffRaw;
    cv::threshold(result.diffGray, diffRaw, fdThreshold, 255, cv::THRESH_BINARY);
    if (!valid.empty()) cv::bitwise_and(diffRaw, valid, diffRaw);
    result.diffMask = cleanupMask(diffRaw);

    // Step 4: Farneback dense optical flow
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prevBlur, currAligned, flow,
                                  fbPyrScale, fbLevels, fbWinSize,
                                  fbIterations, fbPolyN, fbPolySigma, 0);
    result.flow = flow;

    // Flow visualization (HSV color coding)
    std::vector<cv::Mat> flowCh(2);
    cv::split(flow, flowCh);
    cv::Mat mag, ang;
    cv::cartToPolar(flowCh[0], flowCh[1], mag, ang, true);
    cv::Mat hsv(mag.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            float m = mag.at<float>(y, x);
            if (m >= fbMagThreshold) {
                hsv.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(ang.at<float>(y, x) / 2.0);
                hsv.at<cv::Vec3b>(y, x)[1] = 255;
                hsv.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(std::min(m * 10.0f, 255.0f));
            }
        }
    }
    cv::cvtColor(hsv, result.flowVis, cv::COLOR_HSV2BGR);
    result.flowVisLegend = makeColorWheel(120);

    // Step 5: Median-flow residual
    cv::Mat flowRaw = flowResidualMask(flow, valid);
    result.flowResidualMask = cleanupMask(flowRaw);

    // Step 6: Fuse masks
    cv::Mat combined = fuseMasks(result.diffMask, result.flowResidualMask,
                                  diffRaw, flowRaw, valid);
    result.motionMask = cleanupMask(combined);
    result.motionRatio = static_cast<double>(cv::countNonZero(result.motionMask)) /
                         (result.motionMask.rows * result.motionMask.cols);

    return result;
}

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

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

/**
 * @brief Information about one tracked motion region.
 */
struct TrackedRegion {
    cv::Rect bbox;              // axis-aligned bounding box
    cv::Point2d centroid;       // region centroid
    double area;                // pixel area
    int id;                     // region ID (from connected components)

    // Kalman filter state
    cv::KalmanFilter kf;
    bool kfInitialized = false;

    // Estimated velocity (from optical flow)
    cv::Point2d velocity;
};

/**
 * @brief Simple motion-region tracker using connected components + Kalman filter.
 *
 * In the 2-frame scenario described in the report, Kalman filtering mainly
 * serves to smooth measurement noise rather than to predict future positions.
 */
class MotionTracker {
public:
    int minArea = 100;          // minimum area to consider as valid region
    double maxAreaRatio = 0.5;  // reject regions larger than this fraction of image

    /**
     * @brief Extract connected regions from a binary motion mask.
     *
     * @param motionMask  Binary mask (255=motion, 0=static), CV_8UC1
     * @param flow        Optional dense optical flow (CV_32FC2) for velocity estimation.
     *                    Pass empty cv::Mat() to skip.
     * @return List of tracked regions.
     */
    std::vector<TrackedRegion> extractRegions(const cv::Mat& motionMask,
                                               const cv::Mat& flow = cv::Mat());

    /**
     * @brief Draw tracking visualization on an image.
     *
     * @param image    BGR image to draw on (modified in-place)
     * @param regions  List of tracked regions
     * @param scaleArrow  Scale factor for velocity arrows (default 3.0)
     */
    static void drawTracking(cv::Mat& image,
                             const std::vector<TrackedRegion>& regions,
                             double scaleArrow = 3.0);
};

#include "tracking.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

// =====================================================================
//  Object-like component filtering
// =====================================================================

cv::Mat MotionTracker::filterObjectLikeComponents(const cv::Mat& mask, int minArea)
{
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    cv::Mat filtered(mask.size(), CV_8UC1, cv::Scalar(0));

    const double imageArea = static_cast<double>(mask.rows * mask.cols);
    for (int i = 1; i < n; ++i) {
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        double cy = centroids.at<double>(i, 1);
        double fillRatio = static_cast<double>(area) / std::max(1.0, static_cast<double>(w * h));

        if (area < minArea) continue;
        // Too large and sparse → likely background texture
        if (area > 0.16 * imageArea && fillRatio < 0.45) continue;
        if (w > 0.42 * mask.cols && fillRatio < 0.35) continue;
        if (h > 0.62 * mask.rows && fillRatio < 0.35) continue;
        // Too sparse → not object-like
        if (fillRatio < 0.22) continue;
        // Centroid outside plausible road region
        if (cy < 0.24 * mask.rows || cy > 0.68 * mask.rows) continue;
        // Too close to bottom edge (likely car hood / self)
        if (y > 0.82 * mask.rows) continue;

        filtered.setTo(255, labels == i);
    }
    return filtered;
}

std::vector<TrackedRegion> MotionTracker::extractRegions(const cv::Mat& motionMask,
                                                          const cv::Mat& flow) {
    std::vector<TrackedRegion> regions;

    // Connected components with stats (8-connectivity)
    cv::Mat labels, stats, centroids;
    int nComponents = cv::connectedComponentsWithStats(
        motionMask, labels, stats, centroids, 8, CV_32S);

    double maxAllowedArea = maxAreaRatio * motionMask.rows * motionMask.cols;

    for (int i = 1; i < nComponents; i++) {  // skip background (label 0)
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Filter by area
        if (area < minArea || area > maxAllowedArea) continue;

        TrackedRegion region;
        region.id = i;
        region.area = area;

        // Bounding box
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        region.bbox = cv::Rect(left, top, width, height);

        // Centroid
        region.centroid.x = centroids.at<double>(i, 0);
        region.centroid.y = centroids.at<double>(i, 1);

        // Estimate velocity from optical flow (median of flow vectors in bbox)
        if (!flow.empty()) {
            std::vector<double> vx, vy;
            int xEnd = std::min(left + width, flow.cols);
            int yEnd = std::min(top + height, flow.rows);
            for (int y = std::max(top, 0); y < yEnd; y++) {
                for (int x = std::max(left, 0); x < xEnd; x++) {
                    if (labels.at<int>(y, x) == i) {
                        const cv::Vec2f& f = flow.at<cv::Vec2f>(y, x);
                        vx.push_back(f[0]);
                        vy.push_back(f[1]);
                    }
                }
            }

            if (!vx.empty()) {
                // Median (robust to outliers)
                std::sort(vx.begin(), vx.end());
                std::sort(vy.begin(), vy.end());
                size_t mid = vx.size() / 2;
                region.velocity.x = vx[mid];
                region.velocity.y = vy[mid];
            }
        }

        regions.push_back(region);
    }

    return regions;
}

void MotionTracker::drawTracking(cv::Mat& image,
                                  const std::vector<TrackedRegion>& regions,
                                  double scaleArrow) {
    for (const auto& r : regions) {
        // Green bounding box
        cv::rectangle(image, r.bbox, cv::Scalar(0, 255, 0), 2);

        // ID label
        std::string label = "ID:" + std::to_string(r.id);
        cv::putText(image, label,
                    cv::Point(r.bbox.x, r.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);

        // Blue centroid dot
        cv::Point pt(static_cast<int>(r.centroid.x),
                     static_cast<int>(r.centroid.y));
        cv::circle(image, pt, 4, cv::Scalar(255, 0, 0), -1);

        // Red velocity arrow
        if (cv::norm(r.velocity) > 0.5) {
            cv::Point tip(static_cast<int>(r.centroid.x + r.velocity.x * scaleArrow),
                          static_cast<int>(r.centroid.y + r.velocity.y * scaleArrow));
            cv::arrowedLine(image, pt, tip,
                            cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.2);
        }
    }
}

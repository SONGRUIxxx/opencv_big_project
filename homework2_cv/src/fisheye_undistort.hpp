#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

/**
 * @brief Custom fisheye undistorter for WoodScape radial_poly model.
 *
 * WoodScape uses: r(θ) = k1*θ + k2*θ² + k3*θ³ + k4*θ⁴
 * This differs from OpenCV's fisheye model (odd-power polynomial),
 * so we must implement backward-mapping ourselves.
 */
class FisheyeUndistorter {
public:
    /**
     * @param calibJsonPath Path to a calibration JSON file
     * @param outputHfovDeg Horizontal FOV of output perspective image (degrees)
     * @param outputWidth   Width of output perspective image (pixels)
     */
    FisheyeUndistorter(const std::string& calibJsonPath,
                       double outputHfovDeg = 120.0,
                       int outputWidth = 1024);

    /** Undistort a fisheye BGR image to perspective projection. */
    cv::Mat undistort(const cv::Mat& fisheyeImage) const;

    /** Undistort a binary mask (uses INTER_NEAREST to preserve binary values). */
    cv::Mat undistortMask(const cv::Mat& fisheyeMask) const;

    /** Return output image size (width, height). */
    cv::Size getOutputSize() const { return cv::Size(outWidth_, outHeight_); }

    /** Return fisheye principal point (cx, cy). */
    cv::Point2d getPrincipalPoint() const { return cv::Point2d(cxFish_, cyFish_); }

    /** Return maximum radius from principal point to image corner. */
    double getMaxRadius() const;

    /** Return the horizontal FOV in radians. */
    double getHfov() const { return hfov_; }

    /** Access remap tables (for debugging/visualization). */
    const cv::Mat& getMapX() const { return mapX_; }
    const cv::Mat& getMapY() const { return mapY_; }

private:
    /** Forward projection: r(θ) */
    double rOfTheta(double theta) const;

    /** Derivative: r'(θ) */
    double drDtheta(double theta) const;

    /** Newton-Raphson to find max_theta from r_max */
    double computeMaxTheta() const;

    /** Build the backward-mapping lookup tables */
    void buildRemapTables();

    // Calibration parameters
    std::vector<double> k_;     // [k1, k2, k3, k4]
    int polyOrder_;
    double aspectRatio_;

    // Fisheye image dimensions and principal point
    int widthFish_;
    int heightFish_;
    double cxFish_;
    double cyFish_;

    // Output perspective parameters
    double hfov_;
    int outWidth_;
    int outHeight_;
    double focal_;
    double cxOut_;
    double cyOut_;

    // Maximum incidence angle (determines valid region)
    double maxTheta_;

    // Precomputed backward-mapping lookup tables
    cv::Mat mapX_;
    cv::Mat mapY_;
};

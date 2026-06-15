#include "fisheye_undistort.hpp"

#include <opencv2/imgproc.hpp>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- JSON helper (minimal parser – avoids dependency on nlohmann) ----
namespace {

// Extract a double value for a given key from simple JSON
double extractDouble(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return 0.0;
    pos += search.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) return 0.0;
    pos++;
    // skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n'))
        pos++;
    std::string val;
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '.' ||
           json[pos] == '-' || json[pos] == '+' || json[pos] == 'e' || json[pos] == 'E'))
        val += json[pos++];
    if (val.empty()) return 0.0;
    return std::stod(val);
}

std::string extractString(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n'))
        pos++;
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++;
    std::string val;
    while (pos < json.size() && json[pos] != '"')
        val += json[pos++];
    return val;
}

int extractInt(const std::string& json, const std::string& key) {
    return static_cast<int>(extractDouble(json, key));
}

} // anonymous namespace


FisheyeUndistorter::FisheyeUndistorter(const std::string& calibJsonPath,
                                       double outputHfovDeg,
                                       int outputWidth)
{
    // Read JSON file
    std::ifstream file(calibJsonPath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open calibration file: " + calibJsonPath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    // Parse intrinsic parameters from the "intrinsic" block
    std::string intrSearch = "\"intrinsic\"";
    auto intrPos = json.find(intrSearch);
    std::string intrBlock;
    if (intrPos != std::string::npos) {
        auto braceStart = json.find('{', intrPos);
        if (braceStart != std::string::npos) {
            int depth = 0;
            auto i = braceStart;
            while (i < json.size()) {
                if (json[i] == '{') depth++;
                if (json[i] == '}') depth--;
                i++;
                if (depth == 0) {
                    intrBlock = json.substr(braceStart, i - braceStart);
                    break;
                }
            }
        }
    }

    k_.resize(4);
    k_[0] = extractDouble(intrBlock, "k1");
    k_[1] = extractDouble(intrBlock, "k2");
    k_[2] = extractDouble(intrBlock, "k3");
    k_[3] = extractDouble(intrBlock, "k4");
    polyOrder_ = extractInt(intrBlock, "poly_order");
    aspectRatio_ = extractDouble(intrBlock, "aspect_ratio");

    widthFish_ = extractInt(intrBlock, "width");
    heightFish_ = extractInt(intrBlock, "height");

    double cxOffset = extractDouble(intrBlock, "cx_offset");
    double cyOffset = extractDouble(intrBlock, "cy_offset");
    cxFish_ = widthFish_ / 2.0 + cxOffset;
    cyFish_ = heightFish_ / 2.0 + cyOffset;

    // Output parameters
    hfov_ = outputHfovDeg * M_PI / 180.0;
    outWidth_ = outputWidth;
    focal_ = outWidth_ / (2.0 * std::tan(hfov_ / 2.0));

    double vfov = 2.0 * std::atan(std::tan(hfov_ / 2.0) *
                                  (static_cast<double>(heightFish_) / widthFish_));
    outHeight_ = static_cast<int>(2.0 * focal_ * std::tan(vfov / 2.0));
    if (outHeight_ < 1) outHeight_ = 1;

    cxOut_ = outWidth_ / 2.0;
    cyOut_ = outHeight_ / 2.0;

    // Compute max theta and build remap tables
    maxTheta_ = computeMaxTheta();
    buildRemapTables();
}

double FisheyeUndistorter::rOfTheta(double theta) const {
    // Horner's method: (((k4*θ + k3)*θ + k2)*θ + k1)*θ
    return (((k_[3] * theta + k_[2]) * theta + k_[1]) * theta + k_[0]) * theta;
}

double FisheyeUndistorter::drDtheta(double theta) const {
    return k_[0] + 2.0 * k_[1] * theta + 3.0 * k_[2] * theta * theta + 4.0 * k_[3] * theta * theta * theta;
}

double FisheyeUndistorter::computeMaxTheta() const {
    // Maximum radial distance (corner distance from principal point)
    double corners[4][2] = {
        {0.0, 0.0},
        {static_cast<double>(widthFish_ - 1), 0.0},
        {0.0, static_cast<double>(heightFish_ - 1)},
        {static_cast<double>(widthFish_ - 1), static_cast<double>(heightFish_ - 1)}
    };
    double rMax = 0.0;
    for (int i = 0; i < 4; i++) {
        double dx = corners[i][0] - cxFish_;
        double dy = corners[i][1] - cyFish_;
        double r = std::sqrt(dx * dx + dy * dy);
        rMax = std::max(rMax, r);
    }

    // Newton-Raphson: solve r(θ) = rMax
    double theta = rMax / k_[0];  // initial guess
    for (int iter = 0; iter < 100; iter++) {
        double rVal = rOfTheta(theta);
        double dr = drDtheta(theta);
        double delta = (rVal - rMax) / dr;
        theta -= delta;
        if (std::abs(delta) < 1e-10) break;
    }
    return theta;
}

void FisheyeUndistorter::buildRemapTables() {
    mapX_ = cv::Mat(outHeight_, outWidth_, CV_32FC1);
    mapY_ = cv::Mat(outHeight_, outWidth_, CV_32FC1);

    for (int v = 0; v < outHeight_; v++) {
        float* rowX = mapX_.ptr<float>(v);
        float* rowY = mapY_.ptr<float>(v);
        for (int u = 0; u < outWidth_; u++) {
            // Normalized coordinates relative to output principal point
            double nx = (u - cxOut_) / focal_;
            double ny = (v - cyOut_) / focal_;

            // θ = arctan(sqrt(nx² + ny²))
            double theta = std::atan(std::sqrt(nx * nx + ny * ny));

            // φ = atan2(ny, nx)
            double phi = std::atan2(ny, nx);

            if (theta > maxTheta_) {
                // Invalid pixel – outside fisheye FOV
                rowX[u] = -1.0f;
                rowY[u] = -1.0f;
            } else {
                double rFish = rOfTheta(theta);
                double uFish = cxFish_ + rFish * std::cos(phi);
                double vFish = cyFish_ + rFish * std::sin(phi);
                rowX[u] = static_cast<float>(uFish);
                rowY[u] = static_cast<float>(vFish);
            }
        }
    }
}

cv::Mat FisheyeUndistorter::undistort(const cv::Mat& fisheyeImage) const {
    cv::Mat result;
    cv::remap(fisheyeImage, result, mapX_, mapY_,
              cv::INTER_LINEAR,
              cv::BORDER_CONSTANT,
              cv::Scalar(0, 0, 0));
    return result;
}

cv::Mat FisheyeUndistorter::undistortMask(const cv::Mat& fisheyeMask) const {
    cv::Mat result;
    cv::remap(fisheyeMask, result, mapX_, mapY_,
              cv::INTER_NEAREST,
              cv::BORDER_CONSTANT,
              cv::Scalar(0));
    return result;
}

double FisheyeUndistorter::getMaxRadius() const {
    double corners[4][2] = {
        {0.0, 0.0},
        {static_cast<double>(widthFish_ - 1), 0.0},
        {0.0, static_cast<double>(heightFish_ - 1)},
        {static_cast<double>(widthFish_ - 1), static_cast<double>(heightFish_ - 1)}
    };
    double rMax = 0.0;
    for (int i = 0; i < 4; i++) {
        double dx = corners[i][0] - cxFish_;
        double dy = corners[i][1] - cyFish_;
        rMax = std::max(rMax, std::sqrt(dx * dx + dy * dy));
    }
    return rMax;
}

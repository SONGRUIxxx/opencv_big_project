#include "evaluation.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iomanip>

Evaluator::Evaluator(double cx, double cy, double maxR,
                     double centerThresh, double edgeThresh)
    : cx_(cx), cy_(cy), maxR_(maxR),
      centerThresh_(centerThresh), edgeThresh_(edgeThresh)
{}

cv::Mat Evaluator::regionMask(const cv::Size& size, const std::string& region) const {
    cv::Mat mask(size, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < size.height; y++) {
        uchar* row = mask.ptr<uchar>(y);
        for (int x = 0; x < size.width; x++) {
            double dx = x - cx_;
            double dy = y - cy_;
            double dist = std::sqrt(dx * dx + dy * dy);
            double ratio = dist / maxR_;

            if (region == "center" && ratio < centerThresh_) {
                row[x] = 255;
            } else if (region == "edge" && ratio > edgeThresh_) {
                row[x] = 255;
            } else if (region == "full") {
                row[x] = 255;
            }
        }
    }
    return mask;
}

EvalMetrics Evaluator::computeRegion(const cv::Mat& predMask,
                                      const cv::Mat& gtMask,
                                      const cv::Mat& rMask,
                                      const std::string& sampleId,
                                      const std::string& route,
                                      const std::string& method,
                                      const std::string& region) const
{
    EvalMetrics m;
    m.sampleId = sampleId;
    m.route = route;
    m.method = method;
    m.region = region;

    cv::Mat predROI = predMask & rMask;
    cv::Mat gtROI = gtMask & rMask;

    // Count pixels in the region
    long long regionPixels = cv::countNonZero(rMask);
    if (regionPixels == 0) {
        return m;  // no pixels in this region
    }

    // TP: both pred and gt are 255
    cv::Mat tpMat;
    cv::bitwise_and(predROI, gtROI, tpMat);
    m.tp = cv::countNonZero(tpMat);

    // FP: pred=255, gt=0
    cv::Mat fpMat;
    cv::bitwise_and(predROI, (~gtROI), fpMat);
    // Need to also mask by rMask to only count region pixels
    fpMat = fpMat & rMask;
    m.fp = cv::countNonZero(fpMat);

    // FN: pred=0, gt=255
    cv::Mat fnMat;
    cv::bitwise_and((~predROI), gtROI, fnMat);
    fnMat = fnMat & rMask;
    m.fn = cv::countNonZero(fnMat);

    // TN: pred=0, gt=0 (within region)
    // TN = regionPixels - TP - FP - FN
    m.tn = regionPixels - m.tp - m.fp - m.fn;

    // Compute metrics (guard against division by zero)
    long long tp_fp = m.tp + m.fp;
    long long tp_fn = m.tp + m.fn;
    long long tp_fp_fn = m.tp + m.fp + m.fn;

    m.iou = (tp_fp_fn > 0) ? static_cast<double>(m.tp) / tp_fp_fn : 0.0;
    m.precision = (tp_fp > 0) ? static_cast<double>(m.tp) / tp_fp : 0.0;
    m.recall = (tp_fn > 0) ? static_cast<double>(m.tp) / tp_fn : 0.0;

    if (m.precision + m.recall > 0.0) {
        m.f1 = 2.0 * m.precision * m.recall / (m.precision + m.recall);
    }

    return m;
}

std::vector<EvalMetrics> Evaluator::evaluate(const cv::Mat& predMask,
                                              const cv::Mat& gtMask,
                                              const std::string& sampleId,
                                              const std::string& route,
                                              const std::string& method) const
{
    std::vector<EvalMetrics> results;

    // Full image
    cv::Mat fullMask(predMask.size(), CV_8UC1, cv::Scalar(255));
    results.push_back(computeRegion(predMask, gtMask, fullMask,
                                    sampleId, route, method, "full"));

    // Center region
    cv::Mat cMask = regionMask(predMask.size(), "center");
    results.push_back(computeRegion(predMask, gtMask, cMask,
                                    sampleId, route, method, "center"));

    // Edge region
    cv::Mat eMask = regionMask(predMask.size(), "edge");
    results.push_back(computeRegion(predMask, gtMask, eMask,
                                    sampleId, route, method, "edge"));

    return results;
}

cv::Mat Evaluator::makeErrorMap(const cv::Mat& predMask, const cv::Mat& gtMask) {
    cv::Mat errorMap(predMask.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    for (int y = 0; y < predMask.rows; y++) {
        const uchar* predRow = predMask.ptr<uchar>(y);
        const uchar* gtRow = gtMask.ptr<uchar>(y);
        cv::Vec3b* outRow = errorMap.ptr<cv::Vec3b>(y);
        for (int x = 0; x < predMask.cols; x++) {
            bool pred = (predRow[x] > 127);
            bool gt = (gtRow[x] > 127);

            if (pred && gt) {
                outRow[x] = cv::Vec3b(0, 255, 0);      // Green = TP
            } else if (pred && !gt) {
                outRow[x] = cv::Vec3b(0, 0, 255);      // Red = FP
            } else if (!pred && gt) {
                outRow[x] = cv::Vec3b(255, 0, 0);      // Blue = FN
            }
            // else black = TN
        }
    }

    return errorMap;
}

cv::Mat Evaluator::validImageMask(const cv::Mat& img)
{
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();

    cv::Mat mask;
    cv::threshold(gray, mask, 8, 255, cv::THRESH_BINARY);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::erode(mask, mask, kernel, {-1, -1}, 1);
    return mask;
}

void Evaluator::writeResultsCSV(const std::string& path,
                                 const std::vector<EvalMetrics>& allMetrics) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);

    // Header
    f << "sample_id,route,method,region,tp,fp,fn,tn,iou,precision,recall,f1\n";

    for (const auto& m : allMetrics) {
        f << m.sampleId << ","
          << m.route << ","
          << m.method << ","
          << m.region << ","
          << m.tp << "," << m.fp << "," << m.fn << "," << m.tn << ","
          << m.iou << ","
          << m.precision << ","
          << m.recall << ","
          << m.f1 << "\n";
    }
}

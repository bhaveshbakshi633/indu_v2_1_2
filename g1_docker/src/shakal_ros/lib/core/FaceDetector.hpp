#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>

namespace shakal {

struct FaceInfo {
    cv::Rect bbox;
    float confidence;
    std::vector<cv::Point2f> landmarks;
};

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();

    bool init(const std::string& model_path,
              const cv::Size& input_size,
              float conf_threshold = 0.5f,
              float nms_threshold = 0.4f);

    std::vector<FaceInfo> detect(const cv::Mat& frame);

    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);
    void setInputSize(const cv::Size& size);

    bool isInitialized() const { return initialized_; }

private:
    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    bool initialized_;
};

}

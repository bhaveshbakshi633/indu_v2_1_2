#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

namespace shakal {

struct AntiSpoofResult {
    bool is_real;
    float confidence;
    float fake_score;
    float real_score;
};

class AntiSpoof {
public:
    AntiSpoof();
    ~AntiSpoof();

    bool init(const std::string& model_path, float threshold = 0.5f);

    AntiSpoofResult check(const cv::Mat& face);

    void setThreshold(float threshold);
    float getThreshold() const { return threshold_; }
    bool isInitialized() const { return initialized_; }

private:
    cv::dnn::Net net_;
    cv::Size input_size_;
    float threshold_;
    bool initialized_;

    cv::Mat preprocess(const cv::Mat& face);
};

}

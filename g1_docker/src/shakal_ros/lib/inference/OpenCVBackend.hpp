#pragma once

#include "inference/IInferenceBackend.hpp"
#include <opencv2/dnn.hpp>

namespace shakal {

class OpenCVBackend : public IInferenceBackend {
public:
    OpenCVBackend(BackendType type = BackendType::OPENCV_CPU);
    ~OpenCVBackend() override = default;

    bool init(const std::string& model_path,
              const cv::Size& input_size) override;

    bool infer(const cv::Mat& input, std::vector<cv::Mat>& outputs) override;

    BackendType getType() const override { return type_; }
    std::string getName() const override;
    bool isReady() const override { return ready_; }
    cv::Size getInputSize() const override { return input_size_; }

    // Get underlying network for advanced configuration
    cv::dnn::Net& getNet() { return net_; }

private:
    void configureBackend();

    cv::dnn::Net net_;
    cv::Size input_size_;
    BackendType type_;
    bool ready_ = false;
};

} // namespace shakal

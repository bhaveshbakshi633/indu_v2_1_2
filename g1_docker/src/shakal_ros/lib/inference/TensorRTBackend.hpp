#pragma once

#ifdef USE_TENSORRT

#include "inference/IInferenceBackend.hpp"
#include "inference/TensorRTEngine.hpp"

namespace shakal {

class TensorRTBackend : public IInferenceBackend {
public:
    TensorRTBackend();
    ~TensorRTBackend() override = default;

    bool init(const std::string& model_path,
              const cv::Size& input_size) override;

    bool infer(const cv::Mat& input, std::vector<cv::Mat>& outputs) override;

    BackendType getType() const override { return BackendType::TENSORRT; }
    std::string getName() const override { return "TensorRT"; }
    bool isReady() const override { return engine_.isReady(); }
    cv::Size getInputSize() const override { return input_size_; }

    // TensorRT specific config
    void setUseFP16(bool use) { use_fp16_ = use; }
    void setUseINT8(bool use) { use_int8_ = use; }
    void setCacheDir(const std::string& dir) { cache_dir_ = dir; }

private:
    TensorRTEngine engine_;
    cv::Size input_size_;
    bool use_fp16_ = true;
    bool use_int8_ = false;
    std::string cache_dir_;
};

} // namespace shakal

#endif // USE_TENSORRT

#ifdef USE_TENSORRT

#include "inference/TensorRTBackend.hpp"
#include "utils/Logger.hpp"

namespace shakal {

TensorRTBackend::TensorRTBackend() = default;

bool TensorRTBackend::init(const std::string& model_path,
                           const cv::Size& input_size) {
    input_size_ = input_size;

    TensorRTEngine::Config config;
    config.use_fp16 = use_fp16_;
    config.use_int8 = use_int8_;
    config.cache_dir = cache_dir_;

    return engine_.init(model_path, config);
}

bool TensorRTBackend::infer(const cv::Mat& input, std::vector<cv::Mat>& outputs) {
    if (!isReady()) {
        LOG_ERROR("TensorRT backend not ready");
        return false;
    }

    return engine_.infer(input, outputs);
}

} // namespace shakal

#endif // USE_TENSORRT

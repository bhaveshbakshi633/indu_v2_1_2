#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

namespace shakal {

enum class BackendType {
    TENSORRT,
    OPENCV_CUDA,
    OPENCV_CUDA_FP16,
    OPENCV_CPU,
    UNKNOWN
};

inline std::string backendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::TENSORRT: return "TensorRT";
        case BackendType::OPENCV_CUDA: return "OpenCV CUDA";
        case BackendType::OPENCV_CUDA_FP16: return "OpenCV CUDA FP16";
        case BackendType::OPENCV_CPU: return "OpenCV CPU";
        default: return "Unknown";
    }
}

// Abstract interface for inference backends
class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;

    // Initialize the backend with model
    virtual bool init(const std::string& model_path,
                      const cv::Size& input_size) = 0;

    // Run inference - input is preprocessed blob, output is raw network output
    virtual bool infer(const cv::Mat& input, std::vector<cv::Mat>& outputs) = 0;

    // Get backend type
    virtual BackendType getType() const = 0;

    // Get backend name for logging
    virtual std::string getName() const = 0;

    // Check if backend is ready
    virtual bool isReady() const = 0;

    // Get input size
    virtual cv::Size getInputSize() const = 0;
};

// Shared pointer type for backends
using InferenceBackendPtr = std::shared_ptr<IInferenceBackend>;

} // namespace shakal

#pragma once

#include "inference/IInferenceBackend.hpp"
#include "inference/HardwareDetector.hpp"
#include <string>
#include <vector>

namespace shakal {

// Configuration for backend creation (defined outside class for C++ compatibility)
struct BackendConfig {
    std::string cache_dir;          // For TensorRT engine caching
    bool prefer_fp16 = true;        // Use FP16 when available
    bool allow_int8 = false;        // Allow INT8 (needs calibration)
    bool verbose = false;           // Verbose logging
};

class BackendFactory {
public:
    // Create best available backend with automatic fallback
    // Fallback chain: TensorRT → OpenCV CUDA FP16 → OpenCV CUDA → OpenCV CPU
    static InferenceBackendPtr createBest(const std::string& model_path,
                                          const cv::Size& input_size,
                                          const BackendConfig& config = BackendConfig{});

    // Create specific backend type
    static InferenceBackendPtr create(BackendType type,
                                      const std::string& model_path,
                                      const cv::Size& input_size,
                                      const BackendConfig& config = BackendConfig{});

    // Get list of available backends on this system
    static std::vector<BackendType> getAvailableBackends();

    // Get recommended backend for current hardware
    static BackendType getRecommendedBackend();

    // Print backend selection info
    static void printAvailableBackends();

private:
    BackendFactory() = delete;

    static InferenceBackendPtr tryCreateBackend(BackendType type,
                                                const std::string& model_path,
                                                const cv::Size& input_size,
                                                const BackendConfig& config);
};

} // namespace shakal

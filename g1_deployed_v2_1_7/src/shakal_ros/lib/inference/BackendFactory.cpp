#include "inference/BackendFactory.hpp"
#include "inference/OpenCVBackend.hpp"
#include "utils/Logger.hpp"

#ifdef USE_TENSORRT
#include "inference/TensorRTBackend.hpp"
#endif

#include <iostream>

namespace shakal {

std::vector<BackendType> BackendFactory::getAvailableBackends() {
    std::vector<BackendType> available;
    auto caps = HardwareDetector::detect();

#ifdef USE_TENSORRT
    if (caps.has_tensorrt && caps.has_cuda) {
        available.push_back(BackendType::TENSORRT);
    }
#endif

    if (caps.has_cuda) {
        available.push_back(BackendType::OPENCV_CUDA_FP16);
        available.push_back(BackendType::OPENCV_CUDA);
    }

    // CPU always available
    available.push_back(BackendType::OPENCV_CPU);

    return available;
}

BackendType BackendFactory::getRecommendedBackend() {
    auto caps = HardwareDetector::detect();

#ifdef USE_TENSORRT
    if (caps.has_tensorrt && caps.has_cuda) {
        return BackendType::TENSORRT;
    }
#endif

    if (caps.has_cuda) {
        return BackendType::OPENCV_CUDA_FP16;
    }

    return BackendType::OPENCV_CPU;
}

void BackendFactory::printAvailableBackends() {
    auto available = getAvailableBackends();
    auto recommended = getRecommendedBackend();

    std::cout << "\n=== Available Inference Backends ===" << std::endl;
    for (auto type : available) {
        std::cout << "  - " << backendTypeToString(type);
        if (type == recommended) {
            std::cout << " (recommended)";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

InferenceBackendPtr BackendFactory::tryCreateBackend(BackendType type,
                                                      const std::string& model_path,
                                                      const cv::Size& input_size,
                                                      const BackendConfig& config) {
    InferenceBackendPtr backend;

    switch (type) {
#ifdef USE_TENSORRT
        case BackendType::TENSORRT: {
            auto trt_backend = std::make_shared<TensorRTBackend>();
            trt_backend->setUseFP16(config.prefer_fp16);
            trt_backend->setUseINT8(config.allow_int8);
            if (!config.cache_dir.empty()) {
                trt_backend->setCacheDir(config.cache_dir);
            }
            backend = trt_backend;
            break;
        }
#endif

        case BackendType::OPENCV_CUDA_FP16:
            backend = std::make_shared<OpenCVBackend>(BackendType::OPENCV_CUDA_FP16);
            break;

        case BackendType::OPENCV_CUDA:
            backend = std::make_shared<OpenCVBackend>(BackendType::OPENCV_CUDA);
            break;

        case BackendType::OPENCV_CPU:
            backend = std::make_shared<OpenCVBackend>(BackendType::OPENCV_CPU);
            break;

        default:
            return nullptr;
    }

    if (backend) {
        try {
            if (backend->init(model_path, input_size)) {
                return backend;
            }
        } catch (const std::exception& e) {
            LOG_WARN("Backend init exception: " + std::string(e.what()));
        } catch (...) {
            LOG_WARN("Backend init unknown exception");
        }
    }

    return nullptr;
}

InferenceBackendPtr BackendFactory::create(BackendType type,
                                            const std::string& model_path,
                                            const cv::Size& input_size,
                                            const BackendConfig& config) {
    auto backend = tryCreateBackend(type, model_path, input_size, config);
    if (!backend) {
        LOG_ERROR("Failed to create backend: " + backendTypeToString(type));
    }
    return backend;
}

InferenceBackendPtr BackendFactory::createBest(const std::string& model_path,
                                                const cv::Size& input_size,
                                                const BackendConfig& config) {
    // Define fallback chain
    std::vector<BackendType> fallback_chain = {
#ifdef USE_TENSORRT
        BackendType::TENSORRT,
#endif
        BackendType::OPENCV_CUDA_FP16,
        BackendType::OPENCV_CUDA,
        BackendType::OPENCV_CPU
    };

    auto caps = HardwareDetector::detect();

    LOG_INFO("Selecting inference backend...");
    if (config.verbose) {
        HardwareDetector::printCapabilities();
    }

    // Try each backend in order
    for (auto type : fallback_chain) {
        // Skip backends that won't work
        if ((type == BackendType::TENSORRT ||
             type == BackendType::OPENCV_CUDA ||
             type == BackendType::OPENCV_CUDA_FP16) && !caps.has_cuda) {
            continue;
        }

#ifdef USE_TENSORRT
        if (type == BackendType::TENSORRT && !caps.has_tensorrt) {
            continue;
        }
#else
        if (type == BackendType::TENSORRT) {
            continue;
        }
#endif

        LOG_INFO("Trying backend: " + backendTypeToString(type));

        auto backend = tryCreateBackend(type, model_path, input_size, config);
        if (backend) {
            LOG_INFO("Backend selected: " + backend->getName());
            return backend;
        }

        LOG_WARN("Backend failed: " + backendTypeToString(type) + ", trying next...");
    }

    LOG_ERROR("All backends failed!");
    return nullptr;
}

} // namespace shakal

#pragma once

#ifdef USE_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace shakal {

// Custom deleter for TensorRT objects
struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    void setVerbose(bool verbose) { verbose_ = verbose; }
private:
    bool verbose_ = false;
};

// TensorRT Engine - handles building, caching, and inference
class TensorRTEngine {
public:
    struct Config {
        bool use_fp16 = true;
        bool use_int8 = false;
        size_t workspace_size_mb = 1024;  // 1GB default
        std::string cache_dir;  // Empty = same dir as model
        bool verbose = false;
    };

    TensorRTEngine();
    ~TensorRTEngine();

    // Initialize from ONNX model (builds or loads cached engine)
    bool init(const std::string& onnx_path, const Config& config = {});

    // Run inference
    // Input: CHW float blob, Output: network outputs
    bool infer(const float* input, size_t input_size,
               std::vector<std::vector<float>>& outputs);

    // OpenCV Mat interface
    bool infer(const cv::Mat& blob, std::vector<cv::Mat>& outputs);

    // Get info
    std::string getEnginePath() const { return engine_path_; }
    std::vector<int64_t> getInputDims() const;
    std::vector<std::vector<int64_t>> getOutputDims() const;
    bool isReady() const { return ready_; }

    // Static utility
    static std::string generateEnginePath(const std::string& onnx_path,
                                          const std::string& cache_dir = "");

private:
    bool buildEngine(const std::string& onnx_path, const Config& config);
    bool loadEngine(const std::string& engine_path);
    bool saveEngine(const std::string& engine_path);
    bool setupBuffers();
    void cleanup();

    TRTLogger logger_;
    TRTUniquePtr<nvinfer1::IRuntime> runtime_;
    TRTUniquePtr<nvinfer1::ICudaEngine> engine_;
    TRTUniquePtr<nvinfer1::IExecutionContext> context_;

    // CUDA resources
    cudaStream_t stream_ = nullptr;
    std::vector<void*> device_buffers_;
    std::vector<size_t> buffer_sizes_;

    // Binding info
    int num_bindings_ = 0;
    int input_index_ = -1;
    std::vector<int> output_indices_;

    std::string engine_path_;
    bool ready_ = false;
};

} // namespace shakal

#endif // USE_TENSORRT

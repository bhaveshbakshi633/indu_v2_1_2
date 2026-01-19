#ifdef USE_TENSORRT

#include "inference/TensorRTEngine.hpp"
#include "inference/HardwareDetector.hpp"
#include "utils/Logger.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

namespace shakal {

// ============ TRT Logger ============
void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
        LOG_ERROR("[TensorRT] " + std::string(msg));
    } else if (severity == Severity::kWARNING) {
        LOG_WARN("[TensorRT] " + std::string(msg));
    } else if (verbose_ && severity == Severity::kINFO) {
        LOG_INFO("[TensorRT] " + std::string(msg));
    }
}

// ============ TensorRTEngine ============
TensorRTEngine::TensorRTEngine() = default;

TensorRTEngine::~TensorRTEngine() {
    cleanup();
}

void TensorRTEngine::cleanup() {
    // Free CUDA buffers
    for (auto& buf : device_buffers_) {
        if (buf) {
            cudaFree(buf);
            buf = nullptr;
        }
    }
    device_buffers_.clear();
    buffer_sizes_.clear();

    // Destroy stream
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    // TensorRT objects cleaned up by unique_ptr
    ready_ = false;
}

std::string TensorRTEngine::generateEnginePath(const std::string& onnx_path,
                                                const std::string& cache_dir) {
    // Get compute capability
    int cc = HardwareDetector::getComputeCapability();
    std::string trt_version = HardwareDetector::getTensorRTVersion();

    // Build filename: model_cc86_trt8.5.engine
    fs::path onnx_p(onnx_path);
    std::string base_name = onnx_p.stem().string();

    std::stringstream ss;
    ss << base_name << "_cc" << cc;
    if (!trt_version.empty() && trt_version != "detected") {
        // Simplify version to major.minor
        size_t dot1 = trt_version.find('.');
        size_t dot2 = trt_version.find('.', dot1 + 1);
        if (dot2 != std::string::npos) {
            ss << "_trt" << trt_version.substr(0, dot2);
        } else {
            ss << "_trt" << trt_version;
        }
    }
    ss << ".engine";

    // Determine directory
    fs::path dir;
    if (!cache_dir.empty()) {
        dir = cache_dir;
        fs::create_directories(dir);
    } else {
        dir = onnx_p.parent_path();
    }

    return (dir / ss.str()).string();
}

bool TensorRTEngine::init(const std::string& onnx_path, const Config& config) {
    cleanup();

    logger_.setVerbose(config.verbose);

    // Generate engine path
    engine_path_ = generateEnginePath(onnx_path, config.cache_dir);

    // Try loading cached engine
    if (fs::exists(engine_path_)) {
        LOG_INFO("Loading cached TensorRT engine: " + engine_path_);
        if (loadEngine(engine_path_)) {
            if (setupBuffers()) {
                ready_ = true;
                LOG_INFO("TensorRT engine loaded successfully");
                return true;
            }
        }
        LOG_WARN("Failed to load cached engine, rebuilding...");
    }

    // Build from ONNX
    LOG_INFO("Building TensorRT engine from ONNX (this may take a few minutes)...");
    LOG_INFO("Model: " + onnx_path);

    auto start = std::chrono::steady_clock::now();

    if (!buildEngine(onnx_path, config)) {
        LOG_ERROR("Failed to build TensorRT engine");
        return false;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    LOG_INFO("Engine built in " + std::to_string(duration.count()) + " seconds");

    // Save engine
    if (!saveEngine(engine_path_)) {
        LOG_WARN("Failed to cache engine (inference will still work)");
    } else {
        LOG_INFO("Engine cached: " + engine_path_);
    }

    // Setup buffers
    if (!setupBuffers()) {
        LOG_ERROR("Failed to setup CUDA buffers");
        return false;
    }

    ready_ = true;
    LOG_INFO("TensorRT engine ready");
    return true;
}

bool TensorRTEngine::buildEngine(const std::string& onnx_path, const Config& config) {
    // Create builder
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        LOG_ERROR("Failed to create TensorRT builder");
        return false;
    }

    // Create network with explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        LOG_ERROR("Failed to create network definition");
        return false;
    }

    // Create ONNX parser
    auto parser = TRTUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        LOG_ERROR("Failed to create ONNX parser");
        return false;
    }

    // Parse ONNX file
    if (!parser->parseFromFile(onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        LOG_ERROR("Failed to parse ONNX file: " + onnx_path);
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            LOG_ERROR("  " + std::string(parser->getError(i)->desc()));
        }
        return false;
    }

    // Create builder config
    auto builder_config = TRTUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!builder_config) {
        LOG_ERROR("Failed to create builder config");
        return false;
    }

    // Set workspace size
    builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                       config.workspace_size_mb * 1024ULL * 1024ULL);

    // Enable FP16 if supported and requested
    if (config.use_fp16 && builder->platformHasFastFp16()) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        LOG_INFO("FP16 mode enabled");
    }

    // Enable INT8 if supported and requested
    if (config.use_int8 && builder->platformHasFastInt8()) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        LOG_INFO("INT8 mode enabled (note: may need calibration for best accuracy)");
    }

    // Handle dynamic input shapes
    auto input = network->getInput(0);
    auto input_dims = input->getDimensions();

    // Check if any dimension is dynamic (-1)
    bool has_dynamic = false;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        if (input_dims.d[i] == -1) {
            has_dynamic = true;
            break;
        }
    }

    if (has_dynamic) {
        LOG_INFO("Model has dynamic input shape, creating optimization profile");
        auto profile = builder->createOptimizationProfile();

        // For face detection (YuNet): typically NCHW with dynamic HW
        // For face recognition (ArcFace): fixed 1x3x112x112
        // Create reasonable min/opt/max based on typical usage

        nvinfer1::Dims4 min_dims{1, 3, 112, 112};
        nvinfer1::Dims4 opt_dims{1, 3, 640, 640};
        nvinfer1::Dims4 max_dims{1, 3, 1920, 1920};

        profile->setDimensions(input->getName(),
            nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input->getName(),
            nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(input->getName(),
            nvinfer1::OptProfileSelector::kMAX, max_dims);

        builder_config->addOptimizationProfile(profile);
    }

    // Build serialized network
    LOG_INFO("Building serialized network...");
    auto serialized = TRTUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *builder_config));
    if (!serialized || serialized->size() == 0) {
        LOG_ERROR("Failed to build serialized network");
        return false;
    }

    // Create runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        LOG_ERROR("Failed to create TensorRT runtime");
        return false;
    }

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(
        serialized->data(), serialized->size()));
    if (!engine_) {
        LOG_ERROR("Failed to deserialize CUDA engine");
        return false;
    }

    return true;
}

bool TensorRTEngine::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        return false;
    }

    // Create runtime if needed
    if (!runtime_) {
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            return false;
        }
    }

    // Deserialize
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    return engine_ != nullptr;
}

bool TensorRTEngine::saveEngine(const std::string& engine_path) {
    if (!engine_) {
        return false;
    }

    auto serialized = TRTUniquePtr<nvinfer1::IHostMemory>(engine_->serialize());
    if (!serialized) {
        return false;
    }

    std::ofstream file(engine_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    return file.good();
}

bool TensorRTEngine::setupBuffers() {
    if (!engine_) {
        return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        LOG_ERROR("Failed to create execution context");
        return false;
    }

    // Create CUDA stream
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        LOG_ERROR("Failed to create CUDA stream");
        return false;
    }

    // Get binding info
    num_bindings_ = engine_->getNbBindings();
    device_buffers_.resize(num_bindings_, nullptr);
    buffer_sizes_.resize(num_bindings_, 0);

    for (int i = 0; i < num_bindings_; ++i) {
        auto dims = engine_->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= (dims.d[j] > 0) ? dims.d[j] : 1;
        }
        size *= sizeof(float);
        buffer_sizes_[i] = size;

        if (cudaMalloc(&device_buffers_[i], size) != cudaSuccess) {
            LOG_ERROR("Failed to allocate CUDA buffer for binding " + std::to_string(i));
            return false;
        }

        if (engine_->bindingIsInput(i)) {
            input_index_ = i;
        } else {
            output_indices_.push_back(i);
        }
    }

    return true;
}

std::vector<int64_t> TensorRTEngine::getInputDims() const {
    std::vector<int64_t> dims;
    if (engine_ && input_index_ >= 0) {
        auto d = engine_->getBindingDimensions(input_index_);
        for (int i = 0; i < d.nbDims; ++i) {
            dims.push_back(d.d[i]);
        }
    }
    return dims;
}

std::vector<std::vector<int64_t>> TensorRTEngine::getOutputDims() const {
    std::vector<std::vector<int64_t>> all_dims;
    if (engine_) {
        for (int idx : output_indices_) {
            std::vector<int64_t> dims;
            auto d = engine_->getBindingDimensions(idx);
            for (int i = 0; i < d.nbDims; ++i) {
                dims.push_back(d.d[i]);
            }
            all_dims.push_back(dims);
        }
    }
    return all_dims;
}

bool TensorRTEngine::infer(const float* input, size_t input_size,
                           std::vector<std::vector<float>>& outputs) {
    if (!ready_ || !context_) {
        LOG_ERROR("Engine not ready for inference");
        return false;
    }

    // Copy input to device
    if (cudaMemcpyAsync(device_buffers_[input_index_], input,
                        input_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        LOG_ERROR("Failed to copy input to device");
        return false;
    }

    // Execute
    if (!context_->enqueueV2(device_buffers_.data(), stream_, nullptr)) {
        LOG_ERROR("Inference execution failed");
        return false;
    }

    // Copy outputs from device
    outputs.clear();
    outputs.resize(output_indices_.size());

    for (size_t i = 0; i < output_indices_.size(); ++i) {
        int idx = output_indices_[i];
        size_t num_elements = buffer_sizes_[idx] / sizeof(float);
        outputs[i].resize(num_elements);

        if (cudaMemcpyAsync(outputs[i].data(), device_buffers_[idx],
                           buffer_sizes_[idx],
                           cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
            LOG_ERROR("Failed to copy output from device");
            return false;
        }
    }

    // Synchronize
    cudaStreamSynchronize(stream_);

    return true;
}

bool TensorRTEngine::infer(const cv::Mat& blob, std::vector<cv::Mat>& outputs) {
    if (!ready_) {
        return false;
    }

    // Ensure blob is continuous float
    cv::Mat input_blob;
    if (blob.type() != CV_32F) {
        blob.convertTo(input_blob, CV_32F);
    } else {
        input_blob = blob;
    }

    if (!input_blob.isContinuous()) {
        input_blob = input_blob.clone();
    }

    // Run inference
    std::vector<std::vector<float>> raw_outputs;
    if (!infer(input_blob.ptr<float>(), input_blob.total(), raw_outputs)) {
        return false;
    }

    // Convert to cv::Mat
    outputs.clear();
    auto output_dims = getOutputDims();

    for (size_t i = 0; i < raw_outputs.size(); ++i) {
        const auto& out = raw_outputs[i];
        const auto& dims = output_dims[i];

        // Create Mat with appropriate shape
        std::vector<int> mat_dims;
        for (auto d : dims) {
            mat_dims.push_back(static_cast<int>(d));
        }

        cv::Mat output_mat(mat_dims, CV_32F);
        std::memcpy(output_mat.data, out.data(), out.size() * sizeof(float));
        outputs.push_back(output_mat);
    }

    return true;
}

} // namespace shakal

#endif // USE_TENSORRT

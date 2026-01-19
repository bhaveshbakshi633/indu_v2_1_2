#include "inference/OpenCVBackend.hpp"
#include "inference/HardwareDetector.hpp"
#include "utils/Logger.hpp"

namespace shakal {

OpenCVBackend::OpenCVBackend(BackendType type) : type_(type) {}

std::string OpenCVBackend::getName() const {
    return backendTypeToString(type_);
}

void OpenCVBackend::configureBackend() {
    switch (type_) {
        case BackendType::OPENCV_CUDA:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            break;

        case BackendType::OPENCV_CUDA_FP16:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            break;

        case BackendType::OPENCV_CPU:
        default:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            break;
    }
}

bool OpenCVBackend::init(const std::string& model_path,
                         const cv::Size& input_size) {
    input_size_ = input_size;

    try {
        // Determine model format from extension
        std::string ext = model_path.substr(model_path.find_last_of('.') + 1);

        if (ext == "onnx") {
            net_ = cv::dnn::readNetFromONNX(model_path);
        } else if (ext == "caffemodel" || ext == "prototxt") {
            // Would need prototxt path too
            LOG_ERROR("Caffe models not directly supported, use ONNX");
            return false;
        } else {
            // Try generic read
            net_ = cv::dnn::readNet(model_path);
        }

        if (net_.empty()) {
            LOG_ERROR("Failed to load model: " + model_path);
            return false;
        }

        // Configure backend
        configureBackend();

        ready_ = true;
        LOG_INFO("OpenCV DNN backend initialized: " + getName());
        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV error loading model: " + std::string(e.what()));
        return false;
    }
}

bool OpenCVBackend::infer(const cv::Mat& input, std::vector<cv::Mat>& outputs) {
    if (!ready_) {
        LOG_ERROR("OpenCV backend not ready");
        return false;
    }

    try {
        // Set input
        net_.setInput(input);

        // Get output layer names
        std::vector<std::string> output_names = net_.getUnconnectedOutLayersNames();

        // Forward pass
        net_.forward(outputs, output_names);

        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV inference error: " + std::string(e.what()));
        return false;
    }
}

} // namespace shakal

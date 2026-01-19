#include "core/FaceDetector.hpp"
#include "utils/Logger.hpp"
#include <opencv2/core/cuda.hpp>

namespace shakal {

// Runtime GPU detection
static bool hasGpuSupport() {
    try {
        int count = cv::cuda::getCudaEnabledDeviceCount();
        return count > 0;
    } catch (...) {
        return false;
    }
}

FaceDetector::FaceDetector()
    : input_size_(320, 320)
    , conf_threshold_(0.5f)
    , nms_threshold_(0.4f)
    , initialized_(false) {
}

FaceDetector::~FaceDetector() = default;

bool FaceDetector::init(const std::string& model_path,
                         const cv::Size& input_size,
                         float conf_threshold,
                         float nms_threshold) {
    try {
        input_size_ = input_size;
        conf_threshold_ = conf_threshold;
        nms_threshold_ = nms_threshold;

        // Runtime GPU detection with automatic CPU fallback
        bool using_gpu = false;
        if (hasGpuSupport()) {
            try {
                detector_ = cv::FaceDetectorYN::create(
                    model_path,
                    "",
                    input_size_,
                    conf_threshold_,
                    nms_threshold_,
                    5000,  // top_k
                    cv::dnn::DNN_BACKEND_CUDA,
                    cv::dnn::DNN_TARGET_CUDA
                );

                if (!detector_.empty()) {
                    // Test detection to verify CUDA works
                    cv::Mat dummy = cv::Mat::zeros(input_size_, CV_8UC3);
                    cv::Mat output;
                    detector_->detect(dummy, output);
                    using_gpu = true;
                    LOG_INFO("FaceDetector: GPU (CUDA) backend active");
                }
            } catch (const cv::Exception& e) {
                LOG_WARN("CUDA init failed for detector, falling back to CPU: " + std::string(e.what()));
                detector_.reset();
            }
        }

        if (!using_gpu) {
            detector_ = cv::FaceDetectorYN::create(
                model_path,
                "",
                input_size_,
                conf_threshold_,
                nms_threshold_,
                5000,  // top_k
                cv::dnn::DNN_BACKEND_OPENCV,
                cv::dnn::DNN_TARGET_CPU
            );
            LOG_INFO("FaceDetector: CPU backend active");
        }

        if (detector_.empty()) {
            LOG_ERROR("Failed to create FaceDetectorYN: " + model_path);
            return false;
        }

        initialized_ = true;
        LOG_INFO("FaceDetector initialized: " + model_path);
        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception: " + std::string(e.what()));
        return false;
    }
}

std::vector<FaceInfo> FaceDetector::detect(const cv::Mat& frame) {
    std::vector<FaceInfo> faces;

    if (!initialized_ || frame.empty()) {
        return faces;
    }

    // Update input size if frame size changed
    if (frame.cols != input_size_.width || frame.rows != input_size_.height) {
        detector_->setInputSize(frame.size());
    }

    cv::Mat output;
    detector_->detect(frame, output);

    if (output.empty()) {
        return faces;
    }

    // Parse output: each row is [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, confidence]
    for (int i = 0; i < output.rows; ++i) {
        FaceInfo face;

        float x = output.at<float>(i, 0);
        float y = output.at<float>(i, 1);
        float w = output.at<float>(i, 2);
        float h = output.at<float>(i, 3);

        face.bbox = cv::Rect(
            static_cast<int>(x),
            static_cast<int>(y),
            static_cast<int>(w),
            static_cast<int>(h)
        );

        // 5 landmarks
        for (int j = 0; j < 5; ++j) {
            float lx = output.at<float>(i, 4 + j * 2);
            float ly = output.at<float>(i, 5 + j * 2);
            face.landmarks.emplace_back(lx, ly);
        }

        face.confidence = output.at<float>(i, 14);
        faces.push_back(face);
    }

    return faces;
}

void FaceDetector::setConfidenceThreshold(float threshold) {
    conf_threshold_ = threshold;
    if (detector_) {
        detector_->setScoreThreshold(threshold);
    }
}

void FaceDetector::setNMSThreshold(float threshold) {
    nms_threshold_ = threshold;
    if (detector_) {
        detector_->setNMSThreshold(threshold);
    }
}

void FaceDetector::setInputSize(const cv::Size& size) {
    input_size_ = size;
    if (detector_) {
        detector_->setInputSize(size);
    }
}

}

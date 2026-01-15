#include "core/FaceEncoder.hpp"
#include "utils/Logger.hpp"
#include <opencv2/core/cuda.hpp>
#include <cmath>

namespace shakal {

// Runtime GPU detection - no compile-time flags
static bool hasGpuSupport() {
    try {
        int count = cv::cuda::getCudaEnabledDeviceCount();
        return count > 0;
    } catch (...) {
        return false;
    }
}

static const float ALIGN_DST[5][2] = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f}
};

FaceEncoder::FaceEncoder()
    : input_size_(112, 112)
    , embedding_size_(512)
    , initialized_(false) {
}

FaceEncoder::~FaceEncoder() = default;

bool FaceEncoder::init(const std::string& model_path,
                        const cv::Size& input_size,
                        int embedding_size) {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        if (net_.empty()) {
            LOG_ERROR("Failed to load recognition model: " + model_path);
            return false;
        }

        // Runtime GPU detection with automatic CPU fallback
        bool using_gpu = false;
        if (hasGpuSupport()) {
            try {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

                // Test forward pass to verify CUDA actually works
                cv::Mat dummy = cv::Mat::zeros(input_size, CV_8UC3);
                cv::Mat blob;
                cv::dnn::blobFromImage(dummy, blob, 1.0/127.5, input_size,
                                       cv::Scalar(127.5, 127.5, 127.5), true, false);
                net_.setInput(blob);
                net_.forward();

                using_gpu = true;
                LOG_INFO("FaceEncoder: GPU (CUDA) backend active");
            } catch (const cv::Exception& e) {
                LOG_WARN("CUDA init failed, falling back to CPU: " + std::string(e.what()));
            }
        }

        if (!using_gpu) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            LOG_INFO("FaceEncoder: CPU backend active");
        }

        input_size_ = input_size;
        embedding_size_ = embedding_size;
        initialized_ = true;

        LOG_INFO("FaceEncoder initialized: " + model_path);
        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception: " + std::string(e.what()));
        return false;
    }
}

cv::Mat FaceEncoder::alignFace(const cv::Mat& frame,
                                const cv::Rect& bbox,
                                const std::vector<cv::Point2f>& landmarks) {
    cv::Mat aligned;

    if (landmarks.size() == 5) {
        std::vector<cv::Point2f> src_pts = landmarks;
        std::vector<cv::Point2f> dst_pts;

        for (int i = 0; i < 5; ++i) {
            dst_pts.emplace_back(ALIGN_DST[i][0], ALIGN_DST[i][1]);
        }

        cv::Mat transform = cv::estimateAffinePartial2D(src_pts, dst_pts);
        if (!transform.empty()) {
            cv::warpAffine(frame, aligned, transform, input_size_);
        } else {
            cv::Mat face_roi = frame(bbox & cv::Rect(0, 0, frame.cols, frame.rows));
            cv::resize(face_roi, aligned, input_size_);
        }
    } else {
        cv::Rect safe_bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe_bbox.area() > 0) {
            cv::Mat face_roi = frame(safe_bbox);
            cv::resize(face_roi, aligned, input_size_);
        }
    }

    return aligned;
}

cv::Mat FaceEncoder::preprocess(const cv::Mat& face) {
    cv::Mat blob;
    cv::dnn::blobFromImage(face, blob, 1.0/127.5, input_size_,
                           cv::Scalar(127.5, 127.5, 127.5), true, false);
    return blob;
}

std::vector<float> FaceEncoder::encode(const cv::Mat& aligned_face) {
    std::vector<float> embedding;

    if (!initialized_ || aligned_face.empty()) {
        return embedding;
    }

    cv::Mat blob = preprocess(aligned_face);
    net_.setInput(blob);

    cv::Mat output = net_.forward();

    embedding.assign(output.ptr<float>(),
                     output.ptr<float>() + output.total());

    return normalize(embedding);
}

std::vector<std::vector<float>> FaceEncoder::encodeBatch(
    const std::vector<cv::Mat>& faces) {

    std::vector<std::vector<float>> embeddings;
    for (const auto& face : faces) {
        embeddings.push_back(encode(face));
    }
    return embeddings;
}

std::vector<float> FaceEncoder::normalize(const std::vector<float>& embedding) {
    std::vector<float> normalized(embedding.size());

    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-10) {
        for (size_t i = 0; i < embedding.size(); ++i) {
            normalized[i] = embedding[i] / norm;
        }
    }

    return normalized;
}

float FaceEncoder::cosineSimilarity(const std::vector<float>& a,
                                     const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }

    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }

    return dot;
}

}

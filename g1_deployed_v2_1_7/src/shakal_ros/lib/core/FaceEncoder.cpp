#include "core/FaceEncoder.hpp"
#include "inference/BackendFactory.hpp"
#include "utils/Logger.hpp"
#include <cmath>

namespace shakal {

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
        input_size_ = input_size;
        embedding_size_ = embedding_size;

        // Use BackendFactory to create best available backend
        BackendConfig config;
        config.prefer_fp16 = true;
        config.verbose = false;

        backend_ = BackendFactory::createBest(model_path, input_size_, config);

        if (!backend_ || !backend_->isReady()) {
            LOG_ERROR("Failed to create inference backend for: " + model_path);
            return false;
        }

        initialized_ = true;
        LOG_INFO("FaceEncoder initialized with " + backend_->getName() + " backend");
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("Exception initializing FaceEncoder: " + std::string(e.what()));
        return false;
    }
}

std::string FaceEncoder::getBackendName() const {
    return backend_ ? backend_->getName() : "None";
}

BackendType FaceEncoder::getBackendType() const {
    return backend_ ? backend_->getType() : BackendType::UNKNOWN;
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

    if (!initialized_ || !backend_ || aligned_face.empty()) {
        return embedding;
    }

    cv::Mat blob = preprocess(aligned_face);

    std::vector<cv::Mat> outputs;
    if (!backend_->infer(blob, outputs)) {
        LOG_ERROR("Inference failed in FaceEncoder");
        return embedding;
    }

    if (outputs.empty()) {
        LOG_ERROR("No output from inference");
        return embedding;
    }

    cv::Mat& output = outputs[0];
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

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include "inference/IInferenceBackend.hpp"

namespace shakal {

class FaceEncoder {
public:
    FaceEncoder();
    ~FaceEncoder();

    bool init(const std::string& model_path,
              const cv::Size& input_size,
              int embedding_size = 512);

    std::vector<float> encode(const cv::Mat& aligned_face);
    std::vector<std::vector<float>> encodeBatch(const std::vector<cv::Mat>& faces);

    cv::Mat alignFace(const cv::Mat& frame,
                      const cv::Rect& bbox,
                      const std::vector<cv::Point2f>& landmarks);

    static float cosineSimilarity(const std::vector<float>& a,
                                   const std::vector<float>& b);

    bool isInitialized() const { return initialized_; }
    int getEmbeddingSize() const { return embedding_size_; }

    // Get current backend info
    std::string getBackendName() const;
    BackendType getBackendType() const;

private:
    InferenceBackendPtr backend_;
    cv::Size input_size_;
    int embedding_size_;
    bool initialized_;

    cv::Mat preprocess(const cv::Mat& face);
    std::vector<float> normalize(const std::vector<float>& embedding);
};

}

#include "core/AntiSpoof.hpp"
#include "utils/Logger.hpp"

namespace shakal {

AntiSpoof::AntiSpoof()
    : input_size_(112, 112)
    , threshold_(0.5f)
    , initialized_(false) {
}

AntiSpoof::~AntiSpoof() = default;

bool AntiSpoof::init(const std::string& model_path, float threshold) {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        if (net_.empty()) {
            LOG_ERROR("Failed to load anti-spoof model: " + model_path);
            return false;
        }

        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        threshold_ = threshold;
        initialized_ = true;

        LOG_INFO("AntiSpoof initialized: " + model_path);
        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception: " + std::string(e.what()));
        return false;
    }
}

cv::Mat AntiSpoof::preprocess(const cv::Mat& face) {
    cv::Mat resized, rgb, normalized;

    // Resize to 112x112
    cv::resize(face, resized, input_size_);

    // BGR to RGB
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and normalize to [0, 1]
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);

    // Model expects NHWC format [1, 112, 112, 3]
    // Create blob in NHWC format
    cv::Mat blob = normalized.reshape(1, {1, input_size_.height, input_size_.width, 3});

    return blob;
}

AntiSpoofResult AntiSpoof::check(const cv::Mat& face) {
    AntiSpoofResult result;
    result.is_real = false;
    result.confidence = 0.0f;
    result.fake_score = 1.0f;
    result.real_score = 0.0f;

    if (!initialized_ || face.empty()) {
        return result;
    }

    try {
        cv::Mat blob = preprocess(face);
        net_.setInput(blob);

        cv::Mat output = net_.forward();

        // Output is [1, 2] - [fake_prob, real_prob]
        float fake_score = output.at<float>(0, 0);
        float real_score = output.at<float>(0, 1);

        result.fake_score = fake_score;
        result.real_score = real_score;
        result.is_real = (real_score > threshold_) && (real_score > fake_score);
        result.confidence = result.is_real ? real_score : fake_score;

    } catch (const cv::Exception& e) {
        LOG_ERROR("AntiSpoof inference error: " + std::string(e.what()));
    }

    return result;
}

void AntiSpoof::setThreshold(float threshold) {
    threshold_ = threshold;
}

}

#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <map>
#include <chrono>

#include "core/FaceDetector.hpp"
#include "core/FaceEncoder.hpp"
#include "core/FaceDatabase.hpp"
#include "core/AntiSpoof.hpp"

namespace shakal {

struct RecognitionResult {
    FaceInfo face;
    std::string name;
    float similarity;
    bool is_fake;
    float spoof_score;
};

struct PipelineConfig {
    int camera_id = 0;
    int frame_width = 640;
    int frame_height = 480;
    int fps = 30;
    bool use_gstreamer = false;

    // Display resolution (0 = same as capture)
    int display_width = 0;
    int display_height = 0;

    std::string detection_model;
    std::string recognition_model;
    cv::Size detection_input_size = cv::Size(640, 640);
    cv::Size recognition_input_size = cv::Size(112, 112);

    float detection_threshold = 0.5f;
    float recognition_threshold = 0.5f;
    float nms_threshold = 0.4f;

    std::string embeddings_file;

    // Anti-spoofing
    std::string antispoof_model;
    float antispoof_threshold = 0.5f;
    bool antispoof_enabled = true;

    bool show_fps = true;
    bool show_bbox = true;
    bool show_landmarks = false;

    // Headless mode - text output only, no GUI
    bool headless = false;
    float unknown_persist_time = 3.0f;  // seconds before tagging as Unknown
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineConfig& config);
    bool init(const std::string& config_file);

    // Override resolution - sets both capture and display
    void setResolution(int width, int height);

    // Override FPS
    void setFPS(int fps);

    // Enable headless mode (text output only)
    void setHeadless(bool enabled);

    void run();
    void stop();

    using FrameCallback = std::function<void(const cv::Mat&,
                                              const std::vector<RecognitionResult>&)>;
    void setFrameCallback(FrameCallback callback);

    bool isRunning() const { return running_; }
    float getFPS() const { return current_fps_; }

private:
    std::unique_ptr<FaceDetector> detector_;
    std::unique_ptr<FaceEncoder> encoder_;
    std::unique_ptr<FaceDatabase> database_;
    std::unique_ptr<AntiSpoof> antispoof_;

    cv::VideoCapture capture_;
    PipelineConfig config_;

    std::atomic<bool> running_;
    float current_fps_;
    FrameCallback frame_callback_;

    // Headless mode tracking
    std::vector<std::string> last_detected_names_;
    std::map<int, std::chrono::steady_clock::time_point> unknown_track_times_;
    int next_unknown_id_ = 0;

    // Smooth position tracking for GUI
    std::map<std::string, cv::Point2f> smooth_positions_;

    bool initCamera();
    std::string buildGStreamerPipeline();
    std::vector<RecognitionResult> processFrame(const cv::Mat& frame);
    void drawResults(cv::Mat& frame,
                     const std::vector<RecognitionResult>& results);
};

}

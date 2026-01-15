#include "pipeline/Pipeline.hpp"
#include "utils/Logger.hpp"
#include "utils/Timer.hpp"
#include <yaml-cpp/yaml.h>
#include <sstream>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

namespace shakal {

// Standard camera resolutions (for capture)
static const std::vector<std::pair<int, int>> CAMERA_RESOLUTIONS = {
    {640, 480},    // VGA
    {1280, 720},   // HD
    {1920, 1080},  // Full HD
    {2560, 1440},  // QHD
    {3840, 2160},  // 4K
};

// Standard 16:9 and 16:10 display resolutions
static const std::vector<std::pair<int, int>> DISPLAY_RESOLUTIONS = {
    // 16:9
    {640, 360},
    {854, 480},
    {1280, 720},
    {1920, 1080},
    {2560, 1440},
    {3840, 2160},
    // 16:10
    {1280, 800},
    {1440, 900},
    {1680, 1050},
    {1920, 1200},
    {2560, 1600},
};

// Find nearest resolution from a list
static std::pair<int, int> findNearestFromList(int width, int height,
    const std::vector<std::pair<int, int>>& resolutions) {
    int target_area = width * height;
    int min_diff = INT_MAX;
    std::pair<int, int> best = resolutions[0];

    for (const auto& res : resolutions) {
        int area = res.first * res.second;
        int diff = std::abs(area - target_area);
        if (diff < min_diff) {
            min_diff = diff;
            best = res;
        }
    }
    return best;
}

// Find nearest camera resolution
static std::pair<int, int> findNearestCameraResolution(int width, int height) {
    return findNearestFromList(width, height, CAMERA_RESOLUTIONS);
}

// Find nearest 16:9 or 16:10 display resolution
static std::pair<int, int> findNearestDisplayResolution(int width, int height) {
    return findNearestFromList(width, height, DISPLAY_RESOLUTIONS);
}

static std::string resolvePath(const std::string& base_dir, const std::string& path) {
    fs::path p(path);
    if (p.is_absolute()) {
        return path;
    }
    return (fs::path(base_dir) / p).string();
}

Pipeline::Pipeline()
    : running_(false)
    , current_fps_(0.0f) {
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::init(const std::string& config_file) {
    try {
        YAML::Node config = YAML::LoadFile(config_file);
        std::string config_dir = fs::path(config_file).parent_path().string();
        if (config_dir.empty()) config_dir = ".";

        PipelineConfig cfg;

        if (config["camera"]) {
            cfg.camera_id = config["camera"]["device_id"].as<int>(0);
            cfg.frame_width = config["camera"]["width"].as<int>(640);
            cfg.frame_height = config["camera"]["height"].as<int>(480);
            cfg.fps = config["camera"]["fps"].as<int>(30);
            cfg.use_gstreamer = config["camera"]["use_gstreamer"].as<bool>(false);
        }

        if (config["models"]) {
            cfg.detection_model = resolvePath(config_dir,
                config["models"]["detection"]["path"].as<std::string>());
            cfg.recognition_model = resolvePath(config_dir,
                config["models"]["recognition"]["path"].as<std::string>());

            auto det_size = config["models"]["detection"]["input_size"];
            if (det_size) {
                cfg.detection_input_size = cv::Size(
                    det_size[0].as<int>(), det_size[1].as<int>());
            }

            auto rec_size = config["models"]["recognition"]["input_size"];
            if (rec_size) {
                cfg.recognition_input_size = cv::Size(
                    rec_size[0].as<int>(), rec_size[1].as<int>());
            }

            cfg.detection_threshold =
                config["models"]["detection"]["confidence_threshold"].as<float>(0.5f);
            cfg.nms_threshold =
                config["models"]["detection"]["nms_threshold"].as<float>(0.4f);
        }

        if (config["recognition"]) {
            cfg.recognition_threshold =
                config["recognition"]["similarity_threshold"].as<float>(0.5f);
        }

        if (config["database"]) {
            cfg.embeddings_file = resolvePath(config_dir,
                config["database"]["embeddings_file"].as<std::string>());
        }

        if (config["display"]) {
            cfg.show_fps = config["display"]["show_fps"].as<bool>(true);
            cfg.show_bbox = config["display"]["show_bbox"].as<bool>(true);
            cfg.show_landmarks = config["display"]["show_landmarks"].as<bool>(false);
        }

        if (config["antispoof"]) {
            cfg.antispoof_enabled = config["antispoof"]["enabled"].as<bool>(true);
            cfg.antispoof_model = resolvePath(config_dir,
                config["antispoof"]["path"].as<std::string>("../models/anti_spoof.onnx"));
            cfg.antispoof_threshold = config["antispoof"]["threshold"].as<float>(0.5f);
        }

        return init(cfg);

    } catch (const YAML::Exception& e) {
        LOG_ERROR("Failed to parse config: " + std::string(e.what()));
        return false;
    }
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;

    detector_ = std::make_unique<FaceDetector>();
    if (!detector_->init(config_.detection_model,
                          config_.detection_input_size,
                          config_.detection_threshold,
                          config_.nms_threshold)) {
        LOG_ERROR("Failed to initialize face detector");
        return false;
    }

    encoder_ = std::make_unique<FaceEncoder>();
    if (!encoder_->init(config_.recognition_model,
                         config_.recognition_input_size)) {
        LOG_ERROR("Failed to initialize face encoder");
        return false;
    }

    database_ = std::make_unique<FaceDatabase>();
    if (!database_->load(config_.embeddings_file)) {
        LOG_ERROR("Failed to load face database");
        return false;
    }

    if (config_.antispoof_enabled && !config_.antispoof_model.empty()) {
        antispoof_ = std::make_unique<AntiSpoof>();
        if (!antispoof_->init(config_.antispoof_model, config_.antispoof_threshold)) {
            LOG_WARN("Failed to initialize anti-spoof, continuing without it");
            antispoof_.reset();
        }
    }

    if (!initCamera()) {
        LOG_ERROR("Failed to initialize camera");
        return false;
    }

    LOG_INFO("Pipeline initialized successfully");
    return true;
}

void Pipeline::setResolution(int width, int height) {
    // Direct set - only 720p/1080p/1440p/2160p allowed from CLI
    config_.frame_width = width;
    config_.frame_height = height;
    config_.display_width = width;
    config_.display_height = height;

    // Reinitialize camera with new resolution
    if (capture_.isOpened()) {
        capture_.release();
        initCamera();
    }
}

void Pipeline::setFPS(int fps) {
    config_.fps = fps;
    if (capture_.isOpened()) {
        capture_.set(cv::CAP_PROP_FPS, fps);
    }
}

void Pipeline::setHeadless(bool enabled) {
    config_.headless = enabled;
}

bool Pipeline::initCamera() {
    if (config_.use_gstreamer) {
        std::string gst_pipeline = buildGStreamerPipeline();
        capture_.open(gst_pipeline, cv::CAP_GSTREAMER);
    } else {
        capture_.open(config_.camera_id, cv::CAP_V4L2);
    }

    if (!capture_.isOpened()) {
        return false;
    }

    // Find nearest standard resolution for camera capture
    auto [capture_width, capture_height] = findNearestCameraResolution(
        config_.frame_width, config_.frame_height);

    // Use MJPG for higher FPS (YUYV caps at 25fps on most cameras)
    capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, capture_width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, capture_height);
    capture_.set(cv::CAP_PROP_FPS, config_.fps);

    // Store actual capture resolution
    config_.frame_width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    config_.frame_height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));

    return true;
}

std::string Pipeline::buildGStreamerPipeline() {
    std::ostringstream ss;
    ss << "nvarguscamerasrc ! "
       << "video/x-raw(memory:NVMM), "
       << "width=" << config_.frame_width << ", "
       << "height=" << config_.frame_height << ", "
       << "format=NV12, framerate=" << config_.fps << "/1 ! "
       << "nvvidconv ! video/x-raw, format=BGRx ! "
       << "videoconvert ! video/x-raw, format=BGR ! "
       << "appsink drop=1";
    return ss.str();
}

std::vector<RecognitionResult> Pipeline::processFrame(const cv::Mat& frame) {
    std::vector<RecognitionResult> results;

    auto faces = detector_->detect(frame);

    for (const auto& face : faces) {
        RecognitionResult result;
        result.face = face;
        result.is_fake = false;
        result.spoof_score = 0.0f;

        cv::Mat aligned = encoder_->alignFace(frame, face.bbox, face.landmarks);
        if (!aligned.empty()) {
            // Anti-spoof check first
            if (antispoof_) {
                auto spoof_result = antispoof_->check(aligned);
                result.is_fake = !spoof_result.is_real;
                result.spoof_score = spoof_result.fake_score;

                if (result.is_fake) {
                    result.name = "FAKE";
                    result.similarity = 0.0f;
                    results.push_back(result);
                    continue;
                }
            }

            // Recognition only if real face
            auto embedding = encoder_->encode(aligned);
            auto match = database_->match(embedding, config_.recognition_threshold);
            result.name = match.name;
            result.similarity = match.similarity;
        } else {
            result.name = "Unknown";
            result.similarity = 0.0f;
        }

        results.push_back(result);
    }

    return results;
}

void Pipeline::drawResults(cv::Mat& frame,
                            const std::vector<RecognitionResult>& results) {
    const float smoothing = 0.15f;  // Lower = smoother, higher = more responsive

    for (const auto& result : results) {
        const auto& face = result.face;

        // Skip Unknown faces in GUI - only show in headless
        if (result.name == "Unknown" && !result.is_fake) {
            continue;
        }

        if (config_.show_bbox) {
            cv::Scalar bg_color, text_color;
            if (result.is_fake) {
                bg_color = cv::Scalar(0, 0, 200);      // Dark red
                text_color = cv::Scalar(255, 255, 255);
            } else if (result.name != "Unknown") {
                bg_color = cv::Scalar(46, 139, 87);    // Sea green
                text_color = cv::Scalar(255, 255, 255);
            } else {
                bg_color = cv::Scalar(30, 30, 30);     // Dark gray
                text_color = cv::Scalar(200, 200, 200);
            }

            std::string label = result.name;
            if (result.is_fake) {
                label = "FAKE";
            }

            int baseline = 0;
            double font_scale = 0.7;
            int thickness = 2;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                  font_scale, thickness, &baseline);

            // Target position: top right corner, above face
            int pill_w = text_size.width + 20;
            int pill_h = text_size.height + 12;
            float target_x = face.bbox.x + face.bbox.width + 15;  // Right of face
            float target_y = face.bbox.y - pill_h - 20;           // Above face

            // Smooth position using exponential moving average
            std::string track_key = label + "_" + std::to_string(face.bbox.x / 100);
            if (smooth_positions_.find(track_key) == smooth_positions_.end()) {
                smooth_positions_[track_key] = cv::Point2f(target_x, target_y);
            } else {
                smooth_positions_[track_key].x += (target_x - smooth_positions_[track_key].x) * smoothing;
                smooth_positions_[track_key].y += (target_y - smooth_positions_[track_key].y) * smoothing;
            }

            int pill_x = static_cast<int>(smooth_positions_[track_key].x);
            int pill_y = static_cast<int>(smooth_positions_[track_key].y);

            // Keep pill within frame
            pill_x = std::max(4, std::min(pill_x, frame.cols - pill_w - 4));
            pill_y = std::max(4, pill_y);

            // Draw pill background
            cv::Rect pill_rect(pill_x, pill_y, pill_w, pill_h);
            if (pill_rect.x >= 0 && pill_rect.y >= 0 &&
                pill_rect.x + pill_rect.width <= frame.cols &&
                pill_rect.y + pill_rect.height <= frame.rows) {
                cv::Mat roi = frame(pill_rect);
                cv::Mat pill_bg(pill_h, pill_w, frame.type(), bg_color);
                double alpha = 0.85;
                cv::addWeighted(pill_bg, alpha, roi, 1.0 - alpha, 0, roi);

                // Draw text centered in pill
                int text_x = pill_x + (pill_w - text_size.width) / 2;
                int text_y = pill_y + (pill_h + text_size.height) / 2 - 2;
                cv::putText(frame, label, cv::Point(text_x, text_y),
                            cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
            }
        }

        if (config_.show_landmarks && !face.landmarks.empty()) {
            for (const auto& pt : face.landmarks) {
                cv::circle(frame, pt, 2, cv::Scalar(255, 0, 0), -1);
            }
        }
    }

    if (config_.show_fps) {
        std::string fps_text = "FPS: " + std::to_string(int(current_fps_));
        cv::putText(frame, fps_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
}

void Pipeline::run() {
    running_ = true;
    FPSCounter fps_counter;

    // Log camera settings once
    int fourcc = static_cast<int>(capture_.get(cv::CAP_PROP_FOURCC));
    char fourcc_str[5] = {
        static_cast<char>(fourcc & 0xFF),
        static_cast<char>((fourcc >> 8) & 0xFF),
        static_cast<char>((fourcc >> 16) & 0xFF),
        static_cast<char>((fourcc >> 24) & 0xFF),
        '\0'
    };
    LOG_INFO("Camera: " + std::string(fourcc_str) + " " +
             std::to_string(config_.frame_width) + "x" + std::to_string(config_.frame_height) +
             " @ " + std::to_string((int)capture_.get(cv::CAP_PROP_FPS)) + "fps");

    LOG_INFO("Pipeline started" + std::string(config_.headless ? " (headless)" : ""));

    // For unknown persistence tracking (simplified)
    bool unknown_present = false;
    std::chrono::steady_clock::time_point unknown_first_seen;

    while (running_) {
        cv::Mat frame;
        capture_ >> frame;
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        fps_counter.tick();
        current_fps_ = fps_counter.getFPS();

        auto results = processFrame(frame);

        if (frame_callback_) {
            frame_callback_(frame, results);
        }

        if (config_.headless) {
            // Headless mode: text output only
            auto now = std::chrono::steady_clock::now();
            std::vector<std::string> current_names;
            bool has_unknown = false;

            for (const auto& result : results) {
                if (result.name != "Unknown") {
                    current_names.push_back(result.name);
                } else {
                    has_unknown = true;
                }
            }

            // Unknown persistence tracking
            if (has_unknown) {
                if (!unknown_present) {
                    // First time seeing unknown
                    unknown_present = true;
                    unknown_first_seen = now;
                } else {
                    // Check if 3 seconds passed
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - unknown_first_seen).count() / 1000.0f;
                    if (elapsed >= config_.unknown_persist_time) {
                        current_names.push_back("Unknown");
                    }
                }
            } else {
                // Reset unknown tracking
                unknown_present = false;
            }

            // Sort and deduplicate names
            std::sort(current_names.begin(), current_names.end());
            current_names.erase(std::unique(current_names.begin(), current_names.end()),
                               current_names.end());

            // Print only if changed
            if (current_names != last_detected_names_) {
                if (!current_names.empty()) {
                    std::string output;
                    for (size_t i = 0; i < current_names.size(); i++) {
                        if (i > 0) output += ", ";
                        output += current_names[i];
                    }
                    std::cout << output << std::endl << std::flush;
                }
                last_detected_names_ = current_names;
            }

            // Small delay for headless mode
            std::this_thread::sleep_for(std::chrono::milliseconds(30));

        } else {
            // GUI mode
            drawResults(frame, results);

            // Resize to display resolution if set
            cv::Mat display_frame = frame;
            if (config_.display_width > 0 && config_.display_height > 0) {
                cv::resize(frame, display_frame,
                           cv::Size(config_.display_width, config_.display_height));
            }

            cv::imshow("Shakal", display_frame);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') {
                stop();
            }
        }
    }

    capture_.release();
    if (!config_.headless) {
        cv::destroyAllWindows();
    }
    LOG_INFO("Pipeline stopped");
}

void Pipeline::stop() {
    running_ = false;
}

void Pipeline::setFrameCallback(FrameCallback callback) {
    frame_callback_ = callback;
}

}

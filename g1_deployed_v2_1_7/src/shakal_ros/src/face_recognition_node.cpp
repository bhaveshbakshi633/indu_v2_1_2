#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include "shakal_ros/msg/face.hpp"
#include "shakal_ros/msg/face_array.hpp"

#include "core/FaceDetector.hpp"
#include "core/FaceEncoder.hpp"
#include "core/FaceDatabase.hpp"
#include "core/AntiSpoof.hpp"
#include "gpu/CudaUtils.hpp"
#include "utils/Logger.hpp"
#include "utils/Timer.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>

using namespace std::chrono_literals;

class FaceRecognitionNode : public rclcpp::Node
{
public:
    FaceRecognitionNode()
        : Node("face_recognition"),
          unknown_present_(false)
    {
        declare_parameters();
        load_parameters();

        if (shakal::CudaUtils::isAvailable()) {
            RCLCPP_INFO(get_logger(), "CUDA available - GPU acceleration enabled");
        } else {
            RCLCPP_INFO(get_logger(), "Running on CPU");
        }

        if (!init_components()) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize components");
            rclcpp::shutdown();
            return;
        }

        if (!init_camera()) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize camera");
            rclcpp::shutdown();
            return;
        }

        // Publishers
        faces_pub_ = create_publisher<shakal_ros::msg::FaceArray>("~/faces", 10);
        names_pub_ = create_publisher<std_msgs::msg::String>("~/names", 10);

        if (publish_debug_image_) {
            debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);
        }

        // Timer for processing loop
        int period_ms = static_cast<int>(1000.0 / publish_rate_hz_);
        process_timer_ = create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&FaceRecognitionNode::process_frame, this)
        );

        RCLCPP_INFO(get_logger(), "Face recognition node started");
    }

    ~FaceRecognitionNode()
    {
        if (capture_.isOpened()) {
            capture_.release();
        }
    }

private:
    // Core components
    std::unique_ptr<shakal::FaceDetector> detector_;
    std::unique_ptr<shakal::FaceEncoder> encoder_;
    std::unique_ptr<shakal::FaceDatabase> database_;
    std::unique_ptr<shakal::AntiSpoof> antispoof_;

    // Camera
    cv::VideoCapture capture_;

    // Publishers
    rclcpp::Publisher<shakal_ros::msg::FaceArray>::SharedPtr faces_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr names_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr process_timer_;

    // Parameters
    int camera_device_id_;
    int camera_width_;
    int camera_height_;
    int camera_fps_;
    std::string camera_format_;

    std::string detection_model_path_;
    std::string recognition_model_path_;
    int detection_input_w_;
    int detection_input_h_;
    int recognition_input_w_;
    int recognition_input_h_;

    float detection_confidence_threshold_;
    float detection_nms_threshold_;
    float recognition_similarity_threshold_;
    std::string database_path_;

    bool antispoof_enabled_;
    std::string antispoof_model_path_;
    float antispoof_threshold_;

    bool publish_debug_image_;
    double publish_rate_hz_;
    float unknown_persist_time_;

    // State tracking
    std::vector<std::string> last_detected_names_;
    bool unknown_present_;
    std::chrono::steady_clock::time_point unknown_first_seen_;
    shakal::FPSCounter fps_counter_;

    void declare_parameters()
    {
        // Camera
        declare_parameter("camera.device_id", 0);
        declare_parameter("camera.width", 1920);
        declare_parameter("camera.height", 1080);
        declare_parameter("camera.fps", 30);
        declare_parameter("camera.format", "MJPG");

        // Models
        declare_parameter("models.detection_path", "");
        declare_parameter("models.recognition_path", "");
        declare_parameter("models.detection_input_size", std::vector<int64_t>{640, 640});
        declare_parameter("models.recognition_input_size", std::vector<int64_t>{112, 112});

        // Detection
        declare_parameter("detection.confidence_threshold", 0.6);
        declare_parameter("detection.nms_threshold", 0.3);

        // Recognition
        declare_parameter("recognition.similarity_threshold", 0.6);
        declare_parameter("recognition.database_path", "");

        // Anti-spoof
        declare_parameter("antispoof.enabled", false);
        declare_parameter("antispoof.model_path", "");
        declare_parameter("antispoof.threshold", 0.5);

        // Output
        declare_parameter("output.publish_debug_image", false);
        declare_parameter("output.publish_rate_hz", 10.0);
        declare_parameter("output.unknown_persist_time", 3.0);
    }

    void load_parameters()
    {
        camera_device_id_ = get_parameter("camera.device_id").as_int();
        camera_width_ = get_parameter("camera.width").as_int();
        camera_height_ = get_parameter("camera.height").as_int();
        camera_fps_ = get_parameter("camera.fps").as_int();
        camera_format_ = get_parameter("camera.format").as_string();

        detection_model_path_ = get_parameter("models.detection_path").as_string();
        recognition_model_path_ = get_parameter("models.recognition_path").as_string();

        auto det_size = get_parameter("models.detection_input_size").as_integer_array();
        detection_input_w_ = det_size[0];
        detection_input_h_ = det_size[1];

        auto rec_size = get_parameter("models.recognition_input_size").as_integer_array();
        recognition_input_w_ = rec_size[0];
        recognition_input_h_ = rec_size[1];

        detection_confidence_threshold_ = get_parameter("detection.confidence_threshold").as_double();
        detection_nms_threshold_ = get_parameter("detection.nms_threshold").as_double();

        recognition_similarity_threshold_ = get_parameter("recognition.similarity_threshold").as_double();
        database_path_ = get_parameter("recognition.database_path").as_string();

        antispoof_enabled_ = get_parameter("antispoof.enabled").as_bool();
        antispoof_model_path_ = get_parameter("antispoof.model_path").as_string();
        antispoof_threshold_ = get_parameter("antispoof.threshold").as_double();

        publish_debug_image_ = get_parameter("output.publish_debug_image").as_bool();
        publish_rate_hz_ = get_parameter("output.publish_rate_hz").as_double();
        unknown_persist_time_ = get_parameter("output.unknown_persist_time").as_double();
    }

    bool init_components()
    {
        // Face detector
        detector_ = std::make_unique<shakal::FaceDetector>();
        if (!detector_->init(detection_model_path_,
                              cv::Size(detection_input_w_, detection_input_h_),
                              detection_confidence_threshold_,
                              detection_nms_threshold_)) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize face detector");
            return false;
        }

        // Face encoder
        encoder_ = std::make_unique<shakal::FaceEncoder>();
        if (!encoder_->init(recognition_model_path_,
                             cv::Size(recognition_input_w_, recognition_input_h_))) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize face encoder");
            return false;
        }

        // Face database
        database_ = std::make_unique<shakal::FaceDatabase>();
        if (!database_->load(database_path_)) {
            RCLCPP_WARN(get_logger(), "Failed to load face database, starting empty");
        }

        // Anti-spoof (optional)
        if (antispoof_enabled_ && !antispoof_model_path_.empty()) {
            antispoof_ = std::make_unique<shakal::AntiSpoof>();
            if (!antispoof_->init(antispoof_model_path_, antispoof_threshold_)) {
                RCLCPP_WARN(get_logger(), "Failed to initialize anti-spoof, continuing without");
                antispoof_.reset();
            }
        }

        return true;
    }

    bool init_camera()
    {
        capture_.open(camera_device_id_, cv::CAP_V4L2);
        if (!capture_.isOpened()) {
            return false;
        }

        // Set MJPG format for higher FPS
        if (camera_format_ == "MJPG") {
            capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        }

        capture_.set(cv::CAP_PROP_FRAME_WIDTH, camera_width_);
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height_);
        capture_.set(cv::CAP_PROP_FPS, camera_fps_);

        int actual_width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
        int actual_height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
        int actual_fps = static_cast<int>(capture_.get(cv::CAP_PROP_FPS));

        RCLCPP_INFO(get_logger(), "Camera opened: %dx%d @ %dfps",
                    actual_width, actual_height, actual_fps);

        return true;
    }

    void process_frame()
    {
        cv::Mat frame;
        capture_ >> frame;
        if (frame.empty()) {
            return;
        }

        fps_counter_.tick();
        auto start_time = std::chrono::steady_clock::now();

        // Detect faces
        auto det_start = std::chrono::steady_clock::now();
        auto faces = detector_->detect(frame);
        auto det_end = std::chrono::steady_clock::now();
        float det_time_ms = std::chrono::duration<float, std::milli>(det_end - det_start).count();

        // Process each face
        auto rec_start = std::chrono::steady_clock::now();
        shakal_ros::msg::FaceArray face_array_msg;
        face_array_msg.header.stamp = now();
        face_array_msg.header.frame_id = "camera_frame";
        face_array_msg.frame_width = frame.cols;
        face_array_msg.frame_height = frame.rows;

        std::vector<std::string> current_names;
        bool has_unknown = false;

        for (const auto& face : faces) {
            shakal_ros::msg::Face face_msg;
            face_msg.x = face.bbox.x;
            face_msg.y = face.bbox.y;
            face_msg.width = face.bbox.width;
            face_msg.height = face.bbox.height;
            face_msg.detection_confidence = face.confidence;
            face_msg.is_fake = false;
            face_msg.spoof_score = 0.0f;

            // Copy landmarks
            for (size_t i = 0; i < std::min(face.landmarks.size(), size_t(5)); ++i) {
                face_msg.landmarks[i].x = face.landmarks[i].x;
                face_msg.landmarks[i].y = face.landmarks[i].y;
                face_msg.landmarks[i].z = 0.0f;
            }

            // Align and process face
            cv::Mat aligned = encoder_->alignFace(frame, face.bbox, face.landmarks);
            if (!aligned.empty()) {
                // Anti-spoof check
                if (antispoof_) {
                    auto spoof_result = antispoof_->check(aligned);
                    face_msg.is_fake = !spoof_result.is_real;
                    face_msg.spoof_score = spoof_result.fake_score;

                    if (face_msg.is_fake) {
                        face_msg.name = "FAKE";
                        face_msg.similarity = 0.0f;
                        face_msg.recognized = false;
                        face_array_msg.faces.push_back(face_msg);
                        continue;
                    }
                }

                // Recognition
                auto embedding = encoder_->encode(aligned);
                auto match = database_->match(embedding, recognition_similarity_threshold_);
                face_msg.name = match.name;
                face_msg.similarity = match.similarity;
                face_msg.recognized = match.matched;

                if (match.matched) {
                    current_names.push_back(match.name);
                } else {
                    has_unknown = true;
                }
            } else {
                face_msg.name = "Unknown";
                face_msg.similarity = 0.0f;
                face_msg.recognized = false;
                has_unknown = true;
            }

            face_array_msg.faces.push_back(face_msg);
        }

        auto rec_end = std::chrono::steady_clock::now();
        float rec_time_ms = std::chrono::duration<float, std::milli>(rec_end - rec_start).count();

        auto end_time = std::chrono::steady_clock::now();
        float total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

        face_array_msg.detection_time_ms = det_time_ms;
        face_array_msg.recognition_time_ms = rec_time_ms;
        face_array_msg.total_time_ms = total_time_ms;

        // Publish face array
        faces_pub_->publish(face_array_msg);

        // Handle unknown persistence and publish names
        auto now_time = std::chrono::steady_clock::now();
        if (has_unknown) {
            if (!unknown_present_) {
                unknown_present_ = true;
                unknown_first_seen_ = now_time;
            } else {
                float elapsed = std::chrono::duration<float>(now_time - unknown_first_seen_).count();
                if (elapsed >= unknown_persist_time_) {
                    current_names.push_back("Unknown");
                }
            }
        } else {
            unknown_present_ = false;
        }

        // Sort and deduplicate
        std::sort(current_names.begin(), current_names.end());
        current_names.erase(std::unique(current_names.begin(), current_names.end()),
                           current_names.end());

        // Publish names every frame (constant 10 Hz)
        std_msgs::msg::String names_msg;
        for (size_t i = 0; i < current_names.size(); ++i) {
            if (i > 0) names_msg.data += ", ";
            names_msg.data += current_names[i];
        }
        names_pub_->publish(names_msg);
        last_detected_names_ = current_names;

        // Publish debug image if enabled
        if (publish_debug_image_ && debug_image_pub_) {
            draw_results(frame, face_array_msg);
            auto img_msg = cv_bridge::CvImage(face_array_msg.header, "bgr8", frame).toImageMsg();
            debug_image_pub_->publish(*img_msg);
        }
    }

    void draw_results(cv::Mat& frame, const shakal_ros::msg::FaceArray& faces)
    {
        for (const auto& face : faces.faces) {
            if (face.name == "Unknown" && !face.is_fake) {
                continue;  // Skip unknown in debug image
            }

            cv::Scalar bg_color, text_color;
            if (face.is_fake) {
                bg_color = cv::Scalar(0, 0, 200);       // Dark red
                text_color = cv::Scalar(255, 255, 255);
            } else if (face.recognized) {
                bg_color = cv::Scalar(46, 139, 87);     // Sea green
                text_color = cv::Scalar(255, 255, 255);
            } else {
                bg_color = cv::Scalar(30, 30, 30);      // Dark gray
                text_color = cv::Scalar(200, 200, 200);
            }

            std::string label = face.name;
            int baseline = 0;
            double font_scale = 0.7;
            int thickness = 2;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                  font_scale, thickness, &baseline);

            int pill_w = text_size.width + 20;
            int pill_h = text_size.height + 12;
            int pill_x = face.x + face.width + 15;
            int pill_y = face.y - pill_h - 20;

            pill_x = std::max(4, std::min(pill_x, frame.cols - pill_w - 4));
            pill_y = std::max(4, pill_y);

            cv::Rect pill_rect(pill_x, pill_y, pill_w, pill_h);
            if (pill_rect.x >= 0 && pill_rect.y >= 0 &&
                pill_rect.x + pill_rect.width <= frame.cols &&
                pill_rect.y + pill_rect.height <= frame.rows) {
                cv::Mat roi = frame(pill_rect);
                cv::Mat pill_bg(pill_h, pill_w, frame.type(), bg_color);
                cv::addWeighted(pill_bg, 0.85, roi, 0.15, 0, roi);

                int text_x = pill_x + (pill_w - text_size.width) / 2;
                int text_y = pill_y + (pill_h + text_size.height) / 2 - 2;
                cv::putText(frame, label, cv::Point(text_x, text_y),
                            cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
            }
        }

        // FPS counter
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps_counter_.getFPS()));
        cv::putText(frame, fps_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FaceRecognitionNode>());
    rclcpp::shutdown();
    return 0;
}

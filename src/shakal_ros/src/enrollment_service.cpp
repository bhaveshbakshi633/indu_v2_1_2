#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "shakal_ros/srv/enroll.hpp"
#include "shakal_ros/srv/remove.hpp"
#include "shakal_ros/srv/list_persons.hpp"

#include "core/FaceDetector.hpp"
#include "core/FaceEncoder.hpp"
#include "core/FaceDatabase.hpp"
#include "utils/Logger.hpp"

namespace fs = std::filesystem;

class EnrollmentServiceNode : public rclcpp::Node
{
public:
    EnrollmentServiceNode()
        : Node("enrollment_service")
    {
        declare_parameters();
        load_parameters();

        if (!init_components()) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize components");
            rclcpp::shutdown();
            return;
        }

        // Create services
        enroll_srv_ = create_service<shakal_ros::srv::Enroll>(
            "~/enroll",
            std::bind(&EnrollmentServiceNode::enroll_callback, this,
                      std::placeholders::_1, std::placeholders::_2)
        );

        remove_srv_ = create_service<shakal_ros::srv::Remove>(
            "~/remove",
            std::bind(&EnrollmentServiceNode::remove_callback, this,
                      std::placeholders::_1, std::placeholders::_2)
        );

        list_srv_ = create_service<shakal_ros::srv::ListPersons>(
            "~/list_persons",
            std::bind(&EnrollmentServiceNode::list_callback, this,
                      std::placeholders::_1, std::placeholders::_2)
        );

        RCLCPP_INFO(get_logger(), "Enrollment service node started");
    }

private:
    // Core components
    std::unique_ptr<shakal::FaceDetector> detector_;
    std::unique_ptr<shakal::FaceEncoder> encoder_;
    std::unique_ptr<shakal::FaceDatabase> database_;

    // Services
    rclcpp::Service<shakal_ros::srv::Enroll>::SharedPtr enroll_srv_;
    rclcpp::Service<shakal_ros::srv::Remove>::SharedPtr remove_srv_;
    rclcpp::Service<shakal_ros::srv::ListPersons>::SharedPtr list_srv_;

    // Parameters
    std::string detection_model_path_;
    std::string recognition_model_path_;
    std::string database_path_;
    int camera_device_id_;

    void declare_parameters()
    {
        declare_parameter("models.detection_path", "");
        declare_parameter("models.recognition_path", "");
        declare_parameter("recognition.database_path", "");
        declare_parameter("camera.device_id", 0);
    }

    void load_parameters()
    {
        detection_model_path_ = get_parameter("models.detection_path").as_string();
        recognition_model_path_ = get_parameter("models.recognition_path").as_string();
        database_path_ = get_parameter("recognition.database_path").as_string();
        camera_device_id_ = get_parameter("camera.device_id").as_int();
    }

    bool init_components()
    {
        detector_ = std::make_unique<shakal::FaceDetector>();
        if (!detector_->init(detection_model_path_, cv::Size(640, 640))) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize face detector");
            return false;
        }

        encoder_ = std::make_unique<shakal::FaceEncoder>();
        if (!encoder_->init(recognition_model_path_, cv::Size(112, 112))) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize face encoder");
            return false;
        }

        database_ = std::make_unique<shakal::FaceDatabase>();
        database_->load(database_path_);

        return true;
    }

    void enroll_callback(
        const std::shared_ptr<shakal_ros::srv::Enroll::Request> request,
        std::shared_ptr<shakal_ros::srv::Enroll::Response> response)
    {
        RCLCPP_INFO(get_logger(), "Enroll request: name=%s, mode=%s",
                    request->name.c_str(), request->mode.c_str());

        if (request->name.empty()) {
            response->success = false;
            response->message = "Name cannot be empty";
            response->embeddings_count = 0;
            return;
        }

        if (request->mode == "folder") {
            enroll_from_folder(request->name, request->folder_path, response);
        } else if (request->mode == "capture") {
            int num_captures = request->num_captures > 0 ? request->num_captures : 15;
            enroll_from_camera(request->name, num_captures, response);
        } else {
            response->success = false;
            response->message = "Invalid mode. Use 'folder' or 'capture'";
            response->embeddings_count = 0;
        }
    }

    void enroll_from_folder(
        const std::string& name,
        const std::string& folder_path,
        std::shared_ptr<shakal_ros::srv::Enroll::Response> response)
    {
        if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
            response->success = false;
            response->message = "Folder does not exist: " + folder_path;
            response->embeddings_count = 0;
            return;
        }

        std::vector<std::vector<float>> embeddings;
        int processed = 0;
        int failed = 0;

        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") {
                continue;
            }

            cv::Mat image = cv::imread(entry.path().string());
            if (image.empty()) {
                failed++;
                continue;
            }

            auto faces = detector_->detect(image);
            if (faces.empty()) {
                failed++;
                continue;
            }

            // Use largest face
            auto& face = *std::max_element(faces.begin(), faces.end(),
                [](const shakal::FaceInfo& a, const shakal::FaceInfo& b) {
                    return a.bbox.area() < b.bbox.area();
                });

            cv::Mat aligned = encoder_->alignFace(image, face.bbox, face.landmarks);
            if (aligned.empty()) {
                failed++;
                continue;
            }

            auto embedding = encoder_->encode(aligned);
            if (!embedding.empty()) {
                embeddings.push_back(embedding);
                processed++;
            }
        }

        if (embeddings.empty()) {
            response->success = false;
            response->message = "No valid embeddings extracted";
            response->embeddings_count = 0;
            return;
        }

        database_->addPerson(name, embeddings);
        database_->save(database_path_);

        response->success = true;
        response->message = "Enrolled " + name + " with " + std::to_string(embeddings.size()) +
                           " embeddings (" + std::to_string(failed) + " failed)";
        response->embeddings_count = embeddings.size();

        RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    }

    void enroll_from_camera(
        const std::string& name,
        int num_captures,
        std::shared_ptr<shakal_ros::srv::Enroll::Response> response)
    {
        cv::VideoCapture cap(camera_device_id_, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            response->success = false;
            response->message = "Failed to open camera";
            response->embeddings_count = 0;
            return;
        }

        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_FPS, 30);

        std::vector<std::vector<float>> embeddings;
        int frame_count = 0;
        int max_frames = num_captures * 30;  // Try for ~num_captures seconds worth of frames

        RCLCPP_INFO(get_logger(), "Starting camera capture for %s, need %d embeddings",
                    name.c_str(), num_captures);

        while (embeddings.size() < static_cast<size_t>(num_captures) && frame_count < max_frames) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                frame_count++;
                continue;
            }

            auto faces = detector_->detect(frame);
            if (!faces.empty()) {
                const auto& face = faces[0];
                cv::Mat aligned = encoder_->alignFace(frame, face.bbox, face.landmarks);

                if (!aligned.empty()) {
                    auto embedding = encoder_->encode(aligned);
                    if (!embedding.empty()) {
                        embeddings.push_back(embedding);
                        RCLCPP_INFO(get_logger(), "Captured %zu/%d", embeddings.size(), num_captures);
                    }
                }
            }

            frame_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // ~10 captures per second max
        }

        cap.release();

        if (embeddings.empty()) {
            response->success = false;
            response->message = "No faces captured";
            response->embeddings_count = 0;
            return;
        }

        database_->addPerson(name, embeddings);
        database_->save(database_path_);

        response->success = true;
        response->message = "Enrolled " + name + " with " + std::to_string(embeddings.size()) + " embeddings";
        response->embeddings_count = embeddings.size();

        RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    }

    void remove_callback(
        const std::shared_ptr<shakal_ros::srv::Remove::Request> request,
        std::shared_ptr<shakal_ros::srv::Remove::Response> response)
    {
        RCLCPP_INFO(get_logger(), "Remove request: name=%s", request->name.c_str());

        if (database_->removePerson(request->name)) {
            database_->save(database_path_);
            response->success = true;
            response->message = "Removed: " + request->name;
        } else {
            response->success = false;
            response->message = "Person not found: " + request->name;
        }
    }

    void list_callback(
        const std::shared_ptr<shakal_ros::srv::ListPersons::Request> /*request*/,
        std::shared_ptr<shakal_ros::srv::ListPersons::Response> response)
    {
        auto names = database_->getPersonNames();
        response->names = names;
        response->total_persons = names.size();

        // Embedding counts not available via public API
        // Fill with -1 to indicate unknown
        for (size_t i = 0; i < names.size(); ++i) {
            response->embedding_counts.push_back(-1);
        }

        RCLCPP_INFO(get_logger(), "List request: %zu persons enrolled", names.size());
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EnrollmentServiceNode>());
    rclcpp::shutdown();
    return 0;
}

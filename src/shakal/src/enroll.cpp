#include "core/FaceDetector.hpp"
#include "core/FaceEncoder.hpp"
#include "core/FaceDatabase.hpp"
#include "utils/Logger.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " <command> [options]\n"
              << "\nCommands:\n"
              << "  add <name> <image_dir>    Add person from images directory\n"
              << "  remove <name>             Remove person from database\n"
              << "  list                      List all enrolled persons\n"
              << "  capture <name>            Capture faces from camera\n"
              << "\nOptions:\n"
              << "  --detection-model <path>  Detection model path\n"
              << "  --recognition-model <path> Recognition model path\n"
              << "  --database <path>         Database file path\n"
              << std::endl;
}

struct EnrollConfig {
    std::string detection_model = "../models/face_detection.onnx";
    std::string recognition_model = "../models/face_recognition.onnx";
    std::string database_path = "../data/embeddings/database.bin";
};

bool enrollFromImages(const std::string& name,
                      const std::string& image_dir,
                      const EnrollConfig& config) {

    shakal::FaceDetector detector;
    if (!detector.init(config.detection_model, cv::Size(640, 640))) {
        LOG_ERROR("Failed to initialize detector");
        return false;
    }

    shakal::FaceEncoder encoder;
    if (!encoder.init(config.recognition_model, cv::Size(112, 112))) {
        LOG_ERROR("Failed to initialize encoder");
        return false;
    }

    shakal::FaceDatabase database;
    database.load(config.database_path);

    std::vector<std::vector<float>> embeddings;
    int processed = 0;
    int failed = 0;

    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") {
            continue;
        }

        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty()) {
            LOG_WARN("Failed to load: " + entry.path().string());
            failed++;
            continue;
        }

        auto faces = detector.detect(image);
        if (faces.empty()) {
            LOG_WARN("No face detected in: " + entry.path().filename().string());
            failed++;
            continue;
        }

        if (faces.size() > 1) {
            LOG_WARN("Multiple faces in: " + entry.path().filename().string() + ", using largest");
        }

        // Use largest face if multiple
        auto& face = *std::max_element(faces.begin(), faces.end(),
            [](const shakal::FaceInfo& a, const shakal::FaceInfo& b) {
                return a.bbox.area() < b.bbox.area();
            });

        cv::Mat aligned = encoder.alignFace(image, face.bbox, face.landmarks);

        if (aligned.empty()) {
            LOG_WARN("Failed to align face in: " + entry.path().filename().string());
            failed++;
            continue;
        }

        auto embedding = encoder.encode(aligned);
        if (!embedding.empty()) {
            embeddings.push_back(embedding);
            processed++;
            LOG_INFO("Processed: " + entry.path().filename().string());
        }
    }

    if (embeddings.empty()) {
        LOG_ERROR("No valid embeddings extracted");
        return false;
    }

    if (!database.addPerson(name, embeddings)) {
        LOG_ERROR("Failed to add person to database");
        return false;
    }

    if (!database.save(config.database_path)) {
        LOG_ERROR("Failed to save database");
        return false;
    }

    std::cout << "\nEnrollment complete for: " << name << std::endl;
    std::cout << "  Processed: " << processed << " images" << std::endl;
    std::cout << "  Failed:    " << failed << " images" << std::endl;
    std::cout << "  Embeddings: " << embeddings.size() << std::endl;

    return true;
}

bool captureFromCamera(const std::string& name, const EnrollConfig& config) {
    shakal::FaceDetector detector;
    if (!detector.init(config.detection_model, cv::Size(640, 640))) {
        LOG_ERROR("Failed to initialize detector");
        return false;
    }

    shakal::FaceEncoder encoder;
    if (!encoder.init(config.recognition_model, cv::Size(112, 112))) {
        LOG_ERROR("Failed to initialize encoder");
        return false;
    }

    shakal::FaceDatabase database;
    database.load(config.database_path);

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        LOG_ERROR("Failed to open camera");
        return false;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 60);

    std::vector<std::vector<float>> embeddings;
    int target_count = 15;

    std::cout << "\nCapture mode for: " << name << std::endl;
    std::cout << "Press SPACE to capture face (" << target_count << " needed)" << std::endl;
    std::cout << "Press 'q' to finish early, ESC to cancel" << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        auto faces = detector.detect(frame);

        for (const auto& face : faces) {
            cv::rectangle(frame, face.bbox, cv::Scalar(0, 255, 0), 2);
        }

        std::string status = "Captured: " + std::to_string(embeddings.size()) +
                            "/" + std::to_string(target_count);
        cv::putText(frame, status, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Enroll: " + name, frame);

        int key = cv::waitKey(30);

        if (key == 27) {
            std::cout << "Enrollment cancelled" << std::endl;
            cv::destroyAllWindows();
            return false;
        }
        else if (key == 'q' && !embeddings.empty()) {
            break;
        }
        else if (key == ' ' && !faces.empty()) {
            const auto& face = faces[0];
            cv::Mat aligned = encoder.alignFace(frame, face.bbox, face.landmarks);

            if (!aligned.empty()) {
                auto embedding = encoder.encode(aligned);
                if (!embedding.empty()) {
                    embeddings.push_back(embedding);
                    std::cout << "Captured " << embeddings.size() << "/" << target_count << std::endl;

                    if (embeddings.size() >= static_cast<size_t>(target_count)) {
                        break;
                    }
                }
            }
        }
    }

    cv::destroyAllWindows();

    if (embeddings.empty()) {
        LOG_ERROR("No embeddings captured");
        return false;
    }

    database.addPerson(name, embeddings);
    database.save(config.database_path);

    std::cout << "\nEnrollment complete: " << embeddings.size() << " embeddings saved" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    EnrollConfig config;
    std::string command = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--detection-model" && i + 1 < argc) {
            config.detection_model = argv[++i];
        } else if (arg == "--recognition-model" && i + 1 < argc) {
            config.recognition_model = argv[++i];
        } else if (arg == "--database" && i + 1 < argc) {
            config.database_path = argv[++i];
        }
    }

    if (command == "add") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " add <name> <image_dir>" << std::endl;
            return 1;
        }
        return enrollFromImages(argv[2], argv[3], config) ? 0 : 1;
    }
    else if (command == "remove") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " remove <name>" << std::endl;
            return 1;
        }
        shakal::FaceDatabase database;
        database.load(config.database_path);
        if (database.removePerson(argv[2])) {
            database.save(config.database_path);
            std::cout << "Removed: " << argv[2] << std::endl;
            return 0;
        }
        return 1;
    }
    else if (command == "list") {
        shakal::FaceDatabase database;
        database.load(config.database_path);
        auto names = database.getPersonNames();
        std::cout << "Enrolled persons (" << names.size() << "):" << std::endl;
        for (const auto& n : names) {
            std::cout << "  - " << n << std::endl;
        }
        return 0;
    }
    else if (command == "capture") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " capture <name>" << std::endl;
            return 1;
        }
        return captureFromCamera(argv[2], config) ? 0 : 1;
    }
    else {
        printUsage(argv[0]);
        return 1;
    }
}

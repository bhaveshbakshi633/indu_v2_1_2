#include "core/FaceDetector.hpp"
#include "core/FaceEncoder.hpp"
#include "core/FaceDatabase.hpp"
#include "utils/Logger.hpp"
#include "utils/Timer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <thread>

using namespace shakal;

void testCosineSimilarity() {
    std::cout << "Testing cosine similarity..." << std::endl;

    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f};
    float sim = FaceEncoder::cosineSimilarity(a, b);
    assert(std::abs(sim - 1.0f) < 0.001f);

    std::vector<float> c = {0.0f, 1.0f, 0.0f};
    sim = FaceEncoder::cosineSimilarity(a, c);
    assert(std::abs(sim) < 0.001f);

    std::cout << "  PASSED" << std::endl;
}

void testDatabase() {
    std::cout << "Testing database..." << std::endl;

    FaceDatabase db;

    std::vector<std::vector<float>> embeddings = {
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.15f, 0.25f, 0.35f, 0.45f}
    };
    assert(db.addPerson("test_person", embeddings));
    assert(db.hasPerson("test_person"));
    assert(db.getPersonCount() == 1);

    std::vector<float> query = {0.12f, 0.22f, 0.32f, 0.42f};
    auto result = db.match(query, 0.5f);
    std::cout << "  Match result: " << result.name
              << " (sim: " << result.similarity << ")" << std::endl;

    assert(db.removePerson("test_person"));
    assert(!db.hasPerson("test_person"));

    std::cout << "  PASSED" << std::endl;
}

void testTimer() {
    std::cout << "Testing timer..." << std::endl;

    Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    timer.stop();

    double elapsed = timer.elapsedMs();
    assert(elapsed >= 95 && elapsed <= 150);

    std::cout << "  Elapsed: " << elapsed << " ms" << std::endl;
    std::cout << "  PASSED" << std::endl;
}

void testFPSCounter() {
    std::cout << "Testing FPS counter..." << std::endl;

    FPSCounter fps;

    for (int i = 0; i < 50; ++i) {
        fps.tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    float measured_fps = fps.getFPS();
    std::cout << "  Measured FPS: " << measured_fps << std::endl;
    assert(measured_fps > 50 && measured_fps < 150);

    std::cout << "  PASSED" << std::endl;
}

void benchmarkDetector(const std::string& model_path) {
    std::cout << "\nBenchmarking detector..." << std::endl;

    FaceDetector detector;
    if (!detector.init(model_path, cv::Size(640, 640))) {
        std::cout << "  SKIPPED (model not found: " << model_path << ")" << std::endl;
        return;
    }

    cv::Mat test_image(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));

    for (int i = 0; i < 3; ++i) {
        detector.detect(test_image);
    }

    Timer timer;
    int iterations = 20;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        detector.detect(test_image);
    }
    timer.stop();

    double avg_ms = timer.elapsedMs() / iterations;
    std::cout << "  Average detection time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Detection FPS: " << (1000.0 / avg_ms) << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "\n=== Shakal Unit Tests ===" << std::endl;

    testCosineSimilarity();
    testDatabase();
    testTimer();
    testFPSCounter();

    if (argc > 1) {
        benchmarkDetector(argv[1]);
    } else {
        benchmarkDetector("models/face_detection.onnx");
    }

    std::cout << "\n=== All Tests Passed ===" << std::endl;
    return 0;
}

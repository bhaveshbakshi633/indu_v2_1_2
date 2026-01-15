#include "pipeline/Pipeline.hpp"
#include "gpu/CudaUtils.hpp"
#include "utils/Logger.hpp"
#include <iostream>
#include <string>

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nOptions:\n"
              << "  -c, --config <file>      Config file path (default: config/config.yaml)\n"
              << "  -r, --resolution <res>   Resolution: 720p, 1080p, 1440p, 2160p\n"
              << "  -f, --fps <fps>          Target FPS (default: 30)\n"
              << "  --headless               Text output only, no GUI window\n"
              << "  -g, --gpu-info           Print GPU information and exit\n"
              << "  -h, --help               Show this help message\n"
              << "\nExamples:\n"
              << "  " << program << " -r 1080p -f 60              # 1920x1080 @ 60fps\n"
              << "  " << program << " -r 720p                      # 1280x720 @ 30fps\n"
              << "  " << program << " --headless                   # Text output: Tanay, Bhavesh\n"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string config_file = "config/config.yaml";
    int display_width = 0;
    int display_height = 0;
    int target_fps = 30;
    bool headless = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-g" || arg == "--gpu-info") {
            if (shakal::CudaUtils::isAvailable()) {
                shakal::CudaUtils::printDeviceInfo(0);
            } else {
                std::cout << "CUDA not available on this system" << std::endl;
            }
            return 0;
        }
        else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_file = argv[++i];
        }
        else if ((arg == "-r" || arg == "--resolution") && i + 1 < argc) {
            std::string res = argv[++i];
            if (res == "720p") {
                display_width = 1280;
                display_height = 720;
            } else if (res == "1080p") {
                display_width = 1920;
                display_height = 1080;
            } else if (res == "1440p") {
                display_width = 2560;
                display_height = 1440;
            } else if (res == "2160p") {
                display_width = 3840;
                display_height = 2160;
            } else {
                std::cerr << "Invalid resolution. Use: 720p, 1080p, 1440p, 2160p\n";
                return 1;
            }
        }
        else if ((arg == "-f" || arg == "--fps") && i + 1 < argc) {
            target_fps = std::stoi(argv[++i]);
            if (target_fps < 1 || target_fps > 120) {
                std::cerr << "Invalid FPS. Use 1-120\n";
                return 1;
            }
        }
        else if (arg == "--headless") {
            headless = true;
        }
    }

    if (shakal::CudaUtils::isAvailable()) {
        LOG_INFO("CUDA available - GPU acceleration enabled");
        shakal::CudaUtils::printDeviceInfo(0);
    } else {
        LOG_INFO("Running on CPU");
    }

    shakal::Pipeline pipeline;

    if (!pipeline.init(config_file)) {
        LOG_ERROR("Failed to initialize pipeline");
        return 1;
    }

    // Set resolution override if specified (affects both capture and display)
    if (display_width > 0 && display_height > 0) {
        pipeline.setResolution(display_width, display_height);
    }

    // Set FPS
    pipeline.setFPS(target_fps);

    // Set headless mode if specified
    if (headless) {
        pipeline.setHeadless(true);
        LOG_INFO("Headless mode enabled - text output only");
    } else {
        LOG_INFO("Press 'q' or ESC to quit");
    }

    pipeline.run();

    return 0;
}

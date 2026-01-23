/**
 * SLAM Control CLI for G1 Robot
 * Uses Unitree SDK to control SLAM operations and get pose
 *
 * Usage:
 *   ./slam_control start_mapping
 *   ./slam_control stop_mapping [map_path]
 *   ./slam_control relocate [map_path]
 *   ./slam_control goto <x> <y> <z> <qw> <qx> <qy> <qz>
 *   ./slam_control pause
 *   ./slam_control resume
 *   ./slam_control pose              # Get current pose (JSON output)
 */

#include <iostream>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/idl/ros2/String_.hpp>

using namespace unitree::robot;
using namespace unitree::common;

// SLAM API IDs
constexpr int32_t API_START_MAPPING = 1801;
constexpr int32_t API_STOP_MAPPING = 1802;
constexpr int32_t API_START_RELOCATION = 1804;
constexpr int32_t API_POSE_NAV = 1102;
constexpr int32_t API_PAUSE_NAV = 1201;
constexpr int32_t API_RESUME_NAV = 1202;

// Topic names
#define SlamInfoTopic "rt/slam_info"

// Default map path
const std::string DEFAULT_MAP_PATH = "/home/unitree/default_map.pcd";

// Global pose storage
std::atomic<bool> g_poseReceived{false};
std::string g_currentPoseJson = "{}";
std::mutex g_poseMutex;

class SlamClient : public Client {
public:
    SlamClient() : Client("slam_operate", false) {}

    void Init() {
        SetApiVersion("1.0.0.0");
        RegistApi(API_START_MAPPING, 0);
        RegistApi(API_STOP_MAPPING, 0);
        RegistApi(API_START_RELOCATION, 0);
        RegistApi(API_POSE_NAV, 0);
        RegistApi(API_PAUSE_NAV, 0);
        RegistApi(API_RESUME_NAV, 0);
    }

    int32_t StartMapping() {
        std::string param = R"({"data": {"slam_type": "indoor"}})";
        std::string response;
        int32_t ret = Call(API_START_MAPPING, param, response);
        std::cout << "StartMapping response: " << response << std::endl;
        return ret;
    }

    int32_t StopMapping(const std::string& mapPath) {
        std::string param = R"({"data": {"address": ")" + mapPath + R"("}})";
        std::string response;
        int32_t ret = Call(API_STOP_MAPPING, param, response);
        std::cout << "StopMapping response: " << response << std::endl;
        return ret;
    }

    int32_t StartRelocation(const std::string& mapPath) {
        std::string param = R"({"data": {"x": 0.0, "y": 0.0, "z": 0.0, "q_x": 0.0, "q_y": 0.0, "q_z": 0.0, "q_w": 1.0, "address": ")" + mapPath + R"("}})";
        std::string response;
        int32_t ret = Call(API_START_RELOCATION, param, response);
        std::cout << "StartRelocation response: " << response << std::endl;
        return ret;
    }

    int32_t GotoPose(double x, double y, double z, double qw, double qx, double qy, double qz) {
        char buf[512];
        snprintf(buf, sizeof(buf),
            R"({"data": {"targetPose": {"x": %f, "y": %f, "z": %f, "q_x": %f, "q_y": %f, "q_z": %f, "q_w": %f}, "mode": 1}})",
            x, y, z, qx, qy, qz, qw);
        std::string param(buf);
        std::string response;
        int32_t ret = Call(API_POSE_NAV, param, response);
        std::cout << "GotoPose response: " << response << std::endl;
        return ret;
    }

    int32_t Pause() {
        std::string param = R"({"data": {}})";
        std::string response;
        int32_t ret = Call(API_PAUSE_NAV, param, response);
        std::cout << "Pause response: " << response << std::endl;
        return ret;
    }

    int32_t Resume() {
        std::string param = R"({"data": {}})";
        std::string response;
        int32_t ret = Call(API_RESUME_NAV, param, response);
        std::cout << "Resume response: " << response << std::endl;
        return ret;
    }
};

// Callback for slam_info topic
void SlamInfoHandler(const void* message) {
    if (message) {
        const std_msgs::msg::dds_::String_* msg =
            static_cast<const std_msgs::msg::dds_::String_*>(message);
        std::lock_guard<std::mutex> lock(g_poseMutex);
        g_currentPoseJson = msg->data();
        g_poseReceived = true;
    }
}

// Get current pose by subscribing to slam_info
std::string GetCurrentPose(int timeoutMs = 3000) {
    // Reset state
    g_poseReceived = false;

    // Create subscriber for slam_info using correct message type
    ChannelSubscriber<std_msgs::msg::dds_::String_> subscriber(SlamInfoTopic);
    subscriber.InitChannel(SlamInfoHandler, 1);

    // Wait for pose data
    auto start = std::chrono::steady_clock::now();
    while (!g_poseReceived) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed > timeoutMs) {
            return R"({"error": "timeout", "message": "No pose data received"})";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::lock_guard<std::mutex> lock(g_poseMutex);
    return g_currentPoseJson;
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <command> [args]\n"
              << "\nCommands:\n"
              << "  start_mapping              - Start SLAM mapping\n"
              << "  stop_mapping [path]        - Stop mapping and save (default: " << DEFAULT_MAP_PATH << ")\n"
              << "  relocate [path]            - Load map and start navigation mode\n"
              << "  goto <x> <y> <z> <qw> <qx> <qy> <qz> - Navigate to pose\n"
              << "  pause                      - Pause navigation\n"
              << "  resume                     - Resume navigation\n"
              << "  pose                       - Get current pose (JSON)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    // Initialize channel factory (use eth0 for robot internal DDS communication)
    ChannelFactory::Instance()->Init(0, "eth0");

    int32_t ret = 0;

    if (command == "pose") {
        // Special case: just get pose, no API client needed
        std::string pose = GetCurrentPose(3000);
        std::cout << pose << std::endl;
        ChannelFactory::Instance()->Release();
        return g_poseReceived ? 0 : 1;
    }

    // For other commands, create API client
    SlamClient client;
    client.SetTimeout(10.0f);
    client.Init();

    if (command == "start_mapping") {
        ret = client.StartMapping();
    }
    else if (command == "stop_mapping") {
        std::string mapPath = (argc > 2) ? argv[2] : DEFAULT_MAP_PATH;
        ret = client.StopMapping(mapPath);
    }
    else if (command == "relocate") {
        std::string mapPath = (argc > 2) ? argv[2] : DEFAULT_MAP_PATH;
        ret = client.StartRelocation(mapPath);
    }
    else if (command == "goto") {
        if (argc < 9) {
            std::cerr << "Error: goto requires 7 arguments: x y z qw qx qy qz\n";
            ChannelFactory::Instance()->Release();
            return 1;
        }
        double x = std::stod(argv[2]);
        double y = std::stod(argv[3]);
        double z = std::stod(argv[4]);
        double qw = std::stod(argv[5]);
        double qx = std::stod(argv[6]);
        double qy = std::stod(argv[7]);
        double qz = std::stod(argv[8]);
        ret = client.GotoPose(x, y, z, qw, qx, qy, qz);
    }
    else if (command == "pause") {
        ret = client.Pause();
    }
    else if (command == "resume") {
        ret = client.Resume();
    }
    else {
        std::cerr << "Unknown command: " << command << std::endl;
        printUsage(argv[0]);
        ChannelFactory::Instance()->Release();
        return 1;
    }

    if (ret == 0) {
        std::cout << "Command '" << command << "' executed successfully.\n";
    } else {
        std::cerr << "Command '" << command << "' failed with code: " << ret << std::endl;
    }

    ChannelFactory::Instance()->Release();
    return ret;
}

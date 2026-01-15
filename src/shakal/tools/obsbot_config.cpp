/**
 * OBSBOT Tiny 2 Configuration Tool
 * Sets FOV to widest (86°) and Zoom to minimum (1.0)
 * Run this before starting shakal for optimal field of view
 */

#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <vector>

#include <dev/devs.hpp>

std::vector<std::string> detected_devices;
std::shared_ptr<Device> camera;
bool device_connected = false;
std::string connected_sn;

void onDeviceChanged(std::string dev_sn, bool connected, void* param)
{
    if (connected) {
        device_connected = true;
        connected_sn = dev_sn;
        auto it = std::find(detected_devices.begin(), detected_devices.end(), dev_sn);
        if (it == detected_devices.end()) {
            detected_devices.push_back(dev_sn);
        }
    } else {
        detected_devices.erase(
            std::remove(detected_devices.begin(), detected_devices.end(), dev_sn),
            detected_devices.end()
        );
    }
}

void printUsage(const char* prog)
{
    std::cout << "OBSBOT Tiny 2 Configuration Tool\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --fov <wide|medium|narrow>  Set FOV (default: wide/86°)\n";
    std::cout << "  --zoom <1.0-2.0>            Set zoom level (default: 1.0)\n";
    std::cout << "  --info                      Print camera info only\n";
    std::cout << "  --help                      Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << "                      # Reset to widest FOV + no zoom\n";
    std::cout << "  " << prog << " --fov medium         # Set 78° FOV\n";
    std::cout << "  " << prog << " --zoom 1.5           # Set 1.5x zoom\n";
}

int main(int argc, char** argv)
{
    Device::FovType target_fov = Device::FovType86;  // Default: widest
    float target_zoom = 1.0f;                         // Default: no zoom
    bool info_only = false;
    std::string fov_name = "wide (86°)";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--info") {
            info_only = true;
        }
        else if (arg == "--fov" && i + 1 < argc) {
            std::string fov = argv[++i];
            if (fov == "wide" || fov == "86") {
                target_fov = Device::FovType86;
                fov_name = "wide (86°)";
            } else if (fov == "medium" || fov == "78") {
                target_fov = Device::FovType78;
                fov_name = "medium (78°)";
            } else if (fov == "narrow" || fov == "65") {
                target_fov = Device::FovType65;
                fov_name = "narrow (65°)";
            } else {
                std::cerr << "Invalid FOV: " << fov << " (use: wide, medium, narrow)\n";
                return 1;
            }
        }
        else if (arg == "--zoom" && i + 1 < argc) {
            target_zoom = std::stof(argv[++i]);
            if (target_zoom < 1.0f || target_zoom > 2.0f) {
                std::cerr << "Zoom must be between 1.0 and 2.0\n";
                return 1;
            }
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "OBSBOT Tiny 2 Configuration Tool\n";
    std::cout << "================================\n";

    // Register callback
    Devices::get().setDevChangedCallback(onDeviceChanged, nullptr);
    Devices::get().setEnableMdnsScan(false);

    // Wait for SDK callback to report device connected
    std::cout << "Scanning for OBSBOT Tiny 2...\n";

    int wait_count = 0;
    int max_wait = 50;  // 5 seconds max
    while (!device_connected && wait_count < max_wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    if (device_connected) {
        // Small delay to ensure device is fully initialized
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Get device by SN from callback
        camera = Devices::get().getDevBySn(connected_sn);
        if (camera) {
            std::cout << "Device connected: " << connected_sn << "\n";
        }
    }

    // Fallback: check device list
    if (!camera) {
        auto dev_list = Devices::get().getDevList();
        for (const auto& device : dev_list) {
            if (device->productType() == ObsbotProdTiny2) {
                camera = device;
                break;
            }
        }
    }

    if (!camera) {
        std::cerr << "ERROR: OBSBOT Tiny 2 not found!\n";
        std::cerr << "Make sure the camera is connected via USB.\n";
        Devices::get().close();
        return 1;
    }

    std::cout << "Found: " << camera->devName() << "\n";
    std::cout << "  SN: " << camera->devSn() << "\n";
    std::cout << "  Version: " << camera->devVersion() << "\n";

    if (info_only) {
        Devices::get().close();
        return 0;
    }

    // Wake up camera if sleeping
    std::cout << "\nWaking up camera...\n";
    camera->cameraSetDevRunStatusR(Device::DevStatusRun);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Set FOV
    std::cout << "Setting FOV to " << fov_name << "...\n";
    int32_t fov_result = camera->cameraSetFovU(target_fov);
    if (fov_result != 0) {
        std::cerr << "WARNING: FOV setting returned error code: " << fov_result << "\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Set Zoom
    std::cout << "Setting zoom to " << target_zoom << "x...\n";
    int32_t zoom_result = camera->cameraSetZoomAbsoluteR(target_zoom);
    if (zoom_result != 0) {
        std::cerr << "WARNING: Zoom setting returned error code: " << zoom_result << "\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    std::cout << "\nConfiguration complete!\n";
    std::cout << "  FOV: " << fov_name << "\n";
    std::cout << "  Zoom: " << target_zoom << "x\n";

    Devices::get().close();
    return 0;
}

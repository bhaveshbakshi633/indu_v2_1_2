#pragma once

#include <string>

namespace shakal {

struct GpuInfo {
    std::string name;
    int device_id;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    bool tensorrt_available;
};

class CudaUtils {
public:
    // Runtime GPU detection (no compile-time flags needed)
    static bool isAvailable();           // OpenCV CUDA check (preferred)
    static bool isRawCudaAvailable();    // Direct CUDA runtime check
    static int getDeviceCount();
    static bool setDevice(int device_id);

    static GpuInfo getDeviceInfo(int device_id = 0);
    static void printDeviceInfo(int device_id = 0);

    static bool isTensorRTAvailable();

    static size_t getFreeMemory(int device_id = 0);
    static size_t getTotalMemory(int device_id = 0);

    static void synchronize();
    static void resetDevice();

private:
    CudaUtils() = delete;
};

}

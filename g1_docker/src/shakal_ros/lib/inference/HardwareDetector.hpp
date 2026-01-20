#pragma once

#include <string>
#include <vector>

namespace shakal {

struct HardwareCapabilities {
    // GPU
    bool has_cuda = false;
    bool has_cudnn = false;
    bool has_tensorrt = false;
    bool has_opencl = false;

    // GPU details
    int cuda_device_count = 0;
    int compute_capability = 0;  // e.g., 86 for sm_86
    std::string gpu_name;
    size_t gpu_memory_mb = 0;

    // TensorRT details
    std::string tensorrt_version;

    // CPU
    std::string cpu_vendor;
    int cpu_cores = 0;
};

class HardwareDetector {
public:
    // Detect all hardware capabilities at runtime
    static HardwareCapabilities detect();

    // Individual checks (runtime, no compile flags needed)
    static bool hasCuda();
    static bool hasTensorRT();
    static bool hasCuDNN();
    static bool hasOpenCL();

    // Get compute capability (e.g., 86 for RTX A2000, 87 for Orin)
    static int getComputeCapability();

    // Get GPU name
    static std::string getGpuName();

    // Get TensorRT version string (empty if not available)
    static std::string getTensorRTVersion();

    // Print summary
    static void printCapabilities();

private:
    HardwareDetector() = delete;

    // Runtime library detection using dlopen
    static bool checkLibrary(const std::vector<std::string>& lib_names);
};

} // namespace shakal

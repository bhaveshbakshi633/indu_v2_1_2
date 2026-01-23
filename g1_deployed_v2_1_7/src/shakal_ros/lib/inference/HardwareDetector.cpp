#include "inference/HardwareDetector.hpp"
#include "utils/Logger.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <thread>
#include <iostream>

namespace shakal {

bool HardwareDetector::checkLibrary(const std::vector<std::string>& lib_names) {
    for (const auto& lib : lib_names) {
        void* handle = dlopen(lib.c_str(), RTLD_LAZY | RTLD_NOLOAD);
        if (handle) {
            dlclose(handle);
            return true;
        }
        // Try loading it
        handle = dlopen(lib.c_str(), RTLD_LAZY);
        if (handle) {
            dlclose(handle);
            return true;
        }
    }
    return false;
}

bool HardwareDetector::hasCuda() {
    try {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    } catch (...) {
        return false;
    }
}

bool HardwareDetector::hasTensorRT() {
    // Check for TensorRT library at runtime
    std::vector<std::string> libs = {
        "libnvinfer.so",
        "libnvinfer.so.8",
        "libnvinfer.so.10",
        "libnvinfer.so.7"
    };
    return checkLibrary(libs);
}

bool HardwareDetector::hasCuDNN() {
    // Check for cuDNN library at runtime
    std::vector<std::string> libs = {
        "libcudnn.so",
        "libcudnn.so.8",
        "libcudnn.so.9"
    };
    return checkLibrary(libs);
}

bool HardwareDetector::hasOpenCL() {
    try {
        cv::ocl::setUseOpenCL(true);
        return cv::ocl::haveOpenCL();
    } catch (...) {
        return false;
    }
}

int HardwareDetector::getComputeCapability() {
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            cv::cuda::DeviceInfo dev_info(0);
            return dev_info.majorVersion() * 10 + dev_info.minorVersion();
        }
    } catch (...) {}
    return 0;
}

std::string HardwareDetector::getGpuName() {
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            cv::cuda::DeviceInfo dev_info(0);
            return dev_info.name();
        }
    } catch (...) {}
    return "";
}

std::string HardwareDetector::getTensorRTVersion() {
    if (!hasTensorRT()) {
        return "";
    }

    // Try to get version from library
    void* handle = dlopen("libnvinfer.so", RTLD_LAZY);
    if (!handle) {
        handle = dlopen("libnvinfer.so.8", RTLD_LAZY);
    }
    if (!handle) {
        handle = dlopen("libnvinfer.so.10", RTLD_LAZY);
    }

    if (handle) {
        // Try to find getInferLibVersion
        typedef int (*GetVersionFunc)();
        GetVersionFunc getVersion = (GetVersionFunc)dlsym(handle, "getInferLibVersion");
        if (getVersion) {
            int version = getVersion();
            int major = version / 1000;
            int minor = (version % 1000) / 100;
            int patch = version % 100;
            dlclose(handle);

            std::stringstream ss;
            ss << major << "." << minor << "." << patch;
            return ss.str();
        }
        dlclose(handle);
    }

    return "detected";
}

HardwareCapabilities HardwareDetector::detect() {
    HardwareCapabilities caps;

    // GPU detection
    caps.has_cuda = hasCuda();
    caps.has_cudnn = hasCuDNN();
    caps.has_tensorrt = hasTensorRT();
    caps.has_opencl = hasOpenCL();

    if (caps.has_cuda) {
        try {
            caps.cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
            cv::cuda::DeviceInfo dev_info(0);
            caps.compute_capability = dev_info.majorVersion() * 10 + dev_info.minorVersion();
            caps.gpu_name = dev_info.name();
            caps.gpu_memory_mb = dev_info.totalMemory() / (1024 * 1024);
        } catch (...) {}
    }

    if (caps.has_tensorrt) {
        caps.tensorrt_version = getTensorRTVersion();
    }

    // CPU info
    caps.cpu_cores = std::thread::hardware_concurrency();

    // Try to get CPU vendor
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("vendor_id") != std::string::npos ||
                line.find("Hardware") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    caps.cpu_vendor = line.substr(pos + 2);
                    break;
                }
            }
        }
    }

    return caps;
}

void HardwareDetector::printCapabilities() {
    auto caps = detect();

    std::cout << "\n=== Hardware Capabilities ===" << std::endl;
    std::cout << "CUDA:         " << (caps.has_cuda ? "Yes" : "No");
    if (caps.has_cuda) {
        std::cout << " (" << caps.cuda_device_count << " device(s))";
    }
    std::cout << std::endl;

    if (caps.has_cuda) {
        std::cout << "GPU:          " << caps.gpu_name << std::endl;
        std::cout << "Compute Cap:  sm_" << caps.compute_capability << std::endl;
        std::cout << "GPU Memory:   " << caps.gpu_memory_mb << " MB" << std::endl;
    }

    std::cout << "cuDNN:        " << (caps.has_cudnn ? "Yes" : "No") << std::endl;
    std::cout << "TensorRT:     " << (caps.has_tensorrt ? "Yes" : "No");
    if (caps.has_tensorrt && !caps.tensorrt_version.empty()) {
        std::cout << " (v" << caps.tensorrt_version << ")";
    }
    std::cout << std::endl;

    std::cout << "OpenCL:       " << (caps.has_opencl ? "Yes" : "No") << std::endl;
    std::cout << "CPU Cores:    " << caps.cpu_cores << std::endl;
    if (!caps.cpu_vendor.empty()) {
        std::cout << "CPU Vendor:   " << caps.cpu_vendor << std::endl;
    }
    std::cout << std::endl;
}

} // namespace shakal

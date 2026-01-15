#include "gpu/CudaUtils.hpp"
#include "utils/Logger.hpp"
#include <opencv2/core/cuda.hpp>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace shakal {

// Primary check - uses OpenCV's CUDA detection (no compile flags needed)
bool CudaUtils::isAvailable() {
    try {
        int count = cv::cuda::getCudaEnabledDeviceCount();
        return count > 0;
    } catch (...) {
        return false;
    }
}

// Secondary check - direct CUDA runtime (needs USE_CUDA flag)
bool CudaUtils::isRawCudaAvailable() {
#ifdef USE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

int CudaUtils::getDeviceCount() {
    try {
        return cv::cuda::getCudaEnabledDeviceCount();
    } catch (...) {
        return 0;
    }
}

bool CudaUtils::setDevice(int device_id) {
#ifdef USE_CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to set CUDA device: " + std::string(cudaGetErrorString(err)));
        return false;
    }
    return true;
#else
    (void)device_id;
    LOG_WARN("CUDA not available");
    return false;
#endif
}

GpuInfo CudaUtils::getDeviceInfo(int device_id) {
    GpuInfo info{};
    info.device_id = device_id;

    // Try OpenCV CUDA info first
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            cv::cuda::DeviceInfo dev_info(device_id);
            info.name = dev_info.name();
            info.total_memory = dev_info.totalMemory();
            info.free_memory = dev_info.freeMemory();
            info.compute_capability_major = dev_info.majorVersion();
            info.compute_capability_minor = dev_info.minorVersion();
            info.tensorrt_available = isTensorRTAvailable();
            return info;
        }
    } catch (...) {
        // OpenCV CUDA not available, try raw CUDA
    }

#ifdef USE_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        info.name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory = free_mem;

        info.tensorrt_available = isTensorRTAvailable();
    }
#else
    info.name = "N/A (CUDA not available)";
#endif

    return info;
}

void CudaUtils::printDeviceInfo(int device_id) {
    GpuInfo info = getDeviceInfo(device_id);

    std::cout << "\n=== GPU Info ===" << std::endl;
    std::cout << "Device ID:    " << info.device_id << std::endl;
    std::cout << "Name:         " << info.name << std::endl;
    std::cout << "Compute Cap:  " << info.compute_capability_major << "."
              << info.compute_capability_minor << std::endl;
    std::cout << "Total Memory: " << (info.total_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Free Memory:  " << (info.free_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "TensorRT:     " << (info.tensorrt_available ? "Yes" : "No") << std::endl;
}

bool CudaUtils::isTensorRTAvailable() {
#ifdef USE_TENSORRT
    return true;
#else
    return false;
#endif
}

size_t CudaUtils::getFreeMemory(int device_id) {
#ifdef USE_CUDA
    setDevice(device_id);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
#else
    (void)device_id;
    return 0;
#endif
}

size_t CudaUtils::getTotalMemory(int device_id) {
#ifdef USE_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        return prop.totalGlobalMem;
    }
    return 0;
#else
    (void)device_id;
    return 0;
#endif
}

void CudaUtils::synchronize() {
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
}

void CudaUtils::resetDevice() {
#ifdef USE_CUDA
    cudaDeviceReset();
#endif
}

}

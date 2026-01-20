#include "utils/Timer.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace shakal {

Timer::Timer() : running_(false) {
}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    running_ = true;
}

void Timer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    running_ = false;
}

void Timer::reset() {
    running_ = false;
}

double Timer::elapsedMs() const {
    auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    return std::chrono::duration<double, std::milli>(end - start_time_).count();
}

double Timer::elapsedUs() const {
    auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    return std::chrono::duration<double, std::micro>(end - start_time_).count();
}

double Timer::elapsedSec() const {
    return elapsedMs() / 1000.0;
}

FPSCounter::FPSCounter(int smoothing_window)
    : fps_(0.0f)
    , first_tick_(true) {
    alpha_ = 2.0f / (smoothing_window + 1);
}

void FPSCounter::tick() {
    auto now = std::chrono::high_resolution_clock::now();

    if (first_tick_) {
        first_tick_ = false;
    } else {
        double delta = std::chrono::duration<double>(now - last_time_).count();
        if (delta > 0) {
            float instant_fps = 1.0f / static_cast<float>(delta);
            fps_ = alpha_ * instant_fps + (1.0f - alpha_) * fps_;
        }
    }

    last_time_ = now;
}

float FPSCounter::getFPS() const {
    return fps_;
}

void FPSCounter::reset() {
    first_tick_ = true;
    fps_ = 0.0f;
}

ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name) {
    timer_.start();
}

ScopedTimer::~ScopedTimer() {
    timer_.stop();
    ProfilerManager::instance().addTiming(name_, timer_.elapsedMs());
}

ProfilerManager& ProfilerManager::instance() {
    static ProfilerManager instance;
    return instance;
}

void ProfilerManager::addTiming(const std::string& name, double ms) {
    auto& s = stats_[name];
    s.total_ms += ms;
    s.min_ms = std::min(s.min_ms, ms);
    s.max_ms = std::max(s.max_ms, ms);
    s.count++;
}

void ProfilerManager::printStats() {
    std::cout << "\n=== Profiler Stats ===" << std::endl;
    std::cout << std::setw(30) << "Name"
              << std::setw(12) << "Count"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    for (const auto& [name, s] : stats_) {
        double avg = s.count > 0 ? s.total_ms / s.count : 0;
        std::cout << std::setw(30) << name
                  << std::setw(12) << s.count
                  << std::setw(12) << std::fixed << std::setprecision(2) << avg
                  << std::setw(12) << s.min_ms
                  << std::setw(12) << s.max_ms
                  << std::endl;
    }
}

void ProfilerManager::reset() {
    stats_.clear();
}

}

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace shakal {

class Timer {
public:
    Timer();

    void start();
    void stop();
    void reset();

    double elapsedMs() const;
    double elapsedUs() const;
    double elapsedSec() const;

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_;
};

class FPSCounter {
public:
    FPSCounter(int smoothing_window = 30);

    void tick();
    float getFPS() const;
    void reset();

private:
    std::chrono::high_resolution_clock::time_point last_time_;
    float fps_;
    float alpha_;
    bool first_tick_;
};

class ScopedTimer {
public:
    ScopedTimer(const std::string& name);
    ~ScopedTimer();

private:
    std::string name_;
    Timer timer_;
};

class ProfilerManager {
public:
    static ProfilerManager& instance();

    void addTiming(const std::string& name, double ms);
    void printStats();
    void reset();

private:
    ProfilerManager() = default;

    struct Stats {
        double total_ms = 0;
        double min_ms = 1e9;
        double max_ms = 0;
        int count = 0;
    };

    std::unordered_map<std::string, Stats> stats_;
};

#define PROFILE_SCOPE(name) shakal::ScopedTimer _timer_##__LINE__(name)

}

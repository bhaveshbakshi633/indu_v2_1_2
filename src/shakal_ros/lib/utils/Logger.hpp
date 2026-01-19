#pragma once

#include <string>
#include <mutex>

namespace shakal {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

class Logger {
public:
    static Logger& instance();

    void setLevel(LogLevel level);
    void setPrefix(const std::string& prefix);

    void debug(const std::string& msg);
    void info(const std::string& msg);
    void warn(const std::string& msg);
    void error(const std::string& msg);

private:
    Logger();
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void log(LogLevel level, const std::string& msg);
    std::string levelToString(LogLevel level);
    std::string getTimestamp();

    LogLevel level_;
    std::string prefix_;
    std::mutex mutex_;
};

#define LOG_DEBUG(msg) shakal::Logger::instance().debug(msg)
#define LOG_INFO(msg) shakal::Logger::instance().info(msg)
#define LOG_WARN(msg) shakal::Logger::instance().warn(msg)
#define LOG_ERROR(msg) shakal::Logger::instance().error(msg)

}

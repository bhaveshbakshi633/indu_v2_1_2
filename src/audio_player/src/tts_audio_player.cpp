/**
 * TTS Audio Player Node for Maya Demo
 *
 * Enhanced features:
 * - /g1/tts/speaking (Bool) - Published when TTS is playing
 * - /g1/tts/stop (Empty) - Subscribe to stop playback immediately
 * - Volume set to 100% after initialization
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>

#include "g1/g1_audio_client.hpp"

class TTSAudioPlayer : public rclcpp::Node {
public:
    TTSAudioPlayer() : Node("tts_audio_player"), is_speaking_(false), stop_requested_(false), volume_set_(false) {
        // Initialize audio client
        audio_client_ = std::make_shared<unitree::ros2::g1::AudioClient>();

        // Subscribe to audio data
        audio_subscription_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>(
            "/g1/tts/audio_output",
            10,
            std::bind(&TTSAudioPlayer::audioCallback, this, std::placeholders::_1));

        // Subscribe to stop command
        stop_subscription_ = this->create_subscription<std_msgs::msg::Empty>(
            "/g1/tts/stop",
            10,
            std::bind(&TTSAudioPlayer::stopCallback, this, std::placeholders::_1));

        // Publisher for speaking state
        speaking_publisher_ = this->create_publisher<std_msgs::msg::Bool>(
            "/g1/tts/speaking",
            10);

        // Parameters
        this->declare_parameter("app_name", "edge_tts");
        this->declare_parameter("auto_play", true);

        app_name_ = this->get_parameter("app_name").as_string();

        // Publish initial speaking state (false)
        publishSpeakingState(false);

        // Timer to periodically publish speaking state and set volume
        state_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&TTSAudioPlayer::stateTimerCallback, this));

        // One-shot timer to set volume after 2 seconds (gives time for AudioClient to initialize)
        volume_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&TTSAudioPlayer::setVolumeCallback, this));

        RCLCPP_INFO(this->get_logger(), "TTS Audio Player initialized");
        RCLCPP_INFO(this->get_logger(), "Listening for audio on /g1/tts/audio_output");
        RCLCPP_INFO(this->get_logger(), "Publishing speaking state on /g1/tts/speaking");
        RCLCPP_INFO(this->get_logger(), "Listening for stop on /g1/tts/stop");
        RCLCPP_INFO(this->get_logger(), "Volume will be set to 100%% in 2 seconds...");
    }

    void setVolumeCallback() {
        if (!volume_set_) {
            int32_t vol_ret = audio_client_->SetVolume(100);
            if (vol_ret == 0) {
                RCLCPP_INFO(this->get_logger(), "Volume set to 100%%");
                volume_set_ = true;
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to set volume (ret: %d), will retry...", vol_ret);
            }
        }
        // Cancel the timer after successful set or a few tries
        static int retry_count = 0;
        retry_count++;
        if (volume_set_ || retry_count > 5) {
            volume_timer_->cancel();
        }
    }

    void audioCallback(const std_msgs::msg::UInt8MultiArray::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received audio data: %zu bytes", msg->data.size());

        // Check if stop was requested
        if (stop_requested_.load()) {
            RCLCPP_INFO(this->get_logger(), "Stop requested, ignoring audio");
            stop_requested_.store(false);
            return;
        }

        try {
            // Set speaking state to true
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                is_speaking_ = true;
            }
            publishSpeakingState(true);

            // Convert message data to vector<uint8_t>
            std::vector<uint8_t> pcm_data(msg->data.begin(), msg->data.end());

            // Generate unique stream ID using timestamp
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
            current_stream_id_ = std::to_string(timestamp);

            // Play the audio stream
            int32_t ret = audio_client_->PlayStream(app_name_, current_stream_id_, pcm_data);

            if (ret == 0) {
                RCLCPP_INFO(this->get_logger(), "Playing audio stream (ID: %s)", current_stream_id_.c_str());

                // Estimate playback duration (16kHz, 16-bit mono = 32000 bytes/sec)
                double duration_sec = static_cast<double>(pcm_data.size()) / 32000.0;
                int duration_ms = static_cast<int>(duration_sec * 1000) + 300; // +300ms buffer

                // Wait for playback to complete (in background thread)
                std::thread([this, duration_ms]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

                    // Check if stop was requested during playback
                    if (!stop_requested_.load()) {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        is_speaking_ = false;
                    }
                    publishSpeakingState(false);
                }).detach();

            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to play audio stream. Error code: %d", ret);
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    is_speaking_ = false;
                }
                publishSpeakingState(false);
            }

        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error in audio callback: %s", e.what());
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                is_speaking_ = false;
            }
            publishSpeakingState(false);
        }
    }

    void stopCallback(const std_msgs::msg::Empty::SharedPtr /*msg*/) {
        RCLCPP_INFO(this->get_logger(), "Stop command received");
        stopPlayback();
    }

    void stopPlayback() {
        stop_requested_.store(true);

        int32_t ret = audio_client_->PlayStop(app_name_);
        if (ret == 0) {
            RCLCPP_INFO(this->get_logger(), "Stopped audio playback");
        } else {
            RCLCPP_WARN(this->get_logger(), "PlayStop returned: %d", ret);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            is_speaking_ = false;
        }
        publishSpeakingState(false);

        // Reset stop flag after a short delay
        std::thread([this]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            stop_requested_.store(false);
        }).detach();
    }

    // Getter for AudioClient node (needed for multi-threaded executor)
    std::shared_ptr<unitree::ros2::g1::AudioClient> getAudioClient() {
        return audio_client_;
    }

private:
    void publishSpeakingState(bool speaking) {
        auto msg = std_msgs::msg::Bool();
        msg.data = speaking;
        speaking_publisher_->publish(msg);
    }

    void stateTimerCallback() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        publishSpeakingState(is_speaking_);
    }

    std::shared_ptr<unitree::ros2::g1::AudioClient> audio_client_;
    rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr audio_subscription_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr stop_subscription_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr speaking_publisher_;
    rclcpp::TimerBase::SharedPtr state_timer_;
    rclcpp::TimerBase::SharedPtr volume_timer_;

    std::string app_name_;
    std::string current_stream_id_;
    bool is_speaking_;
    bool volume_set_;
    std::atomic<bool> stop_requested_;
    std::mutex state_mutex_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<TTSAudioPlayer>();

        // Use multi-threaded executor to spin both TTSAudioPlayer and the inner AudioClient
        // AudioClient inherits from rclcpp::Node and needs to be spun for its subscriptions to work
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        executor.add_node(node->getAudioClient());
        executor.spin();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}

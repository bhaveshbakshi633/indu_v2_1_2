/**
 * Arm Controller - v2.8
 *
 * FIXES v2.7:
 * 1. Worker thread for long operations
 * 2. getFsmId() API call for proper FSM check
 * 3. cancel command to abort running tasks
 * 4. Topic name fix: arm_sdk
 *
 * FIXES v2.8:
 * 5. Teach mode: position error check disabled in TEACH state
 * 6. Teach mode: commanded_pos_ synced continuously
 * 7. Continuous recording with timestamps
 * 8. Play command with exact timing replay
 */

#ifndef ARM_CONTROLLER__ARM_CONTROLLER_HPP_
#define ARM_CONTROLLER__ARM_CONTROLLER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_api/msg/request.hpp>
#include <unitree_api/msg/response.hpp>
#include <std_msgs/msg/string.hpp>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional>

namespace arm_controller
{

namespace FSM {
    constexpr int ZERO_TORQUE = 0;
    constexpr int DAMP = 1;
    constexpr int STAND_UP = 4;
    constexpr int READY = 801;
}

constexpr std::array<int, 7> LEFT_ARM = {15, 16, 17, 18, 19, 20, 21};
constexpr std::array<int, 7> RIGHT_ARM = {22, 23, 24, 25, 26, 27, 28};
constexpr std::array<int, 3> WAIST = {12, 13, 14};
constexpr int G1_NUM_JOINTS = 29;
constexpr int G1_ARM_SDK_WEIGHT = 29;

constexpr double DEFAULT_ARM_KP = 50.0;
constexpr double DEFAULT_ARM_KD = 1.0;
constexpr double DEFAULT_WAIST_KP = 200.0;
constexpr double DEFAULT_WAIST_KD = 5.0;
constexpr double SLOW_COAST_KP = 0.0;
constexpr double SLOW_COAST_KD = 10.0;

constexpr double BLEND_DURATION = 3.0;
constexpr double MIN_BLEND_TIME = 0.1;

constexpr double MAX_POSITION_ERROR = 0.3;
constexpr double MAX_JOINT_JUMP = 1.5;
constexpr double MAX_JOINT_VELOCITY = 3.0;

constexpr double RAMP_DURATION = 2.0;
constexpr double START_KP = 10.0;
constexpr double END_KP = 50.0;

constexpr int CONTROL_RATE_HZ = 500;

// Recording rate - 50Hz is enough for smooth playback, saves memory
constexpr int RECORDING_RATE_HZ = 50;
constexpr int RECORDING_DIVIDER = CONTROL_RATE_HZ / RECORDING_RATE_HZ;  // 10

constexpr std::array<double, 7> HOME_LEFT = {0.4, 0.15, 0.0, 0.5, 0.0, 0.0, 0.0};
constexpr std::array<double, 7> HOME_RIGHT = {0.4, -0.15, 0.0, 0.5, 0.0, 0.0, 0.0};
constexpr std::array<double, 3> HOME_WAIST = {0.0, 0.0, 0.0};

struct JointLimit { double min; double max; };

inline const std::map<int, JointLimit> LEFT_ARM_LIMITS = {
    {15, {-3.0892, 2.6704}}, {16, {-1.5882, 2.2515}}, {17, {-2.618, 2.618}},
    {18, {-1.0472, 2.0944}}, {19, {-1.9722, 1.9722}}, {20, {-1.6144, 1.6144}}, {21, {-1.6144, 1.6144}},
};

inline const std::map<int, JointLimit> RIGHT_ARM_LIMITS = {
    {22, {-3.0892, 2.6704}}, {23, {-2.2515, 1.5882}}, {24, {-2.618, 2.618}},
    {25, {-1.0472, 2.0944}}, {26, {-1.9722, 1.9722}}, {27, {-1.6144, 1.6144}}, {28, {-1.6144, 1.6144}},
};

inline const std::map<int, JointLimit> WAIST_LIMITS = {
    {12, {-2.618, 2.618}}, {13, {-0.52, 0.52}}, {14, {-0.52, 0.52}},
};

inline std::map<int, JointLimit> getAllJointLimits() {
    std::map<int, JointLimit> all;
    for (const auto& [k, v] : LEFT_ARM_LIMITS) all[k] = v;
    for (const auto& [k, v] : RIGHT_ARM_LIMITS) all[k] = v;
    for (const auto& [k, v] : WAIST_LIMITS) all[k] = v;
    return all;
}

inline const std::map<int, std::string> JOINT_NAMES = {
    {12, "Waist_Yaw"}, {13, "Waist_Roll"}, {14, "Waist_Pitch"},
    {15, "L_Shoulder_Pitch"}, {16, "L_Shoulder_Roll"}, {17, "L_Shoulder_Yaw"},
    {18, "L_Elbow"}, {19, "L_Wrist_Roll"}, {20, "L_Wrist_Pitch"}, {21, "L_Wrist_Yaw"},
    {22, "R_Shoulder_Pitch"}, {23, "R_Shoulder_Roll"}, {24, "R_Shoulder_Yaw"},
    {25, "R_Elbow"}, {26, "R_Wrist_Roll"}, {27, "R_Wrist_Pitch"}, {28, "R_Wrist_Yaw"},
};

enum class BootState { DISCONNECTED, INIT, WAIT_LOWSTATE, DAMPING, STANDING_UP, READY, HOLDING, MOVING_TO_HOME, AT_HOME, ERROR };
enum class ArmState { IDLE, HOLDING, MOVING, TEACH };

inline double smoothstep(double t) {
    t = std::max(0.0, std::min(1.0, t));
    return t * t * (3.0 - 2.0 * t);
}

inline double smoothstepVelocity(double t, double T) {
    if (T <= 0) return 0.0;
    t = std::max(0.0, std::min(1.0, t));
    return 6.0 * t * (1.0 - t) / T;
}

// Trajectory point - timestamp (seconds from start) + joint positions
struct TrajectoryPoint {
    double timestamp;
    std::map<int, double> positions;
};

class ArmController : public rclcpp::Node
{
public:
    ArmController();
    ~ArmController();

    bool initArms();
    bool moveToHome(double blend_time = BLEND_DURATION);
    bool moveTo(const std::map<int, double>& target, double blend_time = BLEND_DURATION);
    bool executeSequence(const std::string& json_sequence);
    bool stop();
    bool cancel();

    // Teach mode
    bool enterTeach();
    bool exitTeach();

    // Recording - NEW v2.8
    bool startRecording();
    bool stopRecording();
    bool playTrajectory();

    void triggerDamp(const std::string& reason);

    ArmState getState() const;
    std::string getStateName() const;
    bool isReady() const;
    bool isMoving() const;
    std::map<int, double> getCurrentPositions() const;
    std::map<int, double> getCommandedPositions() const;

private:
    // ROS2 Communication
    rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_sub_;
    rclcpp::Subscription<unitree_api::msg::Response>::SharedPtr api_response_sub_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr arm_sdk_pub_;
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr api_request_pub_;

    // FSM API
    std::map<uint64_t, unitree_api::msg::Response> response_cache_;
    std::mutex response_mutex_;
    std::atomic<int> cached_fsm_{-1};

    int32_t getFsmId(int& fsm_id);
    void apiResponseCallback(const unitree_api::msg::Response::SharedPtr msg);
    void fsmPollingLoop();
    std::thread fsm_polling_thread_;

    // State
    BootState boot_state_ = BootState::DISCONNECTED;
    ArmState arm_state_ = ArmState::IDLE;

    std::map<int, double> current_pos_;
    std::map<int, double> hold_pos_;
    std::map<int, double> home_pos_;
    std::map<int, double> commanded_pos_;
    std::map<int, double> target_pos_;
    std::map<int, double> start_pos_;
    std::map<int, double> distances_;

    double blend_duration_ = BLEND_DURATION;
    std::chrono::steady_clock::time_point motion_start_time_;

    double arm_kp_ = DEFAULT_ARM_KP;
    double arm_kd_ = DEFAULT_ARM_KD;
    double waist_kp_ = DEFAULT_WAIST_KP;
    double waist_kd_ = DEFAULT_WAIST_KD;

    std::atomic<bool> lowstate_received_{false};
    std::chrono::steady_clock::time_point lowstate_last_time_;
    std::atomic<bool> init_complete_{false};
    std::atomic<bool> safety_ok_{false};
    std::atomic<int> mode_machine_{0};

    // Threads
    std::thread control_thread_;
    std::thread safety_thread_;
    std::thread worker_thread_;
    std::atomic<bool> task_running_{false};
    std::atomic<bool> task_cancel_requested_{false};
    std::atomic<bool> running_{false};
    mutable std::mutex state_mutex_;

    // Velocity tracking
    std::map<int, double> prev_positions_;
    std::chrono::steady_clock::time_point prev_time_;

    // Recording - NEW v2.8
    std::atomic<bool> is_recording_{false};
    std::vector<TrajectoryPoint> trajectory_;
    std::chrono::steady_clock::time_point recording_start_time_;
    int recording_frame_counter_ = 0;

    // Methods
    void lowstateCallback(const unitree_hg::msg::LowState::SharedPtr msg);
    void commandCallback(const std_msgs::msg::String::SharedPtr msg);
    void handleMoveToCommand(const std::string& json_str);

    void sendArmCmd(const std::map<int, double>& positions,
                    const std::map<int, double>& velocities = {},
                    double weight = 1.0);
    void sendArmCmdWithKp(const std::map<int, double>& positions,
                          double kp,
                          const std::map<int, double>& velocities = {},
                          double weight = 1.0, bool bypass_safety = false);

    std::map<int, double> clampJointPositions(const std::map<int, double>& positions);
    bool validateJump(const std::map<int, double>& target);

    void controlLoop();
    void safetyLoop();

    bool checkLowstateFresh();
    bool checkPositionError();
    bool checkVelocity();
    bool checkCancelled();

    void setArmState(ArmState new_state, const std::string& message = "");
    void setBootState(BootState new_state, const std::string& message = "");
    void publishStatus();

    void initArmsWorker();
    void moveToHomeWorker();
    void enterTeachWorker();
    void exitTeachWorker();
    void playTrajectoryWorker();  // NEW v2.8
};

}  // namespace arm_controller

#endif

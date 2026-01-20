/**
 * Arm Controller - v2.10
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
 *
 * FIXES v2.9 (Safety Critical):
 * 9. ARM_DISABLED/ARM_DISABLING states - FSM exit ramps down properly
 * 10. commanded_pos_ cleared on disable - prevents arm snap on re-entry
 * 11. FSM transition handling - immediate response to FSM changes
 * 12. stop() no longer clears safety_ok_ - separate reset_safety command
 * 13. Reduced position error tolerance (0.3 -> 0.1 rad)
 * 14. trajectory_ mutex protection - fixes data race
 * 15. FSM polling reduced to 50Hz (was 5Hz)
 *
 * FIXES v2.10:
 * 16. MOVING state timeout detection - detect stuck motions
 * 17. TRANSITIONING state - prevents position error faults during teach entry/exit
 * 18. Lowstate timeout during motion - abort if comms lost
 * 19. Command age tracking - reject stale commanded_pos_
 * 20. Velocity filter - exponential moving average reduces noise
 * 21. Basic workspace check - elbow collision prevention
 * 22. Command response topic - feedback to brain server
 * 23. Watchdog timer - detects control loop hang
 * 24. Torque monitoring - detect motor fighting obstacles
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

constexpr double MAX_POSITION_ERROR = 0.1;  // v2.9: Reduced from 0.3 rad (17deg) to 0.1 rad (5.7deg)
constexpr double MAX_JOINT_JUMP = 1.5;
constexpr double MAX_JOINT_VELOCITY = 3.0;

// v2.9: FSM transition timing constants
constexpr double DISABLE_RAMP_DURATION = 0.5;  // 500ms to ramp Kp to 0 on FSM exit
constexpr double ENABLE_RAMP_DURATION = 2.0;   // 2s to ramp Kp back up on re-entry
constexpr int FSM_POLL_INTERVAL_MS = 20;       // 50Hz FSM polling (was 200ms/5Hz)

constexpr double RAMP_DURATION = 2.0;
constexpr double START_KP = 10.0;
constexpr double END_KP = 50.0;

constexpr int CONTROL_RATE_HZ = 500;

// Recording rate - 50Hz is enough for smooth playback, saves memory
constexpr int RECORDING_RATE_HZ = 50;
constexpr int RECORDING_DIVIDER = CONTROL_RATE_HZ / RECORDING_RATE_HZ;  // 10

// v2.10: New safety constants
constexpr double MOTION_TIMEOUT_SEC = 30.0;           // Max time for any motion (detect stuck)
constexpr double LOWSTATE_MOTION_TIMEOUT_MS = 50.0;   // Max lowstate age during motion
constexpr double COMMAND_AGE_MAX_MS = 500.0;          // Max age of commanded_pos_ before stale
constexpr double VELOCITY_FILTER_ALPHA = 0.3;         // EMA filter coefficient for velocity
constexpr double MAX_SUSTAINED_TORQUE = 15.0;         // Nm, detect motor fighting obstacle
constexpr double TORQUE_SUSTAINED_TIME_SEC = 0.5;     // How long high torque before fault
constexpr double WATCHDOG_TIMEOUT_MS = 100.0;         // Control loop watchdog timeout

// v2.10: Workspace limits for collision prevention (approximate)
constexpr double ELBOW_MIN_HEIGHT = 0.1;              // Min elbow height from shoulder (prevent body collision)

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

// v2.9: Added ARM_DISABLED, ARM_DISABLING, ARM_ENABLING for safe FSM transitions
// v2.10: Added TRANSITIONING for teach entry/exit (skip position error check)
enum class ArmState {
    ARM_DISABLED,    // Zero torque, no commands sent, waiting for init_arms
    ARM_DISABLING,   // Ramping Kp to 0, clearing commanded_pos_
    ARM_ENABLING,    // Ramping Kp up after init_arms
    IDLE,            // Initialized but not holding position
    HOLDING,         // Holding commanded position
    MOVING,          // Executing trajectory
    TEACH,           // Gravity compensation, low stiffness
    TRANSITIONING    // v2.10: Kp ramp in progress (skip position error check)
};

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
    bool resetSafety();  // v2.9: Explicit safety reset (replaces stop() clearing safety_ok_)
    bool disableArms();  // v2.9: Manually trigger arm disable

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
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_response_pub_;  // v2.10: Feedback to brain

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

    // Velocity tracking - v2.10: EMA filtered velocity
    std::map<int, double> prev_positions_;
    std::chrono::steady_clock::time_point prev_time_;
    std::map<int, double> filtered_velocity_;  // v2.10: EMA filtered for noise reduction

    // Recording - v2.8
    std::atomic<bool> is_recording_{false};
    std::vector<TrajectoryPoint> trajectory_;
    std::mutex trajectory_mutex_;  // v2.9: Protect trajectory_ from data race
    std::chrono::steady_clock::time_point recording_start_time_;
    int recording_frame_counter_ = 0;

    // v2.9: FSM transition handling
    std::atomic<int> last_fsm_{-1};                              // Track FSM changes
    std::chrono::steady_clock::time_point disable_start_time_;   // When disable ramp started
    double disable_start_kp_ = 0.0;                              // Kp at start of disable ramp

    // v2.10: Command age tracking - prevent stale commands
    std::chrono::steady_clock::time_point commanded_pos_timestamp_;

    // v2.10: Motion timeout detection
    std::chrono::steady_clock::time_point motion_start_timestamp_;
    std::atomic<bool> motion_active_{false};

    // v2.10: Watchdog timer
    std::atomic<std::chrono::steady_clock::time_point> last_control_loop_time_;
    std::thread watchdog_thread_;

    // v2.10: Torque monitoring
    std::map<int, double> motor_torque_;
    std::chrono::steady_clock::time_point high_torque_start_time_;
    std::atomic<bool> high_torque_detected_{false};

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
    void playTrajectoryWorker();

    // v2.9: FSM transition handling
    void handleFsmTransition(int old_fsm, int new_fsm);
    void beginArmDisable(const std::string& reason);
    void processArmDisabling();   // Called from controlLoop during ARM_DISABLING state
    void processArmEnabling();    // Called from controlLoop during ARM_ENABLING state

    // v2.10: New safety checks
    bool checkMotionTimeout();                          // Detect stuck motions
    bool checkLowstateForMotion();                      // Stricter lowstate check during motion
    bool checkCommandAge();                             // Detect stale commanded_pos_
    bool checkWorkspace(const std::map<int, double>& positions);  // Basic collision check
    bool checkTorque();                                 // Motor fighting obstacle
    void updateFilteredVelocity();                      // EMA velocity update
    void watchdogLoop();                                // Control loop monitor
    void publishCommandResponse(const std::string& cmd, bool success, const std::string& message);
};

}  // namespace arm_controller

#endif

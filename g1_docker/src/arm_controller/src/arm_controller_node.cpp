/**
 * Arm Controller Node - v2.10
 *
 * v2.7 fixes: Worker thread, getFsmId(), cancel, topic fix
 * v2.8 fixes:
 *   - Teach mode: position error check disabled
 *   - Teach mode: commanded_pos_ synced continuously
 *   - Continuous recording with timestamps (50Hz)
 *   - Play command with exact timing replay
 *
 * v2.9 fixes (SAFETY CRITICAL):
 *   - ARM_DISABLED/ARM_DISABLING states - proper FSM exit handling
 *   - commanded_pos_ cleared on disable - prevents ARM SNAP on re-entry
 *   - FSM transition handling - immediate response to FSM changes
 *   - stop() no longer clears safety_ok_ - use reset_safety instead
 *   - FSM polling reduced to 50Hz (was 5Hz)
 *   - trajectory_ mutex protection - fixes data race
 *   - JSON status format - fixes status_announcer compatibility
 *
 * v2.10 fixes:
 *   - MOVING state timeout - detect stuck motions
 *   - TRANSITIONING state - no position error fault during teach ramp
 *   - Lowstate timeout during motion - abort if comms lost
 *   - Command age tracking - reject stale commanded_pos_
 *   - Velocity EMA filter - reduces noise sensitivity
 *   - Basic workspace check - elbow collision prevention
 *   - Command response topic - feedback to brain
 *   - Watchdog timer - detect control loop hang
 *   - Torque monitoring - motor obstacle detection
 */

#include "arm_controller/arm_controller.hpp"
#include "arm_controller/common/motor_crc_hg.h"
#include <cmath>

namespace arm_controller
{

ArmController::ArmController() : Node("arm_controller")
{
    RCLCPP_INFO(this->get_logger(), "ArmController v2.10 starting...");

    // v2.9: Start in ARM_DISABLED state - requires explicit init_arms
    arm_state_ = ArmState::ARM_DISABLED;

    for (size_t i = 0; i < LEFT_ARM.size(); ++i) home_pos_[LEFT_ARM[i]] = HOME_LEFT[i];
    for (size_t i = 0; i < RIGHT_ARM.size(); ++i) home_pos_[RIGHT_ARM[i]] = HOME_RIGHT[i];
    for (size_t i = 0; i < WAIST.size(); ++i) home_pos_[WAIST[i]] = HOME_WAIST[i];

    // Subscribers
    lowstate_sub_ = this->create_subscription<unitree_hg::msg::LowState>(
        "lowstate", 10, std::bind(&ArmController::lowstateCallback, this, std::placeholders::_1));

    auto cmd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions cmd_options;
    cmd_options.callback_group = cmd_cb_group;
    command_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/arm_ctrl/command", 10,
        std::bind(&ArmController::commandCallback, this, std::placeholders::_1), cmd_options);

    api_response_sub_ = this->create_subscription<unitree_api::msg::Response>(
        "/api/sport/response", 10,
        std::bind(&ArmController::apiResponseCallback, this, std::placeholders::_1));

    // Publishers
    status_pub_ = this->create_publisher<std_msgs::msg::String>("/arm_ctrl/status", 10);
    arm_sdk_pub_ = this->create_publisher<unitree_hg::msg::LowCmd>("arm_sdk", 10);
    api_request_pub_ = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 10);
    command_response_pub_ = this->create_publisher<std_msgs::msg::String>("/arm_ctrl/command_response", 10);  // v2.10

    // v2.10: Initialize watchdog timestamp
    last_control_loop_time_ = std::chrono::steady_clock::now();

    // Start threads
    running_ = true;
    safety_thread_ = std::thread(&ArmController::safetyLoop, this);
    control_thread_ = std::thread(&ArmController::controlLoop, this);
    fsm_polling_thread_ = std::thread(&ArmController::fsmPollingLoop, this);
    watchdog_thread_ = std::thread(&ArmController::watchdogLoop, this);  // v2.10

    RCLCPP_INFO(this->get_logger(), "ArmController ready - waiting for lowstate...");
}

ArmController::~ArmController()
{
    running_ = false;
    task_cancel_requested_ = true;
    is_recording_ = false;

    if (safety_thread_.joinable()) safety_thread_.join();
    if (control_thread_.joinable()) control_thread_.join();
    if (fsm_polling_thread_.joinable()) fsm_polling_thread_.join();
    if (watchdog_thread_.joinable()) watchdog_thread_.join();  // v2.10
    if (worker_thread_.joinable()) worker_thread_.join();
}

// =============================================================================
// FSM API - same as g1_orchestrator
// =============================================================================
void ArmController::apiResponseCallback(const unitree_api::msg::Response::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(response_mutex_);
    response_cache_[msg->header.identity.id] = *msg;
    if (response_cache_.size() > 100) response_cache_.erase(response_cache_.begin());
}

int32_t ArmController::getFsmId(int& fsm_id)
{
    auto now = std::chrono::system_clock::now();
    uint64_t request_id = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    unitree_api::msg::Request req;
    req.header.identity.id = request_id;
    req.header.identity.api_id = 7001;
    api_request_pub_->publish(req);

    auto start = std::chrono::steady_clock::now();
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(response_mutex_);
            auto it = response_cache_.find(request_id);
            if (it != response_cache_.end()) {
                if (it->second.header.status.code == 0) {
                    try {
                        auto js = nlohmann::json::parse(it->second.data);
                        fsm_id = js["data"].get<int>();
                        response_cache_.erase(it);
                        return 0;
                    } catch (...) { return -3; }
                }
                return it->second.header.status.code;
            }
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > 500) return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return -2;
}

void ArmController::fsmPollingLoop()
{
    RCLCPP_INFO(this->get_logger(), "[FSM] Polling thread started (50Hz)");
    while (running_) {
        int fsm = -1;
        if (getFsmId(fsm) == 0) {
            int old_fsm = cached_fsm_.exchange(fsm);
            // v2.9: Detect FSM transitions and handle immediately
            if (old_fsm != fsm && old_fsm != -1) {
                handleFsmTransition(old_fsm, fsm);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(FSM_POLL_INTERVAL_MS));  // v2.9: 20ms instead of 200ms
    }
}

// =============================================================================
// Lowstate Callback
// =============================================================================
void ArmController::lowstateCallback(const unitree_hg::msg::LowState::SharedPtr msg)
{
    auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (lowstate_received_) {
            prev_positions_ = current_pos_;
            prev_time_ = lowstate_last_time_;
        }
        lowstate_last_time_ = now;
        mode_machine_ = msg->mode_machine;

        for (int j : LEFT_ARM) if (j < (int)msg->motor_state.size()) {
            current_pos_[j] = msg->motor_state[j].q;
            motor_torque_[j] = msg->motor_state[j].tau_est;  // v2.10: Capture torque
        }
        for (int j : RIGHT_ARM) if (j < (int)msg->motor_state.size()) {
            current_pos_[j] = msg->motor_state[j].q;
            motor_torque_[j] = msg->motor_state[j].tau_est;  // v2.10: Capture torque
        }
        for (int j : WAIST) if (j < (int)msg->motor_state.size()) {
            current_pos_[j] = msg->motor_state[j].q;
            motor_torque_[j] = msg->motor_state[j].tau_est;  // v2.10: Capture torque
        }

        for (const auto& [joint, pos] : current_pos_) {
            if (std::isnan(pos)) { safety_ok_ = false; return; }
        }
        if (!lowstate_received_) {
            lowstate_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Lowstate received - %zu joints", current_pos_.size());
        }
    }
    // v2.10: Update velocity filter outside lock
    updateFilteredVelocity();
}

// =============================================================================
// Control Loop - v2.9: FSM-aware states, safe disable/enable
// =============================================================================
void ArmController::controlLoop()
{
    auto loop_rate = std::chrono::microseconds(2000);
    while (running_) {
        auto loop_start = std::chrono::steady_clock::now();

        // v2.10: Update watchdog timestamp
        last_control_loop_time_ = loop_start;

        // v2.9: Check for FSM != 801 and trigger disable if needed
        int current_fsm = cached_fsm_.load();
        ArmState current_state;
        { std::lock_guard<std::mutex> lock(state_mutex_); current_state = arm_state_; }

        // If FSM not ready and we're not already disabled/disabling, trigger disable
        if (current_fsm != FSM::READY && current_fsm != -1 &&
            current_state != ArmState::ARM_DISABLED &&
            current_state != ArmState::ARM_DISABLING) {
            beginArmDisable("FSM changed to " + std::to_string(current_fsm));
        }

        if (!safety_ok_.load() && current_state != ArmState::ARM_DISABLED && current_state != ArmState::ARM_DISABLING) {
            std::map<int, double> pos_copy;
            { std::lock_guard<std::mutex> lock(state_mutex_); pos_copy = current_pos_; }
            sendArmCmdWithKp(pos_copy, arm_kp_, {}, 1.0, true);
            std::this_thread::sleep_for(loop_rate);
            continue;
        }

        if (!lowstate_received_.load()) {
            std::this_thread::sleep_for(loop_rate);
            continue;
        }

        std::map<int, double> cmd_pos_copy, cur_pos_copy;
        { std::lock_guard<std::mutex> lock(state_mutex_);
          current_state = arm_state_; cmd_pos_copy = commanded_pos_; cur_pos_copy = current_pos_; }

        switch (current_state) {
            case ArmState::ARM_DISABLED:
                // v2.9: Do NOTHING - no commands sent, waiting for init_arms
                break;

            case ArmState::ARM_DISABLING:
                // v2.9: Ramp Kp to 0, clear commanded_pos_, then go to ARM_DISABLED
                processArmDisabling();
                break;

            case ArmState::ARM_ENABLING:
                // v2.9: Ramp Kp from 0 to target, then go to HOLDING
                processArmEnabling();
                break;

            case ArmState::IDLE:
                break;

            case ArmState::HOLDING:
                // v2.10: Check command age before sending
                if (!cmd_pos_copy.empty() && checkCommandAge()) {
                    sendArmCmd(cmd_pos_copy);
                }
                break;

            case ArmState::MOVING:
                // v2.10: Motion handled by worker thread, but check for timeout
                if (!checkMotionTimeout()) {
                    RCLCPP_ERROR(this->get_logger(), "[SAFETY] Motion timeout - aborting");
                    task_cancel_requested_ = true;
                    safety_ok_ = false;
                }
                break;

            case ArmState::TEACH:
                // v2.8 FIX: Continuously sync commanded_pos_ to prevent position error
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    commanded_pos_ = current_pos_;
                    commanded_pos_timestamp_ = std::chrono::steady_clock::now();  // v2.10
                }
                sendArmCmdWithKp(cur_pos_copy, SLOW_COAST_KP, {}, 1.0);

                // v2.9: Recording with mutex protection
                if (is_recording_.load()) {
                    recording_frame_counter_++;
                    if (recording_frame_counter_ >= RECORDING_DIVIDER) {
                        recording_frame_counter_ = 0;
                        double timestamp = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - recording_start_time_).count();
                        TrajectoryPoint pt;
                        pt.timestamp = timestamp;
                        pt.positions = cur_pos_copy;
                        { std::lock_guard<std::mutex> lock(trajectory_mutex_); trajectory_.push_back(pt); }
                    }
                }
                break;

            case ArmState::TRANSITIONING:
                // v2.10: Ramp in progress - no position error check, sync commanded_pos_
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    commanded_pos_ = current_pos_;
                    commanded_pos_timestamp_ = std::chrono::steady_clock::now();
                }
                break;
        }

        static int status_counter = 0;
        if (++status_counter >= 50) { publishStatus(); status_counter = 0; }

        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        if (elapsed < loop_rate) std::this_thread::sleep_for(loop_rate - elapsed);
    }
}

// =============================================================================
// Send Arm Command - uses cached_fsm_ for FSM check
// =============================================================================
void ArmController::sendArmCmd(const std::map<int, double>& positions,
                               const std::map<int, double>& velocities, double weight)
{
    sendArmCmdWithKp(positions, arm_kp_, velocities, weight);
}

void ArmController::sendArmCmdWithKp(const std::map<int, double>& positions, double kp,
                                     const std::map<int, double>& velocities,
                                     double weight, bool bypass_safety)
{
    int current_fsm = cached_fsm_.load();

    if (current_fsm != FSM::READY) {
        static auto last_warn = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_warn).count() >= 1) {
            RCLCPP_WARN(this->get_logger(), "[SAFETY] FSM=%d (not READY=801) - BLOCKED", current_fsm);
            last_warn = now;
        }
        return;
    }

    if (!bypass_safety && !safety_ok_.load()) return;

    unitree_hg::msg::LowCmd cmd;
    cmd.mode_pr = 1;
    cmd.mode_machine = mode_machine_.load();

    auto clamped = clampJointPositions(positions);
    std::map<int, double> cur_pos_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_); cur_pos_copy = current_pos_; }

    for (int j : LEFT_ARM) {
        cmd.motor_cmd[j].mode = 1;
        auto it = clamped.find(j);
        cmd.motor_cmd[j].q = (it != clamped.end()) ? (float)it->second : (float)cur_pos_copy[j];
        auto vel_it = velocities.find(j);
        cmd.motor_cmd[j].dq = (vel_it != velocities.end()) ? (float)vel_it->second : 0.0f;
        if (std::abs(cmd.motor_cmd[j].dq) > MAX_JOINT_VELOCITY)
            cmd.motor_cmd[j].dq = std::copysign((float)MAX_JOINT_VELOCITY, cmd.motor_cmd[j].dq);
        cmd.motor_cmd[j].tau = 0.0f;
        cmd.motor_cmd[j].kp = (float)kp;
        cmd.motor_cmd[j].kd = (float)arm_kd_;
    }

    for (int j : RIGHT_ARM) {
        cmd.motor_cmd[j].mode = 1;
        auto it = clamped.find(j);
        cmd.motor_cmd[j].q = (it != clamped.end()) ? (float)it->second : (float)cur_pos_copy[j];
        auto vel_it = velocities.find(j);
        cmd.motor_cmd[j].dq = (vel_it != velocities.end()) ? (float)vel_it->second : 0.0f;
        if (std::abs(cmd.motor_cmd[j].dq) > MAX_JOINT_VELOCITY)
            cmd.motor_cmd[j].dq = std::copysign((float)MAX_JOINT_VELOCITY, cmd.motor_cmd[j].dq);
        cmd.motor_cmd[j].tau = 0.0f;
        cmd.motor_cmd[j].kp = (float)kp;
        cmd.motor_cmd[j].kd = (float)arm_kd_;
    }

    for (int j : WAIST) {
        cmd.motor_cmd[j].mode = 1;
        cmd.motor_cmd[j].q = 0.0f;
        cmd.motor_cmd[j].dq = 0.0f;
        cmd.motor_cmd[j].tau = 0.0f;
        cmd.motor_cmd[j].kp = (float)waist_kp_;
        cmd.motor_cmd[j].kd = (float)waist_kd_;
    }

    cmd.motor_cmd[G1_ARM_SDK_WEIGHT].q = (float)weight;
    get_crc(cmd);
    arm_sdk_pub_->publish(cmd);
}

// =============================================================================
// Command Callback - v2.8: added start_recording, stop_recording, play
// =============================================================================
void ArmController::commandCallback(const std_msgs::msg::String::SharedPtr msg)
{
    std::string cmd = msg->data;
    RCLCPP_INFO(this->get_logger(), "[CMD] Received: %s", cmd.c_str());

    // Instant commands (no worker thread needed)
    if (cmd == "cancel") { cancel(); publishCommandResponse(cmd, true, "Cancelled"); return; }
    if (cmd == "stop") { bool ok = stop(); publishCommandResponse(cmd, ok, ok ? "Stopped" : "Failed"); return; }
    if (cmd == "reset_safety") { bool ok = resetSafety(); publishCommandResponse(cmd, ok, ok ? "Safety reset" : "Reset failed"); return; }
    if (cmd == "disable_arms") { bool ok = disableArms(); publishCommandResponse(cmd, ok, ok ? "Disabling" : "Failed"); return; }
    if (cmd == "start_recording") { bool ok = startRecording(); publishCommandResponse(cmd, ok, ok ? "Recording started" : "Failed"); return; }
    if (cmd == "stop_recording") { bool ok = stopRecording(); publishCommandResponse(cmd, ok, ok ? "Recording stopped" : "Failed"); return; }

    // v2.9: Check FSM before accepting commands
    if (cached_fsm_.load() != FSM::READY) {
        std::string reason = "FSM not READY (" + std::to_string(cached_fsm_.load()) + ")";
        RCLCPP_WARN(this->get_logger(), "[CMD] %s, rejecting: %s", reason.c_str(), cmd.c_str());
        publishCommandResponse(cmd, false, reason);
        return;
    }

    // v2.9: Check if arms disabled
    ArmState current_state;
    { std::lock_guard<std::mutex> lock(state_mutex_); current_state = arm_state_; }
    if (current_state == ArmState::ARM_DISABLED && cmd != "init_arms") {
        RCLCPP_WARN(this->get_logger(), "[CMD] Arms disabled, use init_arms first. Rejecting: %s", cmd.c_str());
        publishCommandResponse(cmd, false, "Arms disabled - use init_arms");
        return;
    }

    // Long operations - check task_running and spawn worker
    if (task_running_.load()) {
        RCLCPP_WARN(this->get_logger(), "[CMD] Task running, ignoring: %s (use 'cancel' first)", cmd.c_str());
        publishCommandResponse(cmd, false, "Task already running");
        return;
    }

    if (worker_thread_.joinable()) worker_thread_.join();
    task_cancel_requested_ = false;

    // v2.10: Acknowledge command acceptance
    publishCommandResponse(cmd, true, "Accepted");

    if (cmd == "init_arms") {
        worker_thread_ = std::thread(&ArmController::initArmsWorker, this);
    } else if (cmd == "move_home") {
        worker_thread_ = std::thread(&ArmController::moveToHomeWorker, this);
    } else if (cmd == "enter_teach") {
        worker_thread_ = std::thread(&ArmController::enterTeachWorker, this);
    } else if (cmd == "exit_teach") {
        worker_thread_ = std::thread(&ArmController::exitTeachWorker, this);
    } else if (cmd == "play") {
        worker_thread_ = std::thread(&ArmController::playTrajectoryWorker, this);
    } else if (cmd.rfind("move_to:", 0) == 0) {
        std::string json = cmd.substr(8);
        worker_thread_ = std::thread([this, json]() {
            task_running_ = true;
            handleMoveToCommand(json);
            task_running_ = false;
        });
    } else if (cmd.rfind("policy:", 0) == 0) {
        std::string json = cmd.substr(7);
        worker_thread_ = std::thread([this, json]() {
            task_running_ = true;
            handleMoveToCommand(json);
            task_running_ = false;
        });
    } else if (cmd.rfind("sequence:", 0) == 0) {
        std::string json = cmd.substr(9);
        worker_thread_ = std::thread([this, json]() {
            task_running_ = true;
            executeSequence(json);
            task_running_ = false;
        });
    } else {
        RCLCPP_WARN(this->get_logger(), "[CMD] Unknown: %s", cmd.c_str());
        publishCommandResponse(cmd, false, "Unknown command");
    }
}

// =============================================================================
// Cancel - abort running task
// =============================================================================
bool ArmController::cancel()
{
    if (!task_running_.load()) {
        RCLCPP_INFO(this->get_logger(), "[CANCEL] No task running");
        return false;
    }
    RCLCPP_WARN(this->get_logger(), "[CANCEL] Cancelling current task...");
    task_cancel_requested_ = true;
    is_recording_ = false;  // v2.8: Stop recording on cancel
    return true;
}

bool ArmController::checkCancelled()
{
    if (task_cancel_requested_.load()) {
        RCLCPP_WARN(this->get_logger(), "[CANCEL] Task cancelled by user");
        return true;
    }
    return false;
}

// =============================================================================
// Stop - v2.9: NO LONGER clears safety_ok_ (use reset_safety instead)
// =============================================================================
bool ArmController::stop()
{
    RCLCPP_INFO(this->get_logger(), "[CMD] Stop - holding current position");
    is_recording_ = false;

    ArmState current_state;
    { std::lock_guard<std::mutex> lock(state_mutex_); current_state = arm_state_; }

    // v2.9: Can't stop if disabled
    if (current_state == ArmState::ARM_DISABLED || current_state == ArmState::ARM_DISABLING) {
        RCLCPP_WARN(this->get_logger(), "[STOP] Arms disabled, use init_arms first");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        commanded_pos_ = current_pos_;
    }
    // v2.9: REMOVED safety_ok_ = true - this was dangerous!
    // Safety faults must be explicitly cleared with reset_safety
    setArmState(ArmState::HOLDING, "Stop command");
    RCLCPP_INFO(this->get_logger(), "[CMD] commanded_pos synced to current (safety NOT reset - use reset_safety if needed)");
    return true;
}

// =============================================================================
// v2.9: Reset Safety - explicit safety fault clearing with checks
// =============================================================================
bool ArmController::resetSafety()
{
    RCLCPP_INFO(this->get_logger(), "[CMD] Reset safety requested");

    // Check FSM is ready
    if (cached_fsm_.load() != FSM::READY) {
        RCLCPP_ERROR(this->get_logger(), "[RESET_SAFETY] FSM not READY (%d), cannot reset", cached_fsm_.load());
        return false;
    }

    // Check lowstate is fresh
    if (!checkLowstateFresh()) {
        RCLCPP_ERROR(this->get_logger(), "[RESET_SAFETY] Lowstate stale, cannot reset");
        return false;
    }

    // Sync commanded position to current before resetting
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        commanded_pos_ = current_pos_;
    }

    safety_ok_ = true;
    RCLCPP_INFO(this->get_logger(), "[RESET_SAFETY] Safety cleared, commanded_pos synced");
    return true;
}

// =============================================================================
// v2.9: Disable Arms - manually trigger safe arm disable
// =============================================================================
bool ArmController::disableArms()
{
    ArmState current_state;
    { std::lock_guard<std::mutex> lock(state_mutex_); current_state = arm_state_; }

    if (current_state == ArmState::ARM_DISABLED) {
        RCLCPP_INFO(this->get_logger(), "[DISABLE] Already disabled");
        return true;
    }

    if (current_state == ArmState::ARM_DISABLING) {
        RCLCPP_INFO(this->get_logger(), "[DISABLE] Already disabling");
        return true;
    }

    beginArmDisable("Manual disable request");
    return true;
}

// =============================================================================
// Safety Loop - v2.8: Skip position error check in TEACH state
// =============================================================================
void ArmController::safetyLoop()
{
    auto loop_rate = std::chrono::microseconds(2000);
    while (running_) {
        auto loop_start = std::chrono::steady_clock::now();
        bool all_ok = true;

        if (!checkLowstateFresh()) all_ok = false;

        // v2.8 FIX: Get current state and skip position error check in TEACH
        ArmState current_state;
        std::map<int, double> cmd_copy;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state = arm_state_;
            cmd_copy = commanded_pos_;
        }

        // v2.9/v2.10: Skip position error check in TEACH, TRANSITIONING, and disabled states
        if (all_ok && !cmd_copy.empty() &&
            current_state != ArmState::TEACH &&
            current_state != ArmState::TRANSITIONING &&  // v2.10: BUG-008 fix
            current_state != ArmState::ARM_DISABLED &&
            current_state != ArmState::ARM_DISABLING &&
            current_state != ArmState::ARM_ENABLING) {
            if (!checkPositionError()) all_ok = false;
        }

        // v2.10: Use filtered velocity check instead of raw
        if (all_ok && lowstate_received_ && !checkVelocity()) all_ok = false;

        // v2.10: Torque monitoring - detect motor fighting obstacle
        if (all_ok && lowstate_received_ && !checkTorque()) all_ok = false;

        safety_ok_ = all_ok;

        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        if (elapsed < loop_rate) std::this_thread::sleep_for(loop_rate - elapsed);
    }
}

bool ArmController::checkLowstateFresh()
{
    if (!lowstate_received_) return false;
    std::chrono::steady_clock::time_point last_time;
    { std::lock_guard<std::mutex> lock(state_mutex_); last_time = lowstate_last_time_; }
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - last_time).count();
    return age_ms <= 500;
}

bool ArmController::checkPositionError()
{
    std::map<int, double> cmd_copy, cur_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_); cmd_copy = commanded_pos_; cur_copy = current_pos_; }
    for (const auto& [joint, cmd_pos] : cmd_copy) {
        auto it = cur_copy.find(joint);
        if (it != cur_copy.end()) {
            double error = std::abs(cmd_pos - it->second);
            if (error > MAX_POSITION_ERROR) {
                RCLCPP_WARN(this->get_logger(), "[SAFETY] Position error on joint %d: cmd=%.3f, cur=%.3f, err=%.3f",
                    joint, cmd_pos, it->second, error);
                return false;
            }
        }
    }
    return true;
}

bool ArmController::checkVelocity()
{
    // v2.10: Use filtered velocity to reduce noise sensitivity
    std::map<int, double> filtered_vel_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_); filtered_vel_copy = filtered_velocity_; }

    for (const auto& [joint, vel] : filtered_vel_copy) {
        if (std::abs(vel) > MAX_JOINT_VELOCITY) {
            RCLCPP_WARN(this->get_logger(), "[SAFETY] Filtered velocity exceeded on joint %d: %.3f rad/s",
                        joint, vel);
            return false;
        }
    }
    return true;
}

// v2.10: Update exponential moving average filtered velocity
void ArmController::updateFilteredVelocity()
{
    std::map<int, double> cur_copy, prev_copy;
    std::chrono::steady_clock::time_point prev_t;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        cur_copy = current_pos_;
        prev_copy = prev_positions_;
        prev_t = prev_time_;
    }

    if (prev_copy.empty()) return;
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - prev_t).count();
    if (dt <= 0 || dt > 0.1) return;  // Skip if dt is unreasonable

    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const auto& [joint, curr_pos] : cur_copy) {
        auto it = prev_copy.find(joint);
        if (it != prev_copy.end()) {
            double raw_vel = (curr_pos - it->second) / dt;
            auto filt_it = filtered_velocity_.find(joint);
            if (filt_it != filtered_velocity_.end()) {
                // EMA filter: new = alpha * raw + (1-alpha) * old
                filtered_velocity_[joint] = VELOCITY_FILTER_ALPHA * raw_vel +
                                            (1.0 - VELOCITY_FILTER_ALPHA) * filt_it->second;
            } else {
                filtered_velocity_[joint] = raw_vel;
            }
        }
    }
}

bool ArmController::validateJump(const std::map<int, double>& target)
{
    std::map<int, double> cur_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_); cur_copy = current_pos_; }
    for (const auto& [joint, target_pos] : target) {
        auto it = cur_copy.find(joint);
        if (it != cur_copy.end() && std::abs(target_pos - it->second) > MAX_JOINT_JUMP) {
            RCLCPP_WARN(this->get_logger(), "[SAFETY] Jump too large on joint %d", joint);
            return false;
        }
    }
    return true;
}

std::map<int, double> ArmController::clampJointPositions(const std::map<int, double>& positions)
{
    std::map<int, double> clamped;
    auto all_limits = getAllJointLimits();
    for (const auto& [joint, pos] : positions) {
        double clamped_pos = pos;
        auto it = all_limits.find(joint);
        if (it != all_limits.end()) {
            if (pos < it->second.min) clamped_pos = it->second.min;
            else if (pos > it->second.max) clamped_pos = it->second.max;
        }
        clamped[joint] = clamped_pos;
    }
    return clamped;
}

// =============================================================================
// State Management
// =============================================================================
ArmState ArmController::getState() const { std::lock_guard<std::mutex> lock(state_mutex_); return arm_state_; }
std::string ArmController::getStateName() const {
    ArmState s; { std::lock_guard<std::mutex> lock(state_mutex_); s = arm_state_; }
    switch (s) {
        case ArmState::ARM_DISABLED: return "ARM_DISABLED";
        case ArmState::ARM_DISABLING: return "ARM_DISABLING";
        case ArmState::ARM_ENABLING: return "ARM_ENABLING";
        case ArmState::IDLE: return "IDLE";
        case ArmState::HOLDING: return "HOLDING";
        case ArmState::MOVING: return "MOVING";
        case ArmState::TEACH: return "TEACH";
        case ArmState::TRANSITIONING: return "TRANSITIONING";  // v2.10
        default: return "UNKNOWN";
    }
}
bool ArmController::isReady() const { return init_complete_.load() && safety_ok_.load(); }
bool ArmController::isMoving() const { std::lock_guard<std::mutex> lock(state_mutex_); return arm_state_ == ArmState::MOVING; }
std::map<int, double> ArmController::getCurrentPositions() const { std::lock_guard<std::mutex> lock(state_mutex_); return current_pos_; }
std::map<int, double> ArmController::getCommandedPositions() const { std::lock_guard<std::mutex> lock(state_mutex_); return commanded_pos_; }

void ArmController::setArmState(ArmState new_state, const std::string& message) {
    std::string old_name = getStateName();
    { std::lock_guard<std::mutex> lock(state_mutex_); arm_state_ = new_state; }
    RCLCPP_INFO(this->get_logger(), "[ARM] %s -> %s %s", old_name.c_str(), getStateName().c_str(), message.c_str());
}
void ArmController::setBootState(BootState new_state, const std::string& message) {
    boot_state_ = new_state;
    RCLCPP_INFO(this->get_logger(), "[BOOT] %s", message.c_str());
}
void ArmController::publishStatus() {
    // v2.9: JSON format for compatibility with status_announcer.py
    std_msgs::msg::String status_msg;
    size_t traj_size;
    { std::lock_guard<std::mutex> lock(trajectory_mutex_); traj_size = trajectory_.size(); }

    nlohmann::json status_json;
    status_json["arm_state"] = getStateName();
    status_json["safety"] = safety_ok_.load() ? "OK" : "FAIL";
    status_json["lowstate"] = lowstate_received_.load() ? "YES" : "NO";
    status_json["task"] = task_running_.load() ? "RUNNING" : "IDLE";
    status_json["recording"] = is_recording_.load() ? "YES" : "NO";
    status_json["trajectory"] = traj_size;
    status_json["fsm"] = cached_fsm_.load();
    status_json["init_complete"] = init_complete_.load();

    status_msg.data = status_json.dump();
    status_pub_->publish(status_msg);
}

// =============================================================================
// Worker Functions
// =============================================================================
void ArmController::initArmsWorker() { task_running_ = true; initArms(); task_running_ = false; }
void ArmController::moveToHomeWorker() { task_running_ = true; moveToHome(); task_running_ = false; }
void ArmController::enterTeachWorker() { task_running_ = true; enterTeach(); task_running_ = false; }
void ArmController::exitTeachWorker() { task_running_ = true; exitTeach(); task_running_ = false; }
void ArmController::playTrajectoryWorker() { task_running_ = true; playTrajectory(); task_running_ = false; }

// =============================================================================
// Init Arms
// =============================================================================
bool ArmController::initArms() {
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "    ARM INIT SEQUENCE");
    RCLCPP_INFO(this->get_logger(), "========================================");

    // Reset commanded_pos to prevent stale value issues
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        commanded_pos_ = current_pos_;
    }

    if (init_complete_.load()) { RCLCPP_INFO(this->get_logger(), "[INIT] Already complete"); return true; }

    // Wait for FSM 801
    RCLCPP_INFO(this->get_logger(), "[INIT] Waiting for FSM=801...");
    auto start = std::chrono::steady_clock::now();
    while (running_ && !checkCancelled()) {
        if (cached_fsm_.load() == FSM::READY) break;
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed > 30) {
            RCLCPP_ERROR(this->get_logger(), "[INIT] Timeout waiting for FSM=801 (current=%d)", cached_fsm_.load());
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (checkCancelled()) { setArmState(ArmState::IDLE, "Cancelled"); return false; }
    RCLCPP_INFO(this->get_logger(), "[INIT] FSM=801 confirmed");

    // Wait for lowstate
    RCLCPP_INFO(this->get_logger(), "[INIT] Waiting for lowstate...");
    start = std::chrono::steady_clock::now();
    while (running_ && !checkCancelled() && !lowstate_received_.load()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed > 5) { RCLCPP_ERROR(this->get_logger(), "[INIT] No lowstate"); return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (checkCancelled()) { setArmState(ArmState::IDLE, "Cancelled"); return false; }
    RCLCPP_INFO(this->get_logger(), "[INIT] Lowstate received");

    // Capture hold position
    std::map<int, double> hold_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_);
      hold_pos_ = current_pos_; commanded_pos_ = current_pos_; hold_copy = current_pos_; }
    RCLCPP_INFO(this->get_logger(), "[INIT] Captured %zu joints", hold_copy.size());

    // Stiffness ramp
    RCLCPP_INFO(this->get_logger(), "[INIT] Stiffness ramp (%.1f -> %.1f)...", START_KP, END_KP);
    auto ramp_start = std::chrono::steady_clock::now();
    while (running_ && !checkCancelled()) {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - ramp_start).count();
        if (elapsed >= RAMP_DURATION) break;
        double ratio = elapsed / RAMP_DURATION;
        double current_kp = START_KP + (END_KP - START_KP) * ratio;
        sendArmCmdWithKp(hold_copy, current_kp, {}, 1.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (checkCancelled()) { setArmState(ArmState::IDLE, "Cancelled"); return false; }

    arm_kp_ = END_KP;
    sendArmCmdWithKp(hold_copy, arm_kp_, {}, 1.0);
    setArmState(ArmState::HOLDING, "Init complete");
    init_complete_ = true;

    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "    ARM INIT COMPLETE");
    RCLCPP_INFO(this->get_logger(), "========================================");
    return true;
}

// =============================================================================
// Move To Home
// =============================================================================
bool ArmController::moveToHome(double blend_time) {
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "    MOVE TO HOME");
    RCLCPP_INFO(this->get_logger(), "========================================");

    if (!init_complete_.load()) { RCLCPP_ERROR(this->get_logger(), "[HOME] Not initialized"); return false; }

    std::map<int, double> home_target;
    for (size_t i = 0; i < LEFT_ARM.size(); ++i) home_target[LEFT_ARM[i]] = HOME_LEFT[i];
    for (size_t i = 0; i < RIGHT_ARM.size(); ++i) home_target[RIGHT_ARM[i]] = HOME_RIGHT[i];

    return moveTo(home_target, blend_time > 0 ? blend_time : BLEND_DURATION);
}

// =============================================================================
// Move To
// =============================================================================
bool ArmController::moveTo(const std::map<int, double>& target, double blend_time) {
    double duration = std::max(blend_time, MIN_BLEND_TIME);
    RCLCPP_INFO(this->get_logger(), "[MOTION] moveTo: %zu joints, blend=%.2fs", target.size(), duration);

    if (!validateJump(target)) { RCLCPP_ERROR(this->get_logger(), "[MOTION] Jump too large"); return false; }
    if (!lowstate_received_.load()) { RCLCPP_ERROR(this->get_logger(), "[MOTION] No lowstate"); return false; }

    // v2.10: Basic workspace check
    if (!checkWorkspace(target)) {
        RCLCPP_ERROR(this->get_logger(), "[MOTION] Workspace check failed - collision risk");
        return false;
    }

    auto clamped_target = clampJointPositions(target);

    std::map<int, double> start_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_);
      blend_duration_ = duration; start_pos_ = current_pos_; start_copy = current_pos_;
      distances_.clear();
      for (const auto& [joint, tgt] : clamped_target) {
          auto it = start_pos_.find(joint);
          distances_[joint] = (it != start_pos_.end()) ? tgt - it->second : 0.0;
      }
      motion_start_time_ = std::chrono::steady_clock::now();
    }

    // v2.10: Track motion timeout
    motion_start_timestamp_ = std::chrono::steady_clock::now();
    motion_active_ = true;

    setArmState(ArmState::MOVING, "Starting motion");

    auto loop_rate = std::chrono::microseconds(2000);
    while (running_ && !checkCancelled()) {
        auto loop_start = std::chrono::steady_clock::now();

        if (!safety_ok_.load()) {
            RCLCPP_ERROR(this->get_logger(), "[MOTION] Safety fault");
            motion_active_ = false;
            setArmState(ArmState::HOLDING, "Safety abort");
            return false;
        }

        // v2.10: BUG-026 fix - stricter lowstate check during motion
        if (!checkLowstateForMotion()) {
            RCLCPP_ERROR(this->get_logger(), "[MOTION] Lowstate stale during motion - aborting");
            motion_active_ = false;
            setArmState(ArmState::HOLDING, "Lowstate timeout");
            return false;
        }

        double elapsed, T;
        std::map<int, double> start_local, dist_local;
        { std::lock_guard<std::mutex> lock(state_mutex_);
          elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - motion_start_time_).count();
          T = blend_duration_; start_local = start_pos_; dist_local = distances_; }

        if (elapsed >= T) {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                commanded_pos_ = clamped_target;
                commanded_pos_timestamp_ = std::chrono::steady_clock::now();  // v2.10
            }
            sendArmCmd(clamped_target, {}, 1.0);
            RCLCPP_INFO(this->get_logger(), "[MOTION] Complete");
            break;
        }

        double ratio = std::min(1.0, elapsed / T);
        double smooth_ratio = smoothstep(ratio);
        std::map<int, double> positions, velocities;
        for (const auto& [joint, dist] : dist_local) {
            auto it = start_local.find(joint);
            if (it == start_local.end()) continue;
            positions[joint] = it->second + dist * smooth_ratio;
            velocities[joint] = (ratio < 1.0 && T > 0) ? dist * smoothstepVelocity(ratio, T) : 0.0;
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            commanded_pos_ = positions;
            commanded_pos_timestamp_ = std::chrono::steady_clock::now();  // v2.10
        }
        sendArmCmd(positions, velocities, 1.0);

        auto elapsed_loop = std::chrono::steady_clock::now() - loop_start;
        if (elapsed_loop < loop_rate) std::this_thread::sleep_for(loop_rate - elapsed_loop);
    }

    motion_active_ = false;
    if (checkCancelled()) { setArmState(ArmState::HOLDING, "Cancelled"); return false; }
    setArmState(ArmState::HOLDING, "Motion complete");
    return true;
}

// =============================================================================
// Handle move_to command
// =============================================================================
void ArmController::handleMoveToCommand(const std::string& json_str) {
    nlohmann::json cmd;
    try { cmd = nlohmann::json::parse(json_str); }
    catch (...) { RCLCPP_ERROR(this->get_logger(), "[MOVE_TO] JSON parse error"); return; }

    std::map<int, double> target;
    std::map<int, double> cmd_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_); cmd_copy = commanded_pos_; }

    if (cmd.contains("left") && !cmd["left"].is_null()) {
        auto left_pos = cmd["left"];
        if (left_pos.is_array() && left_pos.size() == LEFT_ARM.size())
            for (size_t i = 0; i < LEFT_ARM.size(); ++i) target[LEFT_ARM[i]] = left_pos[i].get<double>();
    } else {
        for (int j : LEFT_ARM) { auto it = cmd_copy.find(j); if (it != cmd_copy.end()) target[j] = it->second; }
    }

    if (cmd.contains("right") && !cmd["right"].is_null()) {
        auto right_pos = cmd["right"];
        if (right_pos.is_array() && right_pos.size() == RIGHT_ARM.size())
            for (size_t i = 0; i < RIGHT_ARM.size(); ++i) target[RIGHT_ARM[i]] = right_pos[i].get<double>();
    } else {
        for (int j : RIGHT_ARM) { auto it = cmd_copy.find(j); if (it != cmd_copy.end()) target[j] = it->second; }
    }

    double blend_time = std::max(cmd.value("blend_time", BLEND_DURATION), MIN_BLEND_TIME);
    moveTo(target, blend_time);
}

// =============================================================================
// Execute Sequence
// =============================================================================
bool ArmController::executeSequence(const std::string& json_sequence) {
    if (!init_complete_.load()) return false;

    nlohmann::json seq;
    try { seq = nlohmann::json::parse(json_sequence); } catch (...) { return false; }

    // Support both {waypoints: [...]} and direct array [...]
    nlohmann::json waypoints;
    if (seq.is_array()) {
        waypoints = seq;
    } else if (seq.contains("waypoints") && seq["waypoints"].is_array()) {
        waypoints = seq["waypoints"];
    } else {
        RCLCPP_ERROR(this->get_logger(), "[SEQUENCE] Invalid format");
        return false;
    }

    RCLCPP_INFO(this->get_logger(), "[SEQUENCE] Executing %zu waypoints", waypoints.size());

    for (size_t i = 0; i < waypoints.size(); ++i) {
        if (!running_ || checkCancelled()) return false;
        auto& wp = waypoints[i];
        std::map<int, double> target;
        std::map<int, double> cmd_copy;
        { std::lock_guard<std::mutex> lock(state_mutex_); cmd_copy = commanded_pos_; }

        if (wp.contains("left") && !wp["left"].is_null()) {
            auto left_pos = wp["left"];
            if (left_pos.is_array() && left_pos.size() == LEFT_ARM.size())
                for (size_t j = 0; j < LEFT_ARM.size(); ++j) target[LEFT_ARM[j]] = left_pos[j].get<double>();
        } else { for (int j : LEFT_ARM) { auto it = cmd_copy.find(j); if (it != cmd_copy.end()) target[j] = it->second; } }

        if (wp.contains("right") && !wp["right"].is_null()) {
            auto right_pos = wp["right"];
            if (right_pos.is_array() && right_pos.size() == RIGHT_ARM.size())
                for (size_t j = 0; j < RIGHT_ARM.size(); ++j) target[RIGHT_ARM[j]] = right_pos[j].get<double>();
        } else { for (int j : RIGHT_ARM) { auto it = cmd_copy.find(j); if (it != cmd_copy.end()) target[j] = it->second; } }

        double blend_time = std::max(wp.value("blend_time", MIN_BLEND_TIME), MIN_BLEND_TIME);
        RCLCPP_INFO(this->get_logger(), "[SEQUENCE] WP %zu/%zu, blend=%.2fs", i+1, waypoints.size(), blend_time);
        if (!moveTo(target, blend_time)) return false;
    }
    return true;
}

// =============================================================================
// Teach Mode - v2.8: Fixed position error issue
// =============================================================================
bool ArmController::enterTeach() {
    if (!init_complete_.load()) return false;
    ArmState current_state;
    { std::lock_guard<std::mutex> lock(state_mutex_); current_state = arm_state_; }
    if (current_state == ArmState::TEACH) return true;

    // v2.8 FIX: Sync commanded_pos FIRST to prevent position error
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        commanded_pos_ = current_pos_;
        commanded_pos_timestamp_ = std::chrono::steady_clock::now();
    }

    { std::lock_guard<std::mutex> lock(trajectory_mutex_); trajectory_.clear(); }  // v2.9: mutex

    // v2.10: BUG-008 fix - use TRANSITIONING state to skip position error check
    setArmState(ArmState::TRANSITIONING, "Transitioning to teach");

    double start_kp = arm_kp_, end_kp = SLOW_COAST_KP;
    auto ramp_start = std::chrono::steady_clock::now();
    while (running_ && !checkCancelled()) {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - ramp_start).count();
        if (elapsed >= 1.0) break;
        std::map<int, double> hold_pos;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            hold_pos = current_pos_;
            commanded_pos_ = current_pos_;  // v2.8: Keep syncing during ramp
            commanded_pos_timestamp_ = std::chrono::steady_clock::now();
        }
        sendArmCmdWithKp(hold_pos, start_kp + (end_kp - start_kp) * elapsed, {}, 1.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (checkCancelled()) { setArmState(ArmState::IDLE, "Cancelled"); return false; }

    setArmState(ArmState::TEACH, "Gravity compensation active");
    RCLCPP_INFO(this->get_logger(), "TEACH MODE ACTIVE - move arms by hand, use start_recording/stop_recording");
    return true;
}

bool ArmController::exitTeach() {
    ArmState current_state;
    { std::lock_guard<std::mutex> lock(state_mutex_); current_state = arm_state_; }
    if (current_state != ArmState::TEACH) return false;

    is_recording_ = false;  // Stop any ongoing recording

    std::map<int, double> hold_pos;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        hold_pos = current_pos_;
        commanded_pos_ = current_pos_;
        commanded_pos_timestamp_ = std::chrono::steady_clock::now();
    }

    // v2.10: BUG-008 fix - use TRANSITIONING state to skip position error check
    setArmState(ArmState::TRANSITIONING, "Transitioning from teach");

    auto ramp_start = std::chrono::steady_clock::now();
    while (running_ && !checkCancelled()) {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - ramp_start).count();
        if (elapsed >= 1.0) break;
        sendArmCmdWithKp(hold_pos, SLOW_COAST_KP + (END_KP - SLOW_COAST_KP) * elapsed, {}, 1.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (checkCancelled()) { setArmState(ArmState::IDLE, "Cancelled"); return false; }

    arm_kp_ = END_KP;
    setArmState(ArmState::HOLDING, "Exited teach mode");
    size_t traj_size;
    double duration;
    { std::lock_guard<std::mutex> lock(trajectory_mutex_);
      traj_size = trajectory_.size();
      duration = trajectory_.empty() ? 0.0 : trajectory_.back().timestamp;
    }
    RCLCPP_INFO(this->get_logger(), "TEACH MODE EXITED - %zu points recorded (%.1fs duration)",
                traj_size, duration);
    return true;
}

// =============================================================================
// Recording - v2.8: Continuous recording with timestamps
// =============================================================================
bool ArmController::startRecording()
{
    ArmState current_state;
    { std::lock_guard<std::mutex> lock(state_mutex_); current_state = arm_state_; }

    if (current_state != ArmState::TEACH) {
        RCLCPP_WARN(this->get_logger(), "[RECORD] Must be in TEACH mode to record");
        return false;
    }

    if (is_recording_.load()) {
        RCLCPP_WARN(this->get_logger(), "[RECORD] Already recording");
        return false;
    }

    { std::lock_guard<std::mutex> lock(trajectory_mutex_); trajectory_.clear(); }  // v2.9: mutex
    recording_frame_counter_ = 0;
    recording_start_time_ = std::chrono::steady_clock::now();
    is_recording_ = true;

    RCLCPP_INFO(this->get_logger(), "[RECORD] Started continuous recording at %dHz", RECORDING_RATE_HZ);
    return true;
}

bool ArmController::stopRecording()
{
    if (!is_recording_.load()) {
        RCLCPP_WARN(this->get_logger(), "[RECORD] Not recording");
        return false;
    }

    is_recording_ = false;

    size_t traj_size;
    double duration;
    { std::lock_guard<std::mutex> lock(trajectory_mutex_);
      traj_size = trajectory_.size();
      duration = trajectory_.empty() ? 0.0 : trajectory_.back().timestamp;
    }
    RCLCPP_INFO(this->get_logger(), "[RECORD] Stopped - %zu points captured (%.2fs duration)",
                traj_size, duration);
    return true;
}

// =============================================================================
// Play Trajectory - v2.8: Exact timing replay
// =============================================================================
bool ArmController::playTrajectory()
{
    if (!init_complete_.load()) {
        RCLCPP_ERROR(this->get_logger(), "[PLAY] Not initialized");
        return false;
    }

    // v2.9: Copy trajectory under lock to prevent data race
    std::vector<TrajectoryPoint> traj_copy;
    { std::lock_guard<std::mutex> lock(trajectory_mutex_); traj_copy = trajectory_; }

    if (traj_copy.empty()) {
        RCLCPP_WARN(this->get_logger(), "[PLAY] No trajectory recorded");
        return false;
    }

    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "    PLAYING TRAJECTORY");
    RCLCPP_INFO(this->get_logger(), "    Points: %zu, Duration: %.2fs", traj_copy.size(),
                traj_copy.back().timestamp);
    RCLCPP_INFO(this->get_logger(), "========================================");

    // First, move to start position smoothly
    RCLCPP_INFO(this->get_logger(), "[PLAY] Moving to start position...");
    if (!moveTo(traj_copy[0].positions, 2.0)) {
        RCLCPP_ERROR(this->get_logger(), "[PLAY] Failed to reach start position");
        return false;
    }

    setArmState(ArmState::MOVING, "Playing trajectory");

    auto play_start = std::chrono::steady_clock::now();
    size_t point_idx = 0;

    auto loop_rate = std::chrono::microseconds(2000);  // 500Hz control loop
    while (running_ && !checkCancelled() && point_idx < traj_copy.size()) {
        auto loop_start = std::chrono::steady_clock::now();

        if (!safety_ok_.load()) {
            RCLCPP_ERROR(this->get_logger(), "[PLAY] Safety fault");
            setArmState(ArmState::HOLDING, "Safety abort");
            return false;
        }

        // v2.9: Check FSM during playback
        if (cached_fsm_.load() != FSM::READY) {
            RCLCPP_ERROR(this->get_logger(), "[PLAY] FSM changed during playback");
            setArmState(ArmState::HOLDING, "FSM abort");
            return false;
        }

        // Current elapsed time since play started
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - play_start).count();

        // Find the trajectory points to interpolate between
        while (point_idx < traj_copy.size() - 1 &&
               traj_copy[point_idx + 1].timestamp <= elapsed) {
            point_idx++;
        }

        std::map<int, double> cmd_positions;

        if (point_idx >= traj_copy.size() - 1) {
            // At or past last point - use final position
            cmd_positions = traj_copy.back().positions;
        } else {
            // Interpolate between point_idx and point_idx+1
            const auto& p0 = traj_copy[point_idx];
            const auto& p1 = traj_copy[point_idx + 1];

            double dt = p1.timestamp - p0.timestamp;
            double t_local = elapsed - p0.timestamp;
            double ratio = (dt > 0) ? std::min(1.0, t_local / dt) : 1.0;

            // Linear interpolation between points
            for (const auto& [joint, pos0] : p0.positions) {
                auto it = p1.positions.find(joint);
                if (it != p1.positions.end()) {
                    cmd_positions[joint] = pos0 + (it->second - pos0) * ratio;
                } else {
                    cmd_positions[joint] = pos0;
                }
            }
        }

        // Update commanded_pos and send command
        { std::lock_guard<std::mutex> lock(state_mutex_); commanded_pos_ = cmd_positions; }
        sendArmCmd(cmd_positions, {}, 1.0);

        // Check if playback complete
        if (elapsed >= traj_copy.back().timestamp) {
            break;
        }

        auto elapsed_loop = std::chrono::steady_clock::now() - loop_start;
        if (elapsed_loop < loop_rate) std::this_thread::sleep_for(loop_rate - elapsed_loop);
    }

    if (checkCancelled()) {
        setArmState(ArmState::HOLDING, "Cancelled");
        return false;
    }

    setArmState(ArmState::HOLDING, "Playback complete");
    RCLCPP_INFO(this->get_logger(), "[PLAY] Trajectory playback complete");
    return true;
}

// =============================================================================
// v2.9: FSM Transition Handling - CRITICAL for arm safety
// =============================================================================
void ArmController::handleFsmTransition(int old_fsm, int new_fsm)
{
    RCLCPP_WARN(this->get_logger(), "[FSM] Transition detected: %d -> %d", old_fsm, new_fsm);

    if (old_fsm == FSM::READY && new_fsm != FSM::READY) {
        // Lost arm control - begin immediate disable
        RCLCPP_ERROR(this->get_logger(), "[FSM] Lost arm control (801 -> %d) - DISABLING ARMS", new_fsm);
        beginArmDisable("FSM exited READY state");
    } else if (old_fsm != FSM::READY && new_fsm == FSM::READY) {
        // FSM returned to ready - but DON'T auto-enable, require explicit init_arms
        RCLCPP_INFO(this->get_logger(), "[FSM] Arm control available again (FSM=801). Use init_arms to re-enable.");
    }
}

void ArmController::beginArmDisable(const std::string& reason)
{
    ArmState current_state;
    { std::lock_guard<std::mutex> lock(state_mutex_); current_state = arm_state_; }

    if (current_state == ArmState::ARM_DISABLED || current_state == ArmState::ARM_DISABLING) {
        return;  // Already disabled or disabling
    }

    RCLCPP_WARN(this->get_logger(), "[DISABLE] Beginning arm disable: %s", reason.c_str());

    // Cancel any running task
    task_cancel_requested_ = true;
    is_recording_ = false;

    // Store current Kp for ramp-down
    disable_start_kp_ = arm_kp_;
    disable_start_time_ = std::chrono::steady_clock::now();

    setArmState(ArmState::ARM_DISABLING, reason);
}

void ArmController::processArmDisabling()
{
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - disable_start_time_).count();

    std::map<int, double> cur_pos_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_); cur_pos_copy = current_pos_; }

    if (elapsed >= DISABLE_RAMP_DURATION) {
        // Ramp complete - finalize disable
        arm_kp_ = 0.0;

        // CRITICAL: Clear commanded_pos_ to prevent snap on re-entry
        { std::lock_guard<std::mutex> lock(state_mutex_); commanded_pos_.clear(); }

        // Send final zero-weight command
        sendArmCmdWithKp(cur_pos_copy, 0.0, {}, 0.0, true);  // weight=0 releases arm control

        init_complete_ = false;  // Require init_arms again
        setArmState(ArmState::ARM_DISABLED, "Disable complete - commanded_pos cleared");

        RCLCPP_INFO(this->get_logger(), "[DISABLE] Arms disabled. Use init_arms to re-enable.");
    } else {
        // Ramp in progress - decrease Kp and weight
        double ratio = elapsed / DISABLE_RAMP_DURATION;
        double current_kp = disable_start_kp_ * (1.0 - ratio);
        double current_weight = 1.0 - ratio;

        sendArmCmdWithKp(cur_pos_copy, current_kp, {}, current_weight, true);
    }
}

void ArmController::processArmEnabling()
{
    // This is called during init_arms stiffness ramp
    // The actual enabling logic is in initArms() - this is just a placeholder
    // for future use if we want to handle enabling in controlLoop instead
}

// =============================================================================
// v2.10: New safety checks
// =============================================================================

// BUG-005 fix: Detect stuck motions
bool ArmController::checkMotionTimeout()
{
    if (!motion_active_.load()) return true;  // Not in motion

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - motion_start_timestamp_).count();

    if (elapsed > MOTION_TIMEOUT_SEC) {
        RCLCPP_ERROR(this->get_logger(), "[SAFETY] Motion timeout after %.1f seconds", elapsed);
        return false;
    }
    return true;
}

// BUG-026 fix: Stricter lowstate check during motion
bool ArmController::checkLowstateForMotion()
{
    if (!lowstate_received_.load()) return false;

    std::chrono::steady_clock::time_point last_time;
    { std::lock_guard<std::mutex> lock(state_mutex_); last_time = lowstate_last_time_; }

    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - last_time).count();

    return age_ms <= LOWSTATE_MOTION_TIMEOUT_MS;
}

// BUG-009 fix: Check commanded_pos_ age
bool ArmController::checkCommandAge()
{
    std::chrono::steady_clock::time_point cmd_time;
    { std::lock_guard<std::mutex> lock(state_mutex_); cmd_time = commanded_pos_timestamp_; }

    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - cmd_time).count();

    if (age_ms > COMMAND_AGE_MAX_MS) {
        static auto last_warn = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_warn).count() >= 1) {
            RCLCPP_WARN(this->get_logger(), "[SAFETY] commanded_pos_ stale (age=%.0fms)", (double)age_ms);
            last_warn = now;
        }
        return false;
    }
    return true;
}

// BUG-012 fix: Basic workspace/collision check
bool ArmController::checkWorkspace(const std::map<int, double>& positions)
{
    // Approximate elbow position check based on shoulder and elbow angles
    // This is a simplified check - production should use proper kinematics

    // Left arm: joints 15,16,17 (shoulder), 18 (elbow)
    // Right arm: joints 22,23,24 (shoulder), 25 (elbow)

    auto check_elbow = [&](int shoulder_pitch, int shoulder_roll, int elbow, const char* arm_name) {
        auto sp_it = positions.find(shoulder_pitch);
        auto sr_it = positions.find(shoulder_roll);
        auto el_it = positions.find(elbow);

        if (sp_it == positions.end() || sr_it == positions.end() || el_it == positions.end())
            return true;  // Can't check, allow

        double sp = sp_it->second;  // shoulder pitch (for future forward kinematics)
        double sr = sr_it->second;  // shoulder roll
        double el = el_it->second;  // elbow
        (void)sp;  // Reserved for future use in forward kinematics calculation

        // Upper arm length ~0.25m, forearm ~0.25m (approximate G1 dimensions)
        // Elbow y-position relative to shoulder (simplified)
        // When shoulder_roll < -0.5 and elbow > 1.5, elbow may hit body

        // Check for left arm hitting body (shoulder roll too negative + elbow bent inward)
        if (strcmp(arm_name, "left") == 0) {
            if (sr < -0.8 && el > 1.2) {
                RCLCPP_WARN(this->get_logger(), "[WORKSPACE] Left elbow collision risk: roll=%.2f, elbow=%.2f", sr, el);
                return false;
            }
        } else {
            // Right arm - mirrored
            if (sr > 0.8 && el > 1.2) {
                RCLCPP_WARN(this->get_logger(), "[WORKSPACE] Right elbow collision risk: roll=%.2f, elbow=%.2f", sr, el);
                return false;
            }
        }
        return true;
    };

    if (!check_elbow(15, 16, 18, "left")) return false;
    if (!check_elbow(22, 23, 25, "right")) return false;

    return true;
}

// BUG-014 fix: Torque monitoring
bool ArmController::checkTorque()
{
    std::map<int, double> torque_copy;
    { std::lock_guard<std::mutex> lock(state_mutex_); torque_copy = motor_torque_; }

    bool any_high = false;
    for (const auto& [joint, torque] : torque_copy) {
        if (std::abs(torque) > MAX_SUSTAINED_TORQUE) {
            any_high = true;
            break;
        }
    }

    if (any_high) {
        if (!high_torque_detected_.load()) {
            high_torque_start_time_ = std::chrono::steady_clock::now();
            high_torque_detected_ = true;
        } else {
            double sustained_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - high_torque_start_time_).count();
            if (sustained_time > TORQUE_SUSTAINED_TIME_SEC) {
                RCLCPP_WARN(this->get_logger(), "[SAFETY] Sustained high torque detected - motor fighting obstacle");
                return false;
            }
        }
    } else {
        high_torque_detected_ = false;
    }
    return true;
}

// BUG-015 fix: Watchdog timer
void ArmController::watchdogLoop()
{
    RCLCPP_INFO(this->get_logger(), "[WATCHDOG] Watchdog thread started");
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto last_time = last_control_loop_time_.load();
        auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - last_time).count();

        if (age_ms > WATCHDOG_TIMEOUT_MS) {
            RCLCPP_ERROR(this->get_logger(), "[WATCHDOG] Control loop timeout! Last update %.0fms ago", (double)age_ms);
            // Don't set safety_ok_ false here - just log
            // The control loop may be blocked by something
        }
    }
}

// BUG-020 fix: Command response publisher
void ArmController::publishCommandResponse(const std::string& cmd, bool success, const std::string& message)
{
    nlohmann::json response;
    response["command"] = cmd;
    response["success"] = success;
    response["message"] = message;
    response["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    response["arm_state"] = getStateName();
    response["fsm"] = cached_fsm_.load();

    std_msgs::msg::String msg;
    msg.data = response.dump();
    command_response_pub_->publish(msg);
}

void ArmController::triggerDamp(const std::string& reason) {
    RCLCPP_ERROR(this->get_logger(), "[SAFETY] DAMP: %s", reason.c_str());
    safety_ok_ = false;
}

}  // namespace arm_controller

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<arm_controller::ArmController>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}

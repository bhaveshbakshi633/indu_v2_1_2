// G1 Orchestrator Node - v3.0 (Voice Control Edition)
// Non-blocking with manual state transitions
// v2.9: Added FSM state broadcast for arm_controller coordination
// v3.0: Added emergency_stop subscriber for voice-driven control

#include <rclcpp/rclcpp.hpp>
#include <orchestrator_msgs/msg/action_command.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <unitree_api/msg/request.hpp>
#include <unitree_api/msg/response.hpp>
#include <std_msgs/msg/int32.hpp>   // v2.9: FSM state broadcast
#include <std_msgs/msg/bool.hpp>    // v3.0: Emergency stop
#include <std_msgs/msg/string.hpp>  // v3.0: Arm cancel command
#include "g1/g1_loco_client.hpp"
#include "g1/g1_arm_action_client.hpp"

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <map>
#include <atomic>

// Action IDs (API 7106) - v1 se liye
constexpr int ARM_ACTION_HIGH_FIVE = 18;
constexpr int ARM_ACTION_HUG_BUILTIN = 19;
constexpr int ARM_ACTION_HEART_BOTH_HANDS = 20;
constexpr int ARM_ACTION_WAVE_ABOVE_HEAD = 26;
constexpr int ARM_ACTION_SHAKE_HAND = 27;

// Motion params - v1 se liye
constexpr float DEMO_LINEAR_VEL = 0.6f;
constexpr float DEMO_ANGULAR_VEL = 0.5f;
constexpr double DEMO_MOVE_DURATION = 2.0;
constexpr double DEMO_TURN_DURATION = 1.5;

// Sport API IDs - hello, stretch, dance actions
constexpr int SPORT_API_HELLO = 1016;
constexpr int SPORT_API_STRETCH = 1017;
constexpr int SPORT_API_DANCE1 = 1022;
constexpr int SPORT_API_DANCE2 = 1023;

class G1Orchestrator : public rclcpp::Node
{
public:
    G1Orchestrator() : Node("g1_orchestrator")
    {
        RCLCPP_INFO(this->get_logger(), "G1 Orchestrator Node starting...");

        loco_client_ = std::make_unique<unitree::robot::g1::LocoClient>(this);
        arm_action_client_ = std::make_unique<unitree::robot::g1::G1ArmActionClient>(this);

        action_sub_ = this->create_subscription<orchestrator_msgs::msg::ActionCommand>(
            "/orchestrator/action_command", 10,
            std::bind(&G1Orchestrator::action_callback, this, std::placeholders::_1));

        lowstate_sub_ = this->create_subscription<unitree_hg::msg::LowState>(
            "lowstate", 10,
            std::bind(&G1Orchestrator::lowstate_callback, this, std::placeholders::_1));

        api_response_sub_ = this->create_subscription<unitree_api::msg::Response>(
            "/api/sport/response", 10,
            std::bind(&G1Orchestrator::api_response_callback, this, std::placeholders::_1));

        api_request_pub_ = this->create_publisher<unitree_api::msg::Request>(
            "/api/sport/request", 10);

        // v2.9: FSM state broadcast - arm_controller subscribes to this
        fsm_state_pub_ = this->create_publisher<std_msgs::msg::Int32>("/fsm_state", 10);

        // v3.0: Arm cancel command publisher
        arm_cmd_pub_ = this->create_publisher<std_msgs::msg::String>("/arm_ctrl/command", 10);

        // v3.0: EMERGENCY STOP subscriber - highest priority, reliable QoS
        // Voice fast-path se aata hai, immediately DAMP karo
        auto emergency_qos = rclcpp::QoS(1).reliable();
        emergency_stop_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/emergency_stop",
            emergency_qos,
            std::bind(&G1Orchestrator::emergency_stop_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "[ORCHESTRATOR] Supported actions: init, damp, zerotorque, standup, ready, status, getfsm, wave, shake_hand, high_five, hug, heart, forward, backward, left, right, stop, hello, stretch, dance1, dance2");
        RCLCPP_INFO(this->get_logger(), "G1 Orchestrator v3.0 (Voice Control) ready!");
    }

    ~G1Orchestrator()
    {
        shutdown_flag_ = true;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

private:
    std::unique_ptr<unitree::robot::g1::LocoClient> loco_client_;
    std::unique_ptr<unitree::robot::g1::G1ArmActionClient> arm_action_client_;
    rclcpp::Subscription<orchestrator_msgs::msg::ActionCommand>::SharedPtr action_sub_;
    rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_sub_;
    rclcpp::Subscription<unitree_api::msg::Response>::SharedPtr api_response_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_stop_sub_;  // v3.0
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr api_request_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr fsm_state_pub_;  // v2.9
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr arm_cmd_pub_;   // v3.0

    std::atomic<bool> lowstate_received_{false};
    std::atomic<bool> init_complete_{false};
    std::atomic<bool> task_running_{false};
    std::atomic<bool> shutdown_flag_{false};
    std::atomic<bool> task_cancel_requested_{false};  // v3.0: Emergency stop flag
    std::atomic<int> current_mode_machine_{0};
    
    std::mutex state_mutex_;
    unitree_hg::msg::LowState latest_lowstate_;

    std::map<uint64_t, unitree_api::msg::Response> response_cache_;
    std::mutex response_mutex_;

    std::thread worker_thread_;

    // v3.0: EMERGENCY STOP callback - immediate DAMP, no questions asked
    void emergency_stop_callback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data) {
            RCLCPP_WARN(this->get_logger(), "===========================================");
            RCLCPP_WARN(this->get_logger(), "   [EMERGENCY STOP] Voice fast-path triggered!");
            RCLCPP_WARN(this->get_logger(), "===========================================");

            // Step 1: Cancel any running task
            task_cancel_requested_.store(true);

            // Step 2: Broadcast FSM change to arm_controller (disable arms first)
            broadcastFsmChange(1, "EMERGENCY_STOP");

            // Step 3: Immediate DAMP - no delay
            loco_client_->Damp();

            // Step 4: Cancel arm motion
            auto cancel_msg = std_msgs::msg::String();
            cancel_msg.data = "cancel";
            arm_cmd_pub_->publish(cancel_msg);

            // Step 5: Stop any locomotion
            loco_client_->StopMove();

            RCLCPP_WARN(this->get_logger(), "[EMERGENCY STOP] Robot damped, motion stopped");
        }
    }

    void lowstate_callback(const unitree_hg::msg::LowState::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        lowstate_received_ = true;
        latest_lowstate_ = *msg;
        current_mode_machine_ = msg->mode_machine;
    }

    void api_response_callback(const unitree_api::msg::Response::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        response_cache_[msg->header.identity.id] = *msg;
        if (response_cache_.size() > 100) {
            response_cache_.erase(response_cache_.begin());
        }
    }

    int32_t getFsmId(int& fsm_id)
    {
        auto now = std::chrono::system_clock::now();
        uint64_t request_id = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();

        unitree_api::msg::Request req;
        req.header.identity.id = request_id;
        req.header.identity.api_id = 7001;
        api_request_pub_->publish(req);

        auto start = std::chrono::steady_clock::now();
        while (!shutdown_flag_ && !task_cancel_requested_) {  // v3.0: Check cancel flag
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
                        } catch (...) {
                            return -3;
                        }
                    }
                    return it->second.header.status.code;
                }
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > 5) return -1;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return -2;
    }

    // Verify FSM reached target state
    bool verifyFsmState(int target, const std::string& name, int timeout_sec = 10)
    {
        RCLCPP_INFO(this->get_logger(), "Verifying FSM -> %s (%d)...", name.c_str(), target);
        auto start = std::chrono::steady_clock::now();
        
        while (!shutdown_flag_ && !task_cancel_requested_) {  // v3.0: Check cancel flag
            int fsm = -1;
            if (getFsmId(fsm) == 0 && fsm == target) {
                RCLCPP_INFO(this->get_logger(), "FSM verified: %s (%d)", name.c_str(), fsm);
                return true;
            }
            
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeout_sec) {
                RCLCPP_ERROR(this->get_logger(), "FSM verification timeout! Expected %d, got %d", target, fsm);
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        return false;
    }

    void action_callback(const orchestrator_msgs::msg::ActionCommand::SharedPtr msg)
    {
        // v3.0: Clear cancel flag on new action
        task_cancel_requested_.store(false);
        
        RCLCPP_INFO(this->get_logger(), "Action: %s", msg->action_name.c_str());

        if (msg->action_name == "init") {
            if (task_running_) {
                RCLCPP_WARN(this->get_logger(), "Task already running!");
                return;
            }
            if (worker_thread_.joinable()) worker_thread_.join();
            worker_thread_ = std::thread(&G1Orchestrator::execute_init, this);
        }
        else if (msg->action_name == "damp") {
            if (task_running_) { RCLCPP_WARN(this->get_logger(), "Task running!"); return; }
            if (worker_thread_.joinable()) worker_thread_.join();
            worker_thread_ = std::thread(&G1Orchestrator::execute_damp, this);
        }
        else if (msg->action_name == "zerotorque") {
            if (task_running_) { RCLCPP_WARN(this->get_logger(), "Task running!"); return; }
            if (worker_thread_.joinable()) worker_thread_.join();
            worker_thread_ = std::thread(&G1Orchestrator::execute_zerotorque, this);
        }
        else if (msg->action_name == "standup") {
            if (task_running_) { RCLCPP_WARN(this->get_logger(), "Task running!"); return; }
            if (worker_thread_.joinable()) worker_thread_.join();
            worker_thread_ = std::thread(&G1Orchestrator::execute_standup, this);
        }
        else if (msg->action_name == "ready") {
            if (task_running_) { RCLCPP_WARN(this->get_logger(), "Task running!"); return; }
            if (worker_thread_.joinable()) worker_thread_.join();
            worker_thread_ = std::thread(&G1Orchestrator::execute_ready, this);
        }
        else if (msg->action_name == "status") {
            if (worker_thread_.joinable()) worker_thread_.join();
            worker_thread_ = std::thread([this]() {
                int fsm = -1;
                int ret = getFsmId(fsm);
                RCLCPP_INFO(this->get_logger(), "=== STATUS ===");
                RCLCPP_INFO(this->get_logger(), "FSM (API): %d (ret=%d)", fsm, ret);
                RCLCPP_INFO(this->get_logger(), "mode_machine: %d", current_mode_machine_.load());
                RCLCPP_INFO(this->get_logger(), "Init: %s, Task: %s",
                    init_complete_ ? "YES" : "NO",
                    task_running_ ? "RUNNING" : "IDLE");
            });
        }
        else if (msg->action_name == "getfsm") {
            if (worker_thread_.joinable()) worker_thread_.join();
            worker_thread_ = std::thread([this]() {
                int fsm = -1;
                int ret = getFsmId(fsm);
                RCLCPP_INFO(this->get_logger(), "GetFsmId: ret=%d, fsm=%d", ret, fsm);
            });
        }
        // setfsm action - set FSM state directly
        else if (msg->action_name == "setfsm") {
            if (msg->parameters.size() > 0) {
                int fsm_id = std::stoi(msg->parameters[0]);
                RCLCPP_INFO(this->get_logger(), ">>> SETFSM to %d", fsm_id);
                loco_client_->SetFsmId(fsm_id);
            } else {
                RCLCPP_WARN(this->get_logger(), "setfsm requires FSM ID parameter");
            }
        }
        // Gesture handlers (API 7106) - built-in arm actions
        else if (msg->action_name == "wave") {
            RCLCPP_INFO(this->get_logger(), ">>> WAVE (Action ID %d)", ARM_ACTION_WAVE_ABOVE_HEAD);
            arm_action_client_->ExecuteAction(ARM_ACTION_WAVE_ABOVE_HEAD);
        }
        else if (msg->action_name == "shake_hand") {
            RCLCPP_INFO(this->get_logger(), ">>> SHAKE_HAND (Action ID %d)", ARM_ACTION_SHAKE_HAND);
            arm_action_client_->ExecuteAction(ARM_ACTION_SHAKE_HAND);
        }
        else if (msg->action_name == "high_five") {
            RCLCPP_INFO(this->get_logger(), ">>> HIGH_FIVE (Action ID %d)", ARM_ACTION_HIGH_FIVE);
            arm_action_client_->ExecuteAction(ARM_ACTION_HIGH_FIVE);
        }
        else if (msg->action_name == "hug") {
            RCLCPP_INFO(this->get_logger(), ">>> HUG (Action ID %d)", ARM_ACTION_HUG_BUILTIN);
            arm_action_client_->ExecuteAction(ARM_ACTION_HUG_BUILTIN);
        }
        else if (msg->action_name == "heart") {
            RCLCPP_INFO(this->get_logger(), ">>> HEART (Action ID %d)", ARM_ACTION_HEART_BOTH_HANDS);
            arm_action_client_->ExecuteAction(ARM_ACTION_HEART_BOTH_HANDS);
        }
        // Motion handlers - loco_client se velocity control
        else if (msg->action_name == "forward") {
            RCLCPP_INFO(this->get_logger(), ">>> FORWARD (%.1fs)", DEMO_MOVE_DURATION);
            loco_client_->SetVelocity(DEMO_LINEAR_VEL, 0, 0, DEMO_MOVE_DURATION);
        }
        else if (msg->action_name == "backward") {
            RCLCPP_INFO(this->get_logger(), ">>> BACKWARD (%.1fs)", DEMO_MOVE_DURATION);
            loco_client_->SetVelocity(-DEMO_LINEAR_VEL, 0, 0, DEMO_MOVE_DURATION);
        }
        else if (msg->action_name == "left") {
            RCLCPP_INFO(this->get_logger(), ">>> TURN LEFT (%.1fs)", DEMO_TURN_DURATION);
            loco_client_->SetVelocity(0, 0, DEMO_ANGULAR_VEL, DEMO_TURN_DURATION);
        }
        else if (msg->action_name == "right") {
            RCLCPP_INFO(this->get_logger(), ">>> TURN RIGHT (%.1fs)", DEMO_TURN_DURATION);
            loco_client_->SetVelocity(0, 0, -DEMO_ANGULAR_VEL, DEMO_TURN_DURATION);
        }
        else if (msg->action_name == "stop") {
            RCLCPP_INFO(this->get_logger(), ">>> STOP MOVE");
            loco_client_->StopMove();
        }
        // Sport API actions - hello, stretch, dance
        else if (msg->action_name == "hello") {
            RCLCPP_INFO(this->get_logger(), ">>> HELLO (API 1016)");
            unitree_api::msg::Request req;
            req.header.identity.api_id = SPORT_API_HELLO;
            req.parameter = "{}";
            api_request_pub_->publish(req);
        }
        else if (msg->action_name == "stretch") {
            RCLCPP_INFO(this->get_logger(), ">>> STRETCH (API 1017)");
            unitree_api::msg::Request req;
            req.header.identity.api_id = SPORT_API_STRETCH;
            req.parameter = "{}";
            api_request_pub_->publish(req);
        }
        else if (msg->action_name == "dance1") {
            RCLCPP_INFO(this->get_logger(), ">>> DANCE1 (API 1022)");
            unitree_api::msg::Request req;
            req.header.identity.api_id = SPORT_API_DANCE1;
            req.parameter = "{}";
            api_request_pub_->publish(req);
        }
        else if (msg->action_name == "dance2") {
            RCLCPP_INFO(this->get_logger(), ">>> DANCE2 (API 1023)");
            unitree_api::msg::Request req;
            req.header.identity.api_id = SPORT_API_DANCE2;
            req.parameter = "{}";
            api_request_pub_->publish(req);
        }
        else {
            RCLCPP_WARN(this->get_logger(), "Unknown: %s", msg->action_name.c_str());
        }
    }

    // v2.9: Broadcast FSM state change BEFORE sending command to robot
    void broadcastFsmChange(int target_fsm, const std::string& reason)
    {
        RCLCPP_INFO(this->get_logger(), "[FSM_BROADCAST] Transitioning to FSM=%d (%s)", target_fsm, reason.c_str());
        std_msgs::msg::Int32 msg;
        msg.data = target_fsm;
        fsm_state_pub_->publish(msg);
    }

    // Manual transitions with verification

    void execute_damp()
    {
        task_running_ = true;
        RCLCPP_INFO(this->get_logger(), ">>> DAMP (FSM 1)");

        broadcastFsmChange(1, "DAMP");
        std::this_thread::sleep_for(std::chrono::milliseconds(600));

        if (task_cancel_requested_) {
            RCLCPP_WARN(this->get_logger(), "[DAMP] Cancelled by emergency stop");
            task_running_ = false;
            return;
        }

        loco_client_->Damp();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        verifyFsmState(1, "DAMP", 5);
        task_running_ = false;
    }

    void execute_zerotorque()
    {
        task_running_ = true;
        RCLCPP_INFO(this->get_logger(), ">>> ZERO_TORQUE (FSM 0)");

        broadcastFsmChange(0, "ZERO_TORQUE");
        std::this_thread::sleep_for(std::chrono::milliseconds(600));

        if (task_cancel_requested_) {
            RCLCPP_WARN(this->get_logger(), "[ZERO_TORQUE] Cancelled by emergency stop");
            task_running_ = false;
            return;
        }

        loco_client_->ZeroTorque();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        verifyFsmState(0, "ZERO_TORQUE", 5);
        task_running_ = false;
    }

    void execute_standup()
    {
        task_running_ = true;
        RCLCPP_INFO(this->get_logger(), ">>> STANDUP (FSM 4)");

        broadcastFsmChange(4, "STANDUP");
        std::this_thread::sleep_for(std::chrono::milliseconds(600));

        if (task_cancel_requested_) {
            RCLCPP_WARN(this->get_logger(), "[STANDUP] Cancelled by emergency stop");
            task_running_ = false;
            return;
        }

        loco_client_->StandUp();
        
        // v3.0: Check cancel during long standup
        for (int i = 0; i < 10 && !task_cancel_requested_; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        if (task_cancel_requested_) {
            RCLCPP_WARN(this->get_logger(), "[STANDUP] Interrupted by emergency stop");
            task_running_ = false;
            return;
        }
        
        verifyFsmState(4, "STANDUP", 5);
        task_running_ = false;
    }

    void execute_ready()
    {
        task_running_ = true;
        RCLCPP_INFO(this->get_logger(), ">>> READY (FSM 801)");

        broadcastFsmChange(801, "READY");

        if (task_cancel_requested_) {
            RCLCPP_WARN(this->get_logger(), "[READY] Cancelled by emergency stop");
            task_running_ = false;
            return;
        }

        loco_client_->SetFsmId(801);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        verifyFsmState(801, "READY", 5);
        task_running_ = false;
    }

    void execute_init()
    {
        task_running_ = true;
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "    AUTOMATIC BOOT SEQUENCE");
        RCLCPP_INFO(this->get_logger(), "========================================");

        // Step 1: Check communication
        RCLCPP_INFO(this->get_logger(), "[BOOT] Step 1: Checking...");
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (shutdown_flag_ || task_cancel_requested_) { task_running_ = false; return; }
        
        if (!lowstate_received_) {
            RCLCPP_ERROR(this->get_logger(), "[BOOT] FAILED - no lowstate");
            task_running_ = false;
            return;
        }
        
        int fsm = -1;
        getFsmId(fsm);
        RCLCPP_INFO(this->get_logger(), "[BOOT] Current FSM: %d", fsm);

        // Step 2: DAMP
        if (shutdown_flag_ || task_cancel_requested_) { task_running_ = false; return; }
        RCLCPP_INFO(this->get_logger(), "[BOOT] Step 2: DAMP...");
        loco_client_->Damp();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!verifyFsmState(1, "DAMP", 5)) {
            RCLCPP_ERROR(this->get_logger(), "[BOOT] DAMP failed!");
            task_running_ = false;
            return;
        }

        // Step 3: StandUp
        if (shutdown_flag_ || task_cancel_requested_) { task_running_ = false; return; }
        RCLCPP_INFO(this->get_logger(), "[BOOT] Step 3: StandUp (10s)...");
        loco_client_->StandUp();
        
        // v3.0: Check cancel during standup
        for (int i = 0; i < 10 && !task_cancel_requested_; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        if (task_cancel_requested_) {
            RCLCPP_WARN(this->get_logger(), "[BOOT] Interrupted by emergency stop");
            loco_client_->Damp();
            task_running_ = false;
            return;
        }
        
        if (!verifyFsmState(4, "STANDUP", 5)) {
            RCLCPP_ERROR(this->get_logger(), "[BOOT] StandUp failed! Damping...");
            loco_client_->Damp();
            task_running_ = false;
            return;
        }

        // Step 4: Ready (801)
        if (shutdown_flag_ || task_cancel_requested_) { task_running_ = false; return; }
        RCLCPP_INFO(this->get_logger(), "[BOOT] Step 4: Ready (801)...");
        loco_client_->SetFsmId(801);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        if (!verifyFsmState(801, "READY", 5)) {
            RCLCPP_ERROR(this->get_logger(), "[BOOT] Ready failed! Damping...");
            loco_client_->Damp();
            task_running_ = false;
            return;
        }

        init_complete_ = true;
        task_running_ = false;
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "    BOOT COMPLETE");
        RCLCPP_INFO(this->get_logger(), "========================================");
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<G1Orchestrator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

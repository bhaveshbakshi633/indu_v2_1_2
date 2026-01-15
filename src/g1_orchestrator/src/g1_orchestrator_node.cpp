// G1 Orchestrator Node - v2.9
// Non-blocking with manual state transitions
// v2.9: Added FSM state broadcast for arm_controller coordination

#include <rclcpp/rclcpp.hpp>
#include <orchestrator_msgs/msg/action_command.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <unitree_api/msg/request.hpp>
#include <unitree_api/msg/response.hpp>
#include <std_msgs/msg/int32.hpp>  // v2.9: FSM state broadcast
#include "g1/g1_loco_client.hpp"

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <map>
#include <atomic>

class G1Orchestrator : public rclcpp::Node
{
public:
    G1Orchestrator() : Node("g1_orchestrator")
    {
        RCLCPP_INFO(this->get_logger(), "G1 Orchestrator Node starting...");

        loco_client_ = std::make_unique<unitree::robot::g1::LocoClient>(this);

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

        // v2.9: FSM state broadcast - arm_controller subscribes to this for immediate notification
        fsm_state_pub_ = this->create_publisher<std_msgs::msg::Int32>("/fsm_state", 10);

        RCLCPP_INFO(this->get_logger(), "G1 Orchestrator v2.9 ready!");
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
    rclcpp::Subscription<orchestrator_msgs::msg::ActionCommand>::SharedPtr action_sub_;
    rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_sub_;
    rclcpp::Subscription<unitree_api::msg::Response>::SharedPtr api_response_sub_;
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr api_request_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr fsm_state_pub_;  // v2.9: FSM broadcast

    std::atomic<bool> lowstate_received_{false};
    std::atomic<bool> init_complete_{false};
    std::atomic<bool> task_running_{false};
    std::atomic<bool> shutdown_flag_{false};
    std::atomic<int> current_mode_machine_{0};
    
    std::mutex state_mutex_;
    unitree_hg::msg::LowState latest_lowstate_;

    std::map<uint64_t, unitree_api::msg::Response> response_cache_;
    std::mutex response_mutex_;

    std::thread worker_thread_;

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
        while (!shutdown_flag_) {
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
        
        while (!shutdown_flag_) {
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
            // v2.9: Fixed - use worker thread pattern instead of detach
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
            // v2.9: Fixed - use worker thread pattern instead of detach
            if (worker_thread_.joinable()) worker_thread_.join();
            worker_thread_ = std::thread([this]() {
                int fsm = -1;
                int ret = getFsmId(fsm);
                RCLCPP_INFO(this->get_logger(), "GetFsmId: ret=%d, fsm=%d", ret, fsm);
            });
        }
        else {
            RCLCPP_WARN(this->get_logger(), "Unknown: %s", msg->action_name.c_str());
        }
    }

    // v2.9: Broadcast FSM state change BEFORE sending command to robot
    // This gives arm_controller time to disable before FSM actually changes
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

        // v2.9: Broadcast BEFORE changing FSM - gives arm_controller 500ms to disable
        broadcastFsmChange(1, "DAMP");
        std::this_thread::sleep_for(std::chrono::milliseconds(600));  // Wait for arm_controller to disable

        loco_client_->Damp();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        verifyFsmState(1, "DAMP", 5);
        task_running_ = false;
    }

    void execute_zerotorque()
    {
        task_running_ = true;
        RCLCPP_INFO(this->get_logger(), ">>> ZERO_TORQUE (FSM 0)");

        // v2.9: Broadcast BEFORE changing FSM
        broadcastFsmChange(0, "ZERO_TORQUE");
        std::this_thread::sleep_for(std::chrono::milliseconds(600));

        loco_client_->ZeroTorque();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        verifyFsmState(0, "ZERO_TORQUE", 5);
        task_running_ = false;
    }

    void execute_standup()
    {
        task_running_ = true;
        RCLCPP_INFO(this->get_logger(), ">>> STANDUP (FSM 4)");

        // v2.9: Broadcast BEFORE changing FSM
        broadcastFsmChange(4, "STANDUP");
        std::this_thread::sleep_for(std::chrono::milliseconds(600));

        loco_client_->StandUp();
        std::this_thread::sleep_for(std::chrono::seconds(10));
        verifyFsmState(4, "STANDUP", 5);
        task_running_ = false;
    }

    void execute_ready()
    {
        task_running_ = true;
        RCLCPP_INFO(this->get_logger(), ">>> READY (FSM 801)");

        // v2.9: Broadcast READY state
        broadcastFsmChange(801, "READY");

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
        if (shutdown_flag_) { task_running_ = false; return; }
        
        if (!lowstate_received_) {
            RCLCPP_ERROR(this->get_logger(), "[BOOT] FAILED - no lowstate");
            task_running_ = false;
            return;
        }
        
        int fsm = -1;
        getFsmId(fsm);
        RCLCPP_INFO(this->get_logger(), "[BOOT] Current FSM: %d", fsm);

        // Step 2: DAMP
        if (shutdown_flag_) { task_running_ = false; return; }
        RCLCPP_INFO(this->get_logger(), "[BOOT] Step 2: DAMP...");
        loco_client_->Damp();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!verifyFsmState(1, "DAMP", 5)) {
            RCLCPP_ERROR(this->get_logger(), "[BOOT] DAMP failed!");
            task_running_ = false;
            return;
        }

        // Step 3: StandUp
        if (shutdown_flag_) { task_running_ = false; return; }
        RCLCPP_INFO(this->get_logger(), "[BOOT] Step 3: StandUp (10s)...");
        loco_client_->StandUp();
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (!verifyFsmState(4, "STANDUP", 5)) {
            RCLCPP_ERROR(this->get_logger(), "[BOOT] StandUp failed! Damping...");
            loco_client_->Damp();
            task_running_ = false;
            return;
        }

        // Step 4: Ready (801)
        if (shutdown_flag_) { task_running_ = false; return; }
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

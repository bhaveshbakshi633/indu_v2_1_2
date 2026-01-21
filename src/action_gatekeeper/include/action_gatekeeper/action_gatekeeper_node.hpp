// action_gatekeeper_node.hpp
// G1 Robot Action Gatekeeper - Voice commands ke liye safety layer
// v3_voice: Voice-driven control ke liye C++ validation

#ifndef ACTION_GATEKEEPER_NODE_HPP_
#define ACTION_GATEKEEPER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <orchestrator_msgs/msg/action_command.hpp>
#include <unitree_api/msg/request.hpp>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <atomic>
#include <memory>

// nlohmann/json for parsing
#include <nlohmann/json.hpp>

namespace g1_voice
{

class ActionGatekeeperNode : public rclcpp::Node
{
public:
    // Gatekeeper validation result
    enum class GatekeeperResult {
        APPROVED,
        REJECT_MALFORMED,
        REJECT_UNKNOWN_ACTION,
        REJECT_LOW_CONFIDENCE,
        REJECT_FSM_INVALID,
        REJECT_SEMANTIC_VETO,
        REJECT_ARM_NOT_READY,
        REJECT_TASK_RUNNING,
        HOLD_AWAITING_CONFIRMATION
    };

    // Risk levels for actions
    enum class RiskLevel {
        LOW,     // gestures - immediate execution
        MEDIUM,  // locomotion - needs confirmation
        HIGH     // system commands - extra validation
    };

    // Body parts for actions
    enum class BodyPart {
        ARM,
        LEGS,
        WAIST,
        FULL,
        NONE
    };

    // Action info structure
    struct ActionInfo {
        RiskLevel risk;
        std::unordered_set<int> valid_fsm_states;
        BodyPart body_part;
    };

    ActionGatekeeperNode();
    ~ActionGatekeeperNode() = default;

private:
    // Configuration
    static constexpr float CONFIDENCE_THRESHOLD = 0.8f;

    // Action whitelist
    std::unordered_map<std::string, ActionInfo> action_whitelist_;

    // State caching
    std::atomic<int> cached_fsm_;
    std::atomic<bool> arm_ready_;
    std::atomic<bool> task_running_;

    // Timer for timed locomotion
    rclcpp::TimerBase::SharedPtr motion_timer_;

    // ROS2 Subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr action_request_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr fsm_state_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr arm_status_sub_;

    // ROS2 Publishers
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr response_pub_;
    rclcpp::Publisher<orchestrator_msgs::msg::ActionCommand>::SharedPtr action_cmd_pub_;
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr loco_request_pub_;

    // Initialization
    void initializeActionWhitelist();

    // Callback handlers
    void onActionRequest(const std_msgs::msg::String::SharedPtr msg);

    // Validation functions
    GatekeeperResult validate(
        const std::string& transcript,
        const std::string& action_name,
        float confidence,
        bool confirmed
    );

    // Semantic veto functions
    bool containsQuestionWords(const std::string& transcript);
    bool hasImperativeVerb(const std::string& transcript);

    // Action validation helpers
    bool isKnownAction(const std::string& action_name);
    bool isFsmCompatible(const std::string& action_name, int current_fsm);
    RiskLevel getActionRisk(const std::string& action_name);
    bool isArmAction(const std::string& action_name);

    // Action execution
    void executeAction(const std::string& action_name);
    void executeLocomotion(const std::string& action_name);
    void sendMoveCommand(float vx, float vy, float vyaw);
    void scheduleStopAfter(float seconds);
    void cancelScheduledStop();

    // Logging
    void logDecision(
        const std::string& transcript,
        const std::string& action_name,
        float confidence,
        GatekeeperResult result
    );

    // Utility
    std::string toLower(const std::string& str);
    std::string getResultCode(GatekeeperResult result);
    std::string getResultReason(GatekeeperResult result, const std::string& action_name);
};

}  // namespace g1_voice

#endif  // ACTION_GATEKEEPER_NODE_HPP_

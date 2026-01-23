// action_gatekeeper_node.cpp
// G1 Robot Action Gatekeeper - Voice commands ke liye safety layer
// v3_voice: Voice-driven control ke liye C++ validation

#include "action_gatekeeper/action_gatekeeper_node.hpp"
#include <chrono>
#include <algorithm>
#include <cctype>

namespace g1_voice
{

ActionGatekeeperNode::ActionGatekeeperNode()
    : Node("action_gatekeeper"),
      cached_fsm_(0),
      arm_ready_(false),
      task_running_(false)
{
    RCLCPP_INFO(get_logger(), "[GATEKEEPER] Initializing Action Gatekeeper Node...");

    // Initialize action whitelist
    initializeActionWhitelist();

    // QoS profiles
    auto reliable_qos = rclcpp::QoS(10).reliable();
    auto default_qos = rclcpp::QoS(10);

    // Voice action request subscriber - brain_v2 se aata hai
    action_request_sub_ = create_subscription<std_msgs::msg::String>(
        "/voice/action_request",
        reliable_qos,
        std::bind(&ActionGatekeeperNode::onActionRequest, this, std::placeholders::_1)
    );

    // FSM state subscriber - orchestrator se aata hai
    fsm_state_sub_ = create_subscription<std_msgs::msg::Int32>(
        "/fsm_state",
        default_qos,
        [this](const std_msgs::msg::Int32::SharedPtr msg) {
            cached_fsm_.store(msg->data);
            RCLCPP_DEBUG(get_logger(), "[GATEKEEPER] FSM updated: %d", msg->data);
        }
    );

    // Arm controller status subscriber
    arm_status_sub_ = create_subscription<std_msgs::msg::String>(
        "/arm_ctrl/status",
        default_qos,
        [this](const std_msgs::msg::String::SharedPtr msg) {
            arm_ready_ = (msg->data == "ready" || msg->data == "idle");
            RCLCPP_DEBUG(get_logger(), "[GATEKEEPER] Arm status: %s", msg->data.c_str());
        }
    );

    // Gatekeeper response publisher - brain_v2 ko jaata hai
    response_pub_ = create_publisher<std_msgs::msg::String>(
        "/voice/gatekeeper_response",
        reliable_qos
    );

    // Orchestrator command publisher - orchestrator ko jaata hai
    action_cmd_pub_ = create_publisher<orchestrator_msgs::msg::ActionCommand>(
        "/orchestrator/action_command",
        reliable_qos
    );

    // Locomotion command publisher - direct sport request
    loco_request_pub_ = create_publisher<unitree_api::msg::Request>(
        "/api/sport/request",
        reliable_qos
    );

    RCLCPP_INFO(get_logger(), "[GATEKEEPER] Action Gatekeeper v3.0 ready!");
    RCLCPP_INFO(get_logger(), "[GATEKEEPER] Listening on /voice/action_request");
}

void ActionGatekeeperNode::initializeActionWhitelist()
{
    // STRICT WHITELIST - sirf ye actions allowed hain
    // Format: {name, {risk, valid_fsm_states, body_part}}
    
    // Gestures (LOW risk) - FSM 801 (READY) required
    action_whitelist_["WAVE"]       = {RiskLevel::LOW,    {801},           BodyPart::ARM};
    action_whitelist_["SHAKE_HAND"] = {RiskLevel::LOW,    {801},           BodyPart::ARM};
    action_whitelist_["HUG"]        = {RiskLevel::LOW,    {801},           BodyPart::ARM};
    action_whitelist_["HIGH_FIVE"]  = {RiskLevel::LOW,    {801},           BodyPart::ARM};
    action_whitelist_["HEADSHAKE"]  = {RiskLevel::LOW,    {801},           BodyPart::WAIST};
    action_whitelist_["HEART"]      = {RiskLevel::LOW,    {801},           BodyPart::ARM};

    // Motion (MEDIUM risk - confirmation required)
    action_whitelist_["FORWARD"]    = {RiskLevel::MEDIUM, {500, 801},      BodyPart::LEGS};
    action_whitelist_["BACKWARD"]   = {RiskLevel::MEDIUM, {500, 801},      BodyPart::LEGS};
    action_whitelist_["LEFT"]       = {RiskLevel::MEDIUM, {500, 801},      BodyPart::LEGS};
    action_whitelist_["RIGHT"]      = {RiskLevel::MEDIUM, {500, 801},      BodyPart::LEGS};
    action_whitelist_["STOP"]       = {RiskLevel::LOW,    {0,1,2,3,4,500,801}, BodyPart::LEGS};

    // Posture (MEDIUM risk)
    action_whitelist_["STANDUP"]    = {RiskLevel::MEDIUM, {1, 4},          BodyPart::FULL};
    action_whitelist_["SIT"]        = {RiskLevel::MEDIUM, {801},           BodyPart::FULL};
    action_whitelist_["SQUAT"]      = {RiskLevel::MEDIUM, {801},           BodyPart::LEGS};
    action_whitelist_["HIGH_STAND"] = {RiskLevel::LOW,    {801},           BodyPart::LEGS};
    action_whitelist_["LOW_STAND"]  = {RiskLevel::LOW,    {801},           BodyPart::LEGS};

    // System (HIGH risk - extra validation)
    action_whitelist_["INIT"]        = {RiskLevel::HIGH,   {0,1,2,3,4,500,801}, BodyPart::FULL};
    action_whitelist_["READY"]       = {RiskLevel::MEDIUM, {4, 500},      BodyPart::FULL};
    action_whitelist_["DAMP"]       = {RiskLevel::HIGH,   {0,1,2,3,4,500,801}, BodyPart::FULL};
    action_whitelist_["ZERO_TORQUE"]= {RiskLevel::HIGH,   {0,1,2,3,4,500,801}, BodyPart::FULL};

    RCLCPP_INFO(get_logger(), "[GATEKEEPER] Loaded %zu actions in whitelist", 
                action_whitelist_.size());
}

std::string ActionGatekeeperNode::toLower(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

bool ActionGatekeeperNode::containsQuestionWords(const std::string& transcript)
{
    std::string lower = toLower(transcript);
    
    // Question words jo action block karengi
    static const std::vector<std::string> question_words = {
        "what", "who", "where", "when", "why", "how",
        "which", "whose", "whom",
        "can you", "could you", "would you", "will you",
        "is it", "are you", "do you", "does it",
        "tell me", "explain", "describe"
    };

    for (const auto& qw : question_words) {
        if (lower.find(qw) != std::string::npos) {
            // Exception: explicit action verb ke saath
            if (!hasImperativeVerb(lower)) {
                return true;  // VETO
            }
        }
    }
    return false;
}

bool ActionGatekeeperNode::hasImperativeVerb(const std::string& transcript)
{
    // Imperative verbs jo action confirm karti hain
    static const std::vector<std::string> imperative_verbs = {
        "wave", "walk", "move", "turn", "stop", "sit", "stand",
        "shake", "hug", "high five", "squat", "damp", "go"
    };

    for (const auto& verb : imperative_verbs) {
        if (transcript.find(verb) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool ActionGatekeeperNode::isKnownAction(const std::string& action_name)
{
    return action_whitelist_.find(action_name) != action_whitelist_.end();
}

bool ActionGatekeeperNode::isFsmCompatible(const std::string& action_name, int current_fsm)
{
    auto it = action_whitelist_.find(action_name);
    if (it == action_whitelist_.end()) return false;
    
    const auto& valid_states = it->second.valid_fsm_states;
    return valid_states.find(current_fsm) != valid_states.end();
}

ActionGatekeeperNode::RiskLevel ActionGatekeeperNode::getActionRisk(const std::string& action_name)
{
    auto it = action_whitelist_.find(action_name);
    if (it == action_whitelist_.end()) return RiskLevel::HIGH;
    return it->second.risk;
}

bool ActionGatekeeperNode::isArmAction(const std::string& action_name)
{
    auto it = action_whitelist_.find(action_name);
    if (it == action_whitelist_.end()) return false;
    return it->second.body_part == BodyPart::ARM;
}

ActionGatekeeperNode::GatekeeperResult ActionGatekeeperNode::validate(
    const std::string& transcript,
    const std::string& action_name,
    float confidence,
    bool confirmed)
{
    // CHECK 0: Semantic veto (question words in transcript)
    if (containsQuestionWords(transcript)) {
        RCLCPP_WARN(get_logger(), "[GATEKEEPER] SEMANTIC VETO: Question detected in '%s'", 
                    transcript.c_str());
        return GatekeeperResult::REJECT_SEMANTIC_VETO;
    }

    // CHECK 1: Action in whitelist
    if (!isKnownAction(action_name)) {
        RCLCPP_WARN(get_logger(), "[GATEKEEPER] Unknown action: %s", action_name.c_str());
        return GatekeeperResult::REJECT_UNKNOWN_ACTION;
    }

    // CHECK 2: Confidence threshold (0.8)
    if (confidence < CONFIDENCE_THRESHOLD) {
        RCLCPP_WARN(get_logger(), "[GATEKEEPER] Low confidence: %.2f < %.2f", 
                    confidence, CONFIDENCE_THRESHOLD);
        return GatekeeperResult::REJECT_LOW_CONFIDENCE;
    }

    // CHECK 3: FSM state compatible
    int current_fsm = cached_fsm_.load();
    if (!isFsmCompatible(action_name, current_fsm)) {
        RCLCPP_WARN(get_logger(), "[GATEKEEPER] FSM mismatch: action=%s, fsm=%d", 
                    action_name.c_str(), current_fsm);
        return GatekeeperResult::REJECT_FSM_INVALID;
    }

    // CHECK 4: Risk level vs confirmation
    RiskLevel risk = getActionRisk(action_name);
    if ((risk == RiskLevel::MEDIUM || risk == RiskLevel::HIGH) && !confirmed) {
        RCLCPP_INFO(get_logger(), "[GATEKEEPER] Awaiting confirmation for %s (risk=%d)", 
                    action_name.c_str(), static_cast<int>(risk));
        return GatekeeperResult::HOLD_AWAITING_CONFIRMATION;
    }

    // CHECK 5: Arm controller ready (for arm actions)
    if (isArmAction(action_name) && !arm_ready_) {
        RCLCPP_WARN(get_logger(), "[GATEKEEPER] Arm not ready for: %s", action_name.c_str());
        return GatekeeperResult::REJECT_ARM_NOT_READY;
    }

    // CHECK 6: No task running (except STOP)
    if (task_running_ && action_name != "STOP") {
        RCLCPP_WARN(get_logger(), "[GATEKEEPER] Task already running, only STOP allowed");
        return GatekeeperResult::REJECT_TASK_RUNNING;
    }

    return GatekeeperResult::APPROVED;
}

void ActionGatekeeperNode::onActionRequest(const std_msgs::msg::String::SharedPtr msg)
{
    try {
        // Parse JSON from voice bridge
        nlohmann::json request = nlohmann::json::parse(msg->data);
        
        std::string transcript = request.value("transcript", "");
        std::string action_name = request.value("action_name", "");
        float confidence = request.value("confidence", 0.0f);
        bool confirmed = request.value("confirmed", false);

        RCLCPP_INFO(get_logger(), "[GATEKEEPER] Request: action=%s, conf=%.2f, transcript='%s'",
                    action_name.c_str(), confidence, transcript.c_str());

        // Validate
        GatekeeperResult result = validate(transcript, action_name, confidence, confirmed);
        
        // Log decision
        logDecision(transcript, action_name, confidence, result);

        // Handle result
        nlohmann::json response;
        response["action_name"] = action_name;
        response["current_fsm"] = cached_fsm_.load();

        if (result == GatekeeperResult::APPROVED) {
            response["approved"] = true;
            response["rejection_code"] = "";
            response["rejection_reason"] = "";
            
            // Execute the action
            executeAction(action_name);
        } else {
            response["approved"] = false;
            response["rejection_code"] = getResultCode(result);
            response["rejection_reason"] = getResultReason(result, action_name);
        }

        // Publish response
        auto response_msg = std_msgs::msg::String();
        response_msg.data = response.dump();
        response_pub_->publish(response_msg);

    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "[GATEKEEPER] Parse error: %s", e.what());
        
        nlohmann::json error_response;
        error_response["action_name"] = "";
        error_response["approved"] = false;
        error_response["rejection_code"] = "malformed";
        error_response["rejection_reason"] = "Could not parse request";
        error_response["current_fsm"] = cached_fsm_.load();

        auto response_msg = std_msgs::msg::String();
        response_msg.data = error_response.dump();
        response_pub_->publish(response_msg);
    }
}

void ActionGatekeeperNode::executeAction(const std::string& action_name)
{
    RCLCPP_INFO(get_logger(), "[GATEKEEPER] Executing: %s", action_name.c_str());

    // Motion actions - time-based execution
    if (action_name == "FORWARD" || action_name == "BACKWARD" ||
        action_name == "LEFT" || action_name == "RIGHT" || action_name == "STOP") {
        executeLocomotion(action_name);
    }
    // Posture and system actions - send to orchestrator
    else if (action_name == "STANDUP" || action_name == "SIT" || 
             action_name == "INIT" || action_name == "READY" ||
             action_name == "DAMP" || action_name == "ZERO_TORQUE") {
        auto cmd = orchestrator_msgs::msg::ActionCommand();
        // Map action names to orchestrator commands
        if (action_name == "STANDUP") cmd.action_name = "standup";
        else if (action_name == "SIT") cmd.action_name = "sit";
        else if (action_name == "DAMP") cmd.action_name = "damp";
        else if (action_name == "ZERO_TORQUE") cmd.action_name = "zerotorque";
        else if (action_name == "INIT") cmd.action_name = "init";
        else if (action_name == "READY") cmd.action_name = "ready";
        
        action_cmd_pub_->publish(cmd);
    }
    // Gestures - will be handled by arm controller (future)
    else {
        RCLCPP_INFO(get_logger(), "[GATEKEEPER] Gesture %s - sending to arm controller", 
                    action_name.c_str());
        // TODO: Send to arm controller
    }
}

void ActionGatekeeperNode::executeLocomotion(const std::string& action_name)
{
    // HARDCODED durations - NON-NEGOTIABLE
    constexpr float FORWARD_DURATION_SEC = 2.0f;
    constexpr float BACKWARD_DURATION_SEC = 2.0f;
    constexpr float TURN_DURATION_SEC = 1.5f;

    // Conservative velocities
    constexpr float WALK_VELOCITY = 0.2f;    // m/s
    constexpr float TURN_VELOCITY = 0.3f;    // rad/s

    float vx = 0.0f, vy = 0.0f, vyaw = 0.0f;
    float duration = 0.0f;

    if (action_name == "FORWARD") {
        vx = WALK_VELOCITY;
        duration = FORWARD_DURATION_SEC;
    } else if (action_name == "BACKWARD") {
        vx = -WALK_VELOCITY;
        duration = BACKWARD_DURATION_SEC;
    } else if (action_name == "LEFT") {
        vyaw = TURN_VELOCITY;
        duration = TURN_DURATION_SEC;
    } else if (action_name == "RIGHT") {
        vyaw = -TURN_VELOCITY;
        duration = TURN_DURATION_SEC;
    } else if (action_name == "STOP") {
        // Immediate stop
        sendMoveCommand(0.0f, 0.0f, 0.0f);
        cancelScheduledStop();
        RCLCPP_INFO(get_logger(), "[GATEKEEPER] STOP executed immediately");
        return;
    }

    // Start motion
    task_running_ = true;
    RCLCPP_INFO(get_logger(), "[GATEKEEPER] Locomotion: vx=%.2f, vyaw=%.2f, duration=%.1fs",
                vx, vyaw, duration);
    
    sendMoveCommand(vx, vy, vyaw);
    scheduleStopAfter(duration);
}

void ActionGatekeeperNode::sendMoveCommand(float vx, float vy, float vyaw)
{
    // Create sport API request for Move command (API ID 7105)
    unitree_api::msg::Request req;
    req.header.identity.api_id = 7105;
    
    nlohmann::json data;
    data["vx"] = vx;
    data["vy"] = vy;
    data["vyaw"] = vyaw;
    data["continueMove"] = true;
    req.parameter = data.dump();

    loco_request_pub_->publish(req);
}

void ActionGatekeeperNode::scheduleStopAfter(float seconds)
{
    // Cancel any existing timer
    cancelScheduledStop();

    // Create new timer for auto-stop
    motion_timer_ = create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(seconds * 1000)),
        [this]() {
            RCLCPP_INFO(get_logger(), "[GATEKEEPER] Auto-stop after timed motion");
            sendMoveCommand(0.0f, 0.0f, 0.0f);
            task_running_ = false;
            
            // Cancel the timer (one-shot)
            if (motion_timer_) {
                motion_timer_->cancel();
            }
        }
    );
}

void ActionGatekeeperNode::cancelScheduledStop()
{
    if (motion_timer_) {
        motion_timer_->cancel();
        motion_timer_.reset();
    }
    task_running_ = false;
}

void ActionGatekeeperNode::logDecision(
    const std::string& transcript,
    const std::string& action_name,
    float confidence,
    GatekeeperResult result)
{
    std::string result_str = getResultCode(result);

    RCLCPP_INFO(get_logger(),
        "[GATEKEEPER] Decision: Action=%s | Confidence=%.2f | FSM=%d | Result=%s",
        action_name.c_str(),
        confidence,
        cached_fsm_.load(),
        result_str.c_str()
    );

    if (result != GatekeeperResult::APPROVED) {
        RCLCPP_WARN(get_logger(),
            "[GATEKEEPER REJECT] Transcript='%s' | Reason: %s",
            transcript.c_str(),
            getResultReason(result, action_name).c_str()
        );
    }
}

std::string ActionGatekeeperNode::getResultCode(GatekeeperResult result)
{
    switch (result) {
        case GatekeeperResult::APPROVED: return "approved";
        case GatekeeperResult::REJECT_MALFORMED: return "malformed";
        case GatekeeperResult::REJECT_UNKNOWN_ACTION: return "unknown_action";
        case GatekeeperResult::REJECT_LOW_CONFIDENCE: return "low_confidence";
        case GatekeeperResult::REJECT_FSM_INVALID: return "fsm_invalid";
        case GatekeeperResult::REJECT_SEMANTIC_VETO: return "semantic_veto";
        case GatekeeperResult::REJECT_ARM_NOT_READY: return "arm_not_ready";
        case GatekeeperResult::REJECT_TASK_RUNNING: return "task_running";
        case GatekeeperResult::HOLD_AWAITING_CONFIRMATION: return "awaiting_confirmation";
        default: return "unknown";
    }
}

std::string ActionGatekeeperNode::getResultReason(GatekeeperResult result, const std::string& action_name)
{
    switch (result) {
        case GatekeeperResult::APPROVED:
            return "Action approved";
        case GatekeeperResult::REJECT_MALFORMED:
            return "Could not parse request";
        case GatekeeperResult::REJECT_UNKNOWN_ACTION:
            return "Unknown action: " + action_name;
        case GatekeeperResult::REJECT_LOW_CONFIDENCE:
            return "Confidence too low, please speak clearly";
        case GatekeeperResult::REJECT_FSM_INVALID:
            return "Robot is not in the right state for " + action_name;
        case GatekeeperResult::REJECT_SEMANTIC_VETO:
            return "Detected question - this is a conversation, not an action";
        case GatekeeperResult::REJECT_ARM_NOT_READY:
            return "Arm controller not ready";
        case GatekeeperResult::REJECT_TASK_RUNNING:
            return "Another task is running, say STOP first";
        case GatekeeperResult::HOLD_AWAITING_CONFIRMATION:
            return "Please confirm: " + action_name + "? Say yes or no.";
        default:
            return "Unknown error";
    }
}

}  // namespace g1_voice

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<g1_voice::ActionGatekeeperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

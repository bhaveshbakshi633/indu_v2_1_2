#pragma once
/**
 * BaseClient - v2.10
 *
 * v2.10: Fixed subscription leak - now uses persistent subscription with response cache
 *        instead of creating new subscription per call (BUG-023)
 */
#include <cstdint>
#include <future>
#include <rclcpp/rclcpp.hpp>
#include <utility>
#include <map>
#include <mutex>

#include "nlohmann/json.hpp"
#include "time_tools.hpp"
#include "unitree_api/msg/request.hpp"
#include "unitree_api/msg/response.hpp"
#include "ut_errror.hpp"

class BaseClient {
  using Request = unitree_api::msg::Request;
  using Response = unitree_api::msg::Response;

  rclcpp::Node* node_;
  std::string topic_name_request_;
  std::string topic_name_response_;
  rclcpp::Publisher<Request>::SharedPtr req_puber_;

  // v2.10: Persistent subscription instead of per-call creation
  rclcpp::Subscription<Response>::SharedPtr res_suber_;
  std::map<uint64_t, Response> response_cache_;
  std::mutex response_mutex_;

 public:
  BaseClient(rclcpp::Node* node, const std::string& topic_name_request,
             std::string topic_name_response)
      : node_(node),
        topic_name_request_(topic_name_request),
        topic_name_response_(std::move(topic_name_response)),
        req_puber_(node_->create_publisher<Request>(topic_name_request,
                                                    rclcpp::QoS(1))) {
    // v2.10: Create single persistent subscription
    res_suber_ = node_->create_subscription<Response>(
        topic_name_response_, rclcpp::QoS(10),
        [this](const std::shared_ptr<const Response> data) {
          std::lock_guard<std::mutex> lock(response_mutex_);
          response_cache_[data->header.identity.id] = *data;
          // Cleanup old entries - keep max 100
          if (response_cache_.size() > 100) {
            response_cache_.erase(response_cache_.begin());
          }
        });
  }

  int32_t Call(Request req, nlohmann::json& js) {
    req.header.identity.id = unitree::common::GetSystemUptimeInNanoseconds();
    const auto identity_id = req.header.identity.id;

    req_puber_->publish(req);

    // v2.10: Poll response cache instead of using promise
    auto start = std::chrono::steady_clock::now();
    while (true) {
      {
        std::lock_guard<std::mutex> lock(response_mutex_);
        auto it = response_cache_.find(identity_id);
        if (it != response_cache_.end()) {
          Response response = it->second;
          response_cache_.erase(it);

          if (response.header.status.code != 0) {
            std::cout << "error code: " << response.header.status.code << std::endl;
            return response.header.status.code;
          }
          try {
            js = nlohmann::json::parse(response.data.data());
          } catch (nlohmann::detail::exception& e) {
          }
          return UT_ROBOT_SUCCESS;
        }
      }

      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start).count();
      if (elapsed > 5) {
        return UT_ROBOT_TASK_TIMEOUT;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return UT_ROBOT_TASK_UNKNOWN_ERROR;
  }

  int32_t Call(Request req) {
    nlohmann::json js;
    return Call(std::move(req), js);
  }
};

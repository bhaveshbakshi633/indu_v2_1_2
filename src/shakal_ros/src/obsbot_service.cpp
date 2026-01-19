#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <chrono>
#include <algorithm>

#include "shakal_ros/srv/set_fov.hpp"
#include "shakal_ros/srv/set_zoom.hpp"

#include <dev/devs.hpp>

class ObsbotServiceNode : public rclcpp::Node
{
public:
    ObsbotServiceNode()
        : Node("obsbot_service"),
          device_connected_(false)
    {
        // Register SDK callback
        Devices::get().setDevChangedCallback(
            [](std::string dev_sn, bool connected, void* param) {
                auto* self = static_cast<ObsbotServiceNode*>(param);
                if (connected) {
                    self->device_connected_ = true;
                    self->connected_sn_ = dev_sn;
                    RCLCPP_INFO(self->get_logger(), "OBSBOT device connected: %s", dev_sn.c_str());
                } else {
                    if (self->connected_sn_ == dev_sn) {
                        self->device_connected_ = false;
                        self->camera_.reset();
                    }
                    RCLCPP_WARN(self->get_logger(), "OBSBOT device disconnected: %s", dev_sn.c_str());
                }
            },
            this
        );
        Devices::get().setEnableMdnsScan(false);

        // Wait for device
        RCLCPP_INFO(get_logger(), "Scanning for OBSBOT Tiny 2...");
        int wait_count = 0;
        while (!device_connected_ && wait_count < 50) {  // 5 seconds max
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            wait_count++;
        }

        if (device_connected_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            camera_ = Devices::get().getDevBySn(connected_sn_);
        }

        // Fallback: check device list
        if (!camera_) {
            auto dev_list = Devices::get().getDevList();
            for (const auto& device : dev_list) {
                if (device->productType() == ObsbotProdTiny2) {
                    camera_ = device;
                    break;
                }
            }
        }

        if (!camera_) {
            RCLCPP_ERROR(get_logger(), "OBSBOT Tiny 2 not found!");
        } else {
            RCLCPP_INFO(get_logger(), "Found: %s (SN: %s)",
                        camera_->devName().c_str(), camera_->devSn().c_str());

            // Wake up camera
            camera_->cameraSetDevRunStatusR(Device::DevStatusRun);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // Create services
        set_fov_srv_ = create_service<shakal_ros::srv::SetFov>(
            "~/set_fov",
            std::bind(&ObsbotServiceNode::set_fov_callback, this,
                      std::placeholders::_1, std::placeholders::_2)
        );

        set_zoom_srv_ = create_service<shakal_ros::srv::SetZoom>(
            "~/set_zoom",
            std::bind(&ObsbotServiceNode::set_zoom_callback, this,
                      std::placeholders::_1, std::placeholders::_2)
        );

        RCLCPP_INFO(get_logger(), "OBSBOT service node started");
    }

    ~ObsbotServiceNode()
    {
        Devices::get().close();
    }

private:
    std::shared_ptr<Device> camera_;
    bool device_connected_;
    std::string connected_sn_;
    std::string current_fov_ = "wide";
    float current_zoom_ = 1.0f;

    rclcpp::Service<shakal_ros::srv::SetFov>::SharedPtr set_fov_srv_;
    rclcpp::Service<shakal_ros::srv::SetZoom>::SharedPtr set_zoom_srv_;

    void set_fov_callback(
        const std::shared_ptr<shakal_ros::srv::SetFov::Request> request,
        std::shared_ptr<shakal_ros::srv::SetFov::Response> response)
    {
        if (!camera_) {
            response->success = false;
            response->message = "OBSBOT camera not connected";
            response->current_fov = current_fov_;
            return;
        }

        Device::FovType target_fov;
        std::string fov_name;

        if (request->fov == "wide" || request->fov == "86") {
            target_fov = Device::FovType86;
            fov_name = "wide (86)";
        } else if (request->fov == "medium" || request->fov == "78") {
            target_fov = Device::FovType78;
            fov_name = "medium (78)";
        } else if (request->fov == "narrow" || request->fov == "65") {
            target_fov = Device::FovType65;
            fov_name = "narrow (65)";
        } else {
            response->success = false;
            response->message = "Invalid FOV. Use: wide, medium, narrow";
            response->current_fov = current_fov_;
            return;
        }

        RCLCPP_INFO(get_logger(), "Setting FOV to %s", fov_name.c_str());
        int32_t result = camera_->cameraSetFovU(target_fov);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        if (result == 0) {
            current_fov_ = request->fov;
            response->success = true;
            response->message = "FOV set to " + fov_name;
        } else {
            response->success = false;
            response->message = "FOV setting failed with code: " + std::to_string(result);
        }
        response->current_fov = current_fov_;
    }

    void set_zoom_callback(
        const std::shared_ptr<shakal_ros::srv::SetZoom::Request> request,
        std::shared_ptr<shakal_ros::srv::SetZoom::Response> response)
    {
        if (!camera_) {
            response->success = false;
            response->message = "OBSBOT camera not connected";
            response->current_zoom = current_zoom_;
            return;
        }

        if (request->zoom < 1.0f || request->zoom > 2.0f) {
            response->success = false;
            response->message = "Zoom must be between 1.0 and 2.0";
            response->current_zoom = current_zoom_;
            return;
        }

        RCLCPP_INFO(get_logger(), "Setting zoom to %.1fx", request->zoom);
        int32_t result = camera_->cameraSetZoomAbsoluteR(request->zoom);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        if (result == 0) {
            current_zoom_ = request->zoom;
            response->success = true;
            response->message = "Zoom set to " + std::to_string(request->zoom) + "x";
        } else {
            response->success = false;
            response->message = "Zoom setting failed with code: " + std::to_string(result);
        }
        response->current_zoom = current_zoom_;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObsbotServiceNode>());
    rclcpp::shutdown();
    return 0;
}

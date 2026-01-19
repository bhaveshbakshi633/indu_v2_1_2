/**
 * CRC32 implementation for G1 HG LowCmd
 * Based on unitree_sdk2py/utils/crc.py
 */

#include "arm_controller/common/motor_crc_hg.h"
#include <cstring>
#include <vector>

// CRC32 polynomial (same as Python)
constexpr uint32_t CRC32_POLYNOMIAL = 0x04c11db7;

/**
 * CRC32 core calculation
 * EXACT port from Python _crc_py function
 */
uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    uint32_t crc = 0xFFFFFFFF;
    
    for (uint32_t i = 0; i < len; i++) {
        uint32_t current = ptr[i];
        uint32_t bit = 1u << 31;
        
        for (int b = 0; b < 32; b++) {
            if (crc & 0x80000000) {
                crc = (crc << 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc = crc << 1;
            }
            
            if (current & bit) {
                crc ^= CRC32_POLYNOMIAL;
            }
            
            bit >>= 1;
        }
    }
    
    return crc;
}

/**
 * Calculate and set CRC for unitree_hg::msg::LowCmd
 */
void get_crc(unitree_hg::msg::LowCmd &msg)
{
    std::vector<uint32_t> data;
    
    // First uint32: mode_pr (1 byte) + mode_machine (1 byte) + 2 padding bytes
    uint32_t header = (static_cast<uint32_t>(msg.mode_pr)) |
                      (static_cast<uint32_t>(msg.mode_machine) << 8);
    data.push_back(header);
    
    // 35 motor commands, each packed as 7 uint32s
    for (int i = 0; i < 35; i++) {
        // mode + 3 padding bytes
        uint32_t mode_word = static_cast<uint32_t>(msg.motor_cmd[i].mode);
        data.push_back(mode_word);
        
        // q as uint32 (bit representation of float)
        uint32_t q_bits;
        std::memcpy(&q_bits, &msg.motor_cmd[i].q, sizeof(float));
        data.push_back(q_bits);
        
        // dq as uint32
        uint32_t dq_bits;
        std::memcpy(&dq_bits, &msg.motor_cmd[i].dq, sizeof(float));
        data.push_back(dq_bits);
        
        // tau as uint32
        uint32_t tau_bits;
        std::memcpy(&tau_bits, &msg.motor_cmd[i].tau, sizeof(float));
        data.push_back(tau_bits);
        
        // kp as uint32
        uint32_t kp_bits;
        std::memcpy(&kp_bits, &msg.motor_cmd[i].kp, sizeof(float));
        data.push_back(kp_bits);
        
        // kd as uint32
        uint32_t kd_bits;
        std::memcpy(&kd_bits, &msg.motor_cmd[i].kd, sizeof(float));
        data.push_back(kd_bits);
        
        // reserve
        data.push_back(msg.motor_cmd[i].reserve);
    }
    
    // 4 reserve uint32s
    for (int i = 0; i < 4; i++) {
        data.push_back(msg.reserve[i]);
    }
    
    // Calculate CRC (exclude the CRC field itself)
    msg.crc = crc32_core(data.data(), static_cast<uint32_t>(data.size()));
}

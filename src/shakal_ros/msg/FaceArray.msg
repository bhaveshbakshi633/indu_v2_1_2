# Array of detected faces in a single frame

# Header with timestamp
std_msgs/Header header

# Frame dimensions
int32 frame_width
int32 frame_height

# All detected faces
shakal_ros/Face[] faces

# Processing metadata
float32 detection_time_ms
float32 recognition_time_ms
float32 total_time_ms

# Single detected face with recognition result

# Bounding box (image coordinates)
int32 x
int32 y
int32 width
int32 height

# Detection confidence (0.0 - 1.0)
float32 detection_confidence

# Recognition result
string name
float32 similarity
bool recognized

# Facial landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
geometry_msgs/Point32[5] landmarks

# Anti-spoof result
bool is_fake
float32 spoof_score

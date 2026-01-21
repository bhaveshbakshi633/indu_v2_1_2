# VoiceActionRequest.msg
# brain_v2 se gatekeeper ko bheja jayega

string transcript           # Raw STT text (semantic veto ke liye)
string action_name          # LLM se parsed action name
float32 confidence          # LLM confidence 0.0-1.0
bool confirmed              # User ne confirm kiya (MEDIUM/HIGH risk ke liye)
int64 timestamp_ns          # Request creation time nanoseconds

# GatekeeperResponse.msg
# Gatekeeper se brain_v2 ko response

string action_name          # Requested action name
bool approved               # Gatekeeper ne approve kiya ya nahi
string rejection_code       # malformed/unknown_action/low_confidence/fsm_invalid/semantic_veto
string rejection_reason     # Human-readable explanation
int32 current_fsm           # Robot ka FSM state at decision time

# ActionCommand.msg
# Custom message for orchestrator action commands

string action_name          # e.g., "init", "damp", "home"
string[] parameters         # action-specific params like "skip_if_done=true"
int32 priority              # execution priority (higher = more urgent)

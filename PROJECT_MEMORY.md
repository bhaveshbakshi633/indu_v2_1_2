# Project: Voice-Driven Humanoid Control (v3_voice)

## Latest Checkpoint
- **Date:** 2026-01-19 16:30
- **Status:** ACTIVE
- **Current Task:** v3_voice implementation complete, pending build/test

## Checkpoints History

### 2026-01-19 16:30 - v3_voice Core Implementation Complete
- All safety components created:
  - C++ Action Gatekeeper node with semantic veto rules
  - STOP fast-path (Python) in brain_v2
  - Emergency stop subscriber in orchestrator
  - Time-based locomotion (2s forward/back, 1.5s turn)
- v2_tanay UNTOUCHED - all changes in v3_voice
- launch_version.sh created for version switching
- Known issues: None
- Pending: Build packages on G1, integration testing

### Components Created:

**G1 Robot (v3_voice):**
- `/home/unitree/deployed/v3_voice/src/gatekeeper_msgs/` - ROS2 message package
- `/home/unitree/deployed/v3_voice/src/action_gatekeeper/` - C++ validation node
- `/home/unitree/deployed/v3_voice/src/g1_orchestrator/` - Modified with emergency_stop
- `/home/unitree/deployed/launch_version.sh` - Version selector

**brain_v2 (this PC):**
- `action_registry.py` - Action definitions with metadata
- `stop_fast_path.py` - Emergency STOP detection
- `voice_bridge.py` - ROS2 communication with G1

## Safety Features Implemented
1. STOP fast-path - bypasses LLM pipeline
2. C++ whitelist validation - defense-in-depth
3. Semantic veto rules - question words block actions
4. Time-based locomotion - hardcoded durations
5. FSM state validation - actions only in valid states

## Next Steps
1. Build gatekeeper_msgs and action_gatekeeper on G1
2. Test STOP fast-path
3. Test semantic veto rules
4. Integration test with full pipeline

#!/usr/bin/env python3
"""
Quick test to verify calibration file saving works
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CALIBRATION_FILE = SCRIPT_DIR / "calibration_google.json"

print("Testing calibration file saving...")
print(f"File path: {CALIBRATION_FILE}")
print(f"Absolute path: {CALIBRATION_FILE.resolve()}")

# Test 1: Write calibration file
test_echo_scale = 0.75
print(f"\nTest 1: Writing echo_scale={test_echo_scale}")

try:
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump({"echo_scale": float(test_echo_scale)}, f, indent=2)
    print(f"✅ File written successfully")
except Exception as e:
    print(f"❌ Write failed: {e}")
    exit(1)

# Test 2: Verify file exists
print(f"\nTest 2: Checking if file exists")
if CALIBRATION_FILE.exists():
    print(f"✅ File exists at: {CALIBRATION_FILE}")
else:
    print(f"❌ File does not exist!")
    exit(1)

# Test 3: Read back the value
print(f"\nTest 3: Reading calibration file")
try:
    with open(CALIBRATION_FILE, 'r') as f:
        data = json.load(f)
        echo_scale = data.get('echo_scale', 0.9)
    print(f"✅ Read echo_scale={echo_scale}")

    if echo_scale == test_echo_scale:
        print(f"✅ Value matches what we wrote!")
    else:
        print(f"❌ Value mismatch: expected {test_echo_scale}, got {echo_scale}")
        exit(1)
except Exception as e:
    print(f"❌ Read failed: {e}")
    exit(1)

# Test 4: Show file contents
print(f"\nTest 4: File contents:")
with open(CALIBRATION_FILE, 'r') as f:
    print(f.read())

print(f"\n✅ ALL TESTS PASSED - Calibration file system works correctly!")

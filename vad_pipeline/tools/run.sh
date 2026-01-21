#!/bin/bash
# BR_AI_N Voice Assistant Launcher

cd "$(dirname "$0")"

echo "======================================"
echo "🎙️🧠 BR_AI_N Voice Assistant"
echo "======================================"
echo ""

# Check if config exists
if [ ! -f "config.json" ]; then
    echo "⚠️  config.json not found. It will be created on first run."
fi

# Run the assistant
python3 br_ai_n.py "$@"

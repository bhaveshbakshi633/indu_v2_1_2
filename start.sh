#!/bin/bash
# INDU Brain v2.1 - Quick Start Script
# ====================================

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run INSTALL.sh first."
    exit 1
fi

# Check if vectorstore exists
if [ ! -d "indu_vectorstore" ]; then
    echo "⚠️  Vectorstore not found. Creating it now..."
    python setup.py
fi

# Start the server
echo ""
echo "=========================================="
echo "Starting INDU Brain v2.1 Server"
echo "=========================================="
echo ""
echo "Server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python server.py

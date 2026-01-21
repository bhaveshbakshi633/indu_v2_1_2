#!/bin/bash
# INDU Brain v2.1 - Quick Installation Script
# ==========================================
# This script automates the installation process

set -e  # Exit on error

echo "=========================================="
echo "INDU Brain v2.1 - Quick Install"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✓ Ollama found: $(ollama --version)"
fi
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ Pip upgraded"
echo ""

# Install requirements
echo "📚 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Pull Ollama model
echo "🤖 Pulling Ollama model (llama3.1:8b)..."
ollama pull llama3.1:8b
echo "✓ Ollama model ready"
echo ""

# Run setup script
echo "⚙️  Running setup script..."
python setup.py

echo ""
echo "=========================================="
echo "🎉 Installation Complete!"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the server: python server.py"
echo "  3. Open browser: http://localhost:5000"
echo ""
echo "For more information, see README_DEPLOYMENT.md"
echo ""

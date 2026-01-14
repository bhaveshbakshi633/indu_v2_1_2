#!/usr/bin/env python3
"""
INDU Brain v2.1 Setup Script
============================

This script initializes the INDU voice assistant system:
1. Checks dependencies
2. Creates vectorstore from knowledge base
3. Validates configuration
4. Verifies Ollama connection

Run this script before starting the server for the first time.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_success(text):
    """Print success message"""
    print(f"✓ {text}")

def print_error(text):
    """Print error message"""
    print(f"✗ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ {text}")

def check_python_version():
    """Check Python version"""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_info("Checking dependencies...")

    required_packages = [
        'flask', 'flask_sock', 'langchain', 'langchain_community',
        'faiss', 'sentence_transformers', 'requests', 'sounddevice',
        'soundfile', 'edge_tts', 'speech_recognition'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)

    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
        print_info("Install with: pip install -r requirements.txt")
        return False
    else:
        print_success("All dependencies installed")
        return True

def check_ollama():
    """Check if Ollama is accessible"""
    print_info("Checking Ollama connection...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print_success(f"Ollama connected. Models: {', '.join(model_names)}")

            # Check if gemma2:2b is installed
            if any('gemma2:2b' in name for name in model_names):
                print_success("gemma2:2b model found")
            else:
                print_error("gemma2:2b model not found")
                print_info("Install with: ollama pull gemma2:2b")
                return False

            return True
        else:
            print_error(f"Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cannot connect to Ollama: {e}")
        print_info("Start Ollama with: ollama serve")
        return False

def check_files():
    """Check if required files exist"""
    print_info("Checking required files...")

    required_files = [
        'server.py',
        'indu_rag.py',
        'indu_system_prompt.txt',
        'indu_knowledge_base.txt',
        'config.json',
        'requirements.txt'
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print_error(f"Missing files: {', '.join(missing)}")
        return False
    else:
        print_success("All required files present")
        return True

def validate_config():
    """Validate config.json"""
    print_info("Validating configuration...")

    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        required_keys = ['stt_backend', 'tts_backend', 'ollama_model',
                        'ollama_base_url', 'enable_rag']

        missing = [key for key in required_keys if key not in config]

        if missing:
            print_error(f"Missing config keys: {', '.join(missing)}")
            return False
        else:
            print_success(f"Config valid - RAG: {config['enable_rag']}, STT: {config['stt_backend']}, TTS: {config['tts_backend']}")
            return True
    except Exception as e:
        print_error(f"Invalid config.json: {e}")
        return False

def create_vectorstore():
    """Create FAISS vectorstore from knowledge base"""
    print_info("Creating vectorstore from knowledge base...")

    try:
        from indu_rag import InduAgent

        # Delete existing vectorstore
        vectorstore_path = Path("indu_vectorstore")
        if vectorstore_path.exists():
            print_info("Removing existing vectorstore...")
            import shutil
            shutil.rmtree(vectorstore_path)

        # Create new agent (will build vectorstore)
        print_info("Building new vectorstore (this may take a minute)...")
        agent = InduAgent(debug_mode=False)

        print_success("Vectorstore created successfully")

        # Verify vectorstore
        if vectorstore_path.exists():
            files = list(vectorstore_path.glob("*"))
            print_success(f"Vectorstore files created: {len(files)} files")
            return True
        else:
            print_error("Vectorstore directory not created")
            return False

    except Exception as e:
        print_error(f"Failed to create vectorstore: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_directories():
    """Create necessary directories"""
    print_info("Creating directories...")

    directories = ['filler_audio', 'static', 'templates']

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print_success("Directories created")
    return True

def main():
    """Main setup function"""
    print_header("INDU Brain v2.1 Setup")
    print_info("Initializing deployment setup...\n")

    checks = [
        ("Python Version", check_python_version),
        ("Required Files", check_files),
        ("Configuration", validate_config),
        ("Dependencies", check_dependencies),
        ("Ollama Connection", check_ollama),
        ("Directories", create_directories),
        ("Vectorstore", create_vectorstore),
    ]

    failed = []

    for check_name, check_func in checks:
        print_header(check_name)
        if not check_func():
            failed.append(check_name)

    print_header("Setup Complete")

    if failed:
        print_error(f"\nSetup failed. Issues found in: {', '.join(failed)}")
        print_info("\nPlease fix the issues above and run setup.py again.")
        sys.exit(1)
    else:
        print_success("\n🎉 Setup completed successfully!")
        print_info("\nYou can now start the server:")
        print_info("  python server.py")
        print_info("\nThen open your browser to:")
        print_info("  http://localhost:5000")
        sys.exit(0)

if __name__ == "__main__":
    main()

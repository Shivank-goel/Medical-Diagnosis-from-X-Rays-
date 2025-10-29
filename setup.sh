#!/bin/bash

# Medical X-Ray CNN Setup Script
# This script sets up the environment and prepares the project for running

echo "🔬 Medical X-Ray CNN Diagnosis System Setup"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📁 Project structure:"
echo "├── medical_xray_cnn.py    # Main training script"
echo "├── requirements.txt       # Dependencies"
echo "├── data/                  # Place your X-ray images here"
echo "│   ├── train/"
echo "│   │   ├── normal/"
echo "│   │   └── pneumonia/"
echo "│   ├── val/"
echo "│   │   ├── normal/"
echo "│   │   └── pneumonia/"
echo "│   └── test/"
echo "│       ├── normal/"
echo "│       └── pneumonia/"
echo "├── models/                # Trained models will be saved here"
echo "└── outputs/               # Results and visualizations"
echo ""
echo "🚀 To run the project:"
echo "1. Add your X-ray images to the appropriate data directories"
echo "2. Activate the virtual environment: source .venv/bin/activate"
echo "3. Run the training script: python medical_xray_cnn.py"
echo ""
echo "📝 Note: The script will create sample directory structure if no images are found."
echo "You can download X-ray datasets from sources like Kaggle or NIH."

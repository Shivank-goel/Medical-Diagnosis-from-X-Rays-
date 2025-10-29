@echo off
REM Medical X-Ray CNN Setup Script for Windows
REM This script sets up the environment and prepares the project for running

echo 🔬 Medical X-Ray CNN Diagnosis System Setup
echo ==============================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3 is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

echo ✅ Python 3 found

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📋 Installing requirements...
pip install -r requirements.txt

echo.
echo ✅ Setup completed successfully!
echo.
echo 📁 Project structure:
echo ├── medical_xray_cnn.py    # Main training script
echo ├── requirements.txt       # Dependencies
echo ├── data/                  # Place your X-ray images here
echo │   ├── train/
echo │   │   ├── normal/
echo │   │   └── pneumonia/
echo │   ├── val/
echo │   │   ├── normal/
echo │   │   └── pneumonia/
echo │   └── test/
echo │       ├── normal/
echo │       └── pneumonia/
echo ├── models/                # Trained models will be saved here
echo └── outputs/               # Results and visualizations
echo.
echo 🚀 To run the project:
echo 1. Add your X-ray images to the appropriate data directories
echo 2. Activate the virtual environment: .venv\Scripts\activate.bat
echo 3. Run the training script: python medical_xray_cnn.py
echo.
echo 📝 Note: The script will create sample directory structure if no images are found.
echo You can download X-ray datasets from sources like Kaggle or NIH.

pause

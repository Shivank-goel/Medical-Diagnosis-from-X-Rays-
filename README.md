# 🔬 Medical Diagnosis from X-Rays

A deep learning CNN-based system for medical diagnosis from X-ray images using transfer learning with EfficientNetB0. This project includes model explainability through Grad-CAM visualizations.

## 🌟 Features

- **Transfer Learning**: Uses EfficientNetB0 pre-trained on ImageNet
- **Data Augmentation**: On-the-fly augmentation for better generalization
- **Model Explainability**: Grad-CAM visualizations to understand model decisions
- **Comprehensive Evaluation**: Classification reports and confusion matrices
- **Easy Setup**: Automated environment setup scripts
- **Structured Pipeline**: Complete training, validation, and testing workflow

## 📁 Project Structure

```
Medical-Diagnosis-from-X-Rays-/
├── medical_xray_cnn.py        # Main training script
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script for Linux/macOS
├── setup.bat                  # Setup script for Windows
├── data/                      # Dataset directory
│   ├── train/
│   │   ├── normal/           # Normal X-ray images
│   │   └── pneumonia/        # Pneumonia X-ray images
│   ├── val/
│   │   ├── normal/
│   │   └── pneumonia/
│   └── test/
│       ├── normal/
│       └── pneumonia/
├── models/                    # Trained models
│   ├── best_xray_model.h5    # Best model from training
│   └── final_xray_model.h5   # Final model
└── outputs/                   # Results and visualizations
    ├── confusion_matrix.png   # Confusion matrix plot
    └── gradcam_*.jpg          # Grad-CAM visualizations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Option 1: Automated Setup (Recommended)

**For Linux/macOS:**
```bash
./setup.sh
```

**For Windows:**
```cmd
setup.bat
```

### Option 2: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shivank-goel/Medical-Diagnosis-from-X-Rays-.git
   cd Medical-Diagnosis-from-X-Rays-
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate virtual environment:**
   
   **Linux/macOS:**
   ```bash
   source .venv/bin/activate
   ```
   
   **Windows:**
   ```cmd
   .venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Setup

1. **Download X-ray dataset** (e.g., from Kaggle):
   - [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
   - [NIH Chest X-rays](https://www.kaggle.com/nih-chest-xrays/data)

2. **Organize your data** in the following structure:
   ```
   data/
   ├── train/
   │   ├── normal/        # Normal chest X-rays
   │   └── pneumonia/     # Pneumonia chest X-rays
   ├── val/
   │   ├── normal/
   │   └── pneumonia/
   └── test/
       ├── normal/
       └── pneumonia/
   ```

3. **Supported formats**: `.jpg`, `.jpeg`, `.png`

## 🏃‍♂️ Running the Model

1. **Activate the virtual environment** (if not already active):
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Run the training script:**
   ```bash
   python medical_xray_cnn.py
   ```

## 🔧 Configuration

You can modify the following parameters in `medical_xray_cnn.py`:

```python
BATCH_SIZE = 16          # Batch size for training
IMG_SIZE = (224, 224)    # Input image size
EPOCHS = 25              # Number of training epochs
```

## 📈 Model Architecture

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Transfer Learning**: Two-phase training
  - Phase 1: Frozen base model (25 epochs)
  - Phase 2: Fine-tuning (10 additional epochs)
- **Data Augmentation**: 
  - Horizontal flip
  - Random rotation (±5%)
  - Random zoom (±5%)
- **Regularization**: Dropout layers (0.3 and 0.2)

## 📊 Outputs

The model generates:

1. **Training History**: Loss and accuracy plots
2. **Confusion Matrix**: Visual representation of model performance
3. **Classification Report**: Precision, recall, and F1-score metrics
4. **Grad-CAM Visualizations**: Heatmaps showing important image regions
5. **Saved Models**: Best and final model weights

## 🎯 Performance

- **Accuracy**: 90%+ on test set
- **Model Explainability**: Grad-CAM highlights pneumonia-affected areas
- **Fast Training**: Transfer learning reduces training time significantly

## 🔍 Model Explainability

The project includes Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize:
- Which parts of the X-ray the model focuses on
- How confident the model is in its predictions
- Whether the model is looking at clinically relevant areas

## 📋 Requirements

- `tensorflow>=2.10.0`
- `matplotlib>=3.5.0`
- `scikit-learn>=1.1.0`
- `opencv-python>=4.6.0`
- `numpy>=1.21.0`
- `Pillow>=9.0.0`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚨 Disclaimer

This project is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and approval from medical professionals.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section below
2. Open an issue on GitHub
3. Contact the maintainers

## 🛠️ Troubleshooting

**Issue: "No images found in data directories"**
- Solution: Make sure you've added X-ray images to the train, val, and test directories

**Issue: "CUDA out of memory"**
- Solution: Reduce `BATCH_SIZE` in the script (try 8 or 4)

**Issue: "Module not found"**
- Solution: Make sure the virtual environment is activated and all dependencies are installed

**Issue: "Poor model performance"**
- Solution: Ensure you have balanced datasets and enough training images (at least 100+ per class)

---

Made with ❤️ for medical AI research

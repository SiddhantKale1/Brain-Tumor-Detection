# Brain Tumor MRI Detection & Classification

A deep learning system for classifying brain MRI scans into four categories with explainable visual attention (Grad-CAM) and a modern Flask web interface.

---

## 🧠 Supported Categories
The model classifies brain MRI slices into one of four classes:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor** (Healthy Brain Tissue)

---

## 🚀 Key Features

* **Transfer Learning Architecture:** Utilizes a pretrained `ResNet50` backbone with custom classification layers, dropout regularization, and L2 weight decay.
* **Anatomically Sound Data Augmentation:** Restricts spatial transformations to horizontal flipping, subtle rotations, and zooms, preserving the cranial/caudal anatomical orientation essential for localized brain structures (such as the pituitary gland).
* **Balanced Class Weighting:** Inverse-frequency class weighting ensures minority classes (like pituitary tumors) are effectively learned without majority-class bias.
* **Explainable AI (Grad-CAM):** Generates gradient-weighted class activation heatmaps showing the specific region of the MRI that influenced the prediction.
* **Pre-Inference MRI Validator:** Validates uploaded images against MRI-specific characteristics (grayscale distribution, contrast variation, structural edge density) to reject invalid or unrelated images.
* **Interactive Web Interface:** Clean Flask application with drag-and-drop upload, confidence bars, and real-time Grad-CAM overlay.

---

## 📁 Project Structure

```text
Brain-Tumor-Detection/
├── ALGORITHM_DOCUMENTATION.md   # Comprehensive technical report
├── README.md                    # Setup and usage guide
├── requirements.txt             # Python dependencies
├── app.py                       # Flask web application
├── main.py                      # Two-stage training and fine-tuning pipeline
├── debug_model.py               # Model evaluation, confusion matrix & metrics
├── gradcam.py                   # Grad-CAM heatmap generation engine
├── mri_validator.py             # Structural MRI validation heuristics
├── prepare_dataset_splits.py    # Leakage-safe dataset splitting & deduplication
└── templates/                   # HTML5 templates
    ├── index.html               # Landing page
    ├── detect.html              # Upload page
    ├── results.html             # Diagnosis & Grad-CAM results view
    └── shared_styles.html       # Shared CSS design system
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone git@github.com:SiddhantKale1/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### 2. Create a virtual environment (Python 3.10 – 3.12 recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

The pipeline is designed for the standard [Kaggle Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset.

1. Download and extract the dataset into a `dataset/` directory:
   ```text
   dataset/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```

2. Run the splitting script to deduplicate files and generate clean train/val/test splits:
   ```bash
   python prepare_dataset_splits.py
   ```

---

## 🏋️ Training the Model

To train the model using frozen-base feature extraction followed by deeper fine-tuning:
```bash
python main.py
```
This produces:
* `initial_weights.weights.h5` — Weights from Stage 1 (frozen base).
* `brain_tumor_model.keras` — Checkpointed best model based on validation accuracy.

---

## 🔍 Diagnostics & Model Evaluation

To generate the classification report, confusion matrix, and class distribution plots:
```bash
python debug_model.py
```
Visualizations will be saved to the `debug_outputs/` directory.

---

## 🌐 Running the Web Application

To start the local Flask inference server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

---

## ⚠️ Medical Disclaimer

This project is developed for educational, research, and technical demonstration purposes only. It is **not** certified as a medical device and must **never** be used as a standalone diagnostic tool in clinical practice. Always consult a qualified medical professional or radiologist for medical diagnosis.

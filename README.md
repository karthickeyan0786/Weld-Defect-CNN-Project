Save the following content as `README.md` in the root directory of your project.

```markdown
# Industrial Weld Defect Detection (RIAWELC)

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains a complete PyTorch deep learning pipeline for detecting and classifying industrial weld defects from radiography (X-ray) images. Built around the RIAWELC dataset, the project implements custom Convolutional Neural Networks (CNNs) and two-stage transfer learning to categorize weld conditions into four distinct classes (CR, LP, ND, PO).

Recognizing the spatial data leakage vulnerabilities inherent in cropped industrial datasets, this repository includes a dedicated evidence-based data leakage audit framework utilizing MD5 and Perceptual Hashing (pHash).

## 🗂 Project Structure

```text
├── configs/
│   ├── model_a.yaml           # Hyperparameters & settings for Model A
│   ├── model_b.yaml           # Hyperparameters & settings for Model B
│   └── model_c.yaml           # Hyperparameters & settings for Model C
├── src/
│   ├── models/
│   │   ├── baseline_cnn.py    # Model A: Baseline custom CNN
│   │   ├── weldnet.py         # Model B: WelDeNet advanced architecture
│   │   └── transfer_learning.py # Model C: Transfer learning backbone
│   ├── dataset.py             # PyTorch Dataset class (WeldDataset)
│   ├── augmentation.py        # Train/Eval image transforms (CLAHE, etc.)
│   ├── train_bc.py            # Primary training script with Early Stopping
│   ├── test_folder.py         # Batch inference script for folder of images
│   ├── leakage_audit.py       # Hashing and hold-out evaluation audit
│   └── utils.py               # Seed setting and helper functions
├── outputs/
│   ├── checkpoints/           # Saved .pth weights
│   └── plots/                 # Training curves and confusion matrices
├── experiment_results.csv     # Auto-generated metrics tracking
├── requirements.txt
└── README.md

```

## 🧠 Model Architectures

* **Model A (Baseline CNN):** A lightweight custom convolutional neural network designed to establish baseline performance on the dataset.
* **Model B (WelDeNet):** A deeper, optimized custom architecture incorporating dropout and batch normalization to capture complex industrial textures.
* **Model C (Transfer Learning):** A pre-trained backbone utilizing a specialized two-stage training loop:
* **Stage 1:** Feature extraction (backbone frozen, classifier trained).
* **Stage 2:** Fine-tuning (full network unfrozen with a reduced learning rate).



## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/weld-defect-detection.git](https://github.com/yourusername/weld-defect-detection.git)
cd weld-defect-detection

```


2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


*Required packages: `torch`, `torchvision`, `pandas`, `scikit-learn`, `Pillow`, `pyyaml`, `tqdm`, `imagehash`.*
4. **Data Preparation:**
Ensure your dataset is split into `train`, `val`, and `test` directories inside your configured split folder.

## 🚀 Usage

### 1. Training a Model

To train a model, run `train_bc.py` and pass the model identifier (`A`, `B`, or `C`). The script automatically loads the corresponding `.yaml` configuration file.

```bash
# Set PYTHONPATH to root and train Model B
set PYTHONPATH=. && python src/train_bc.py --model B

```

*Note: You can safely interrupt training (`Ctrl+C`); the script automatically saves the `best_model.pth` based on validation loss. Model C supports resuming directly into Stage 2 Fine-tuning.*

### 2. Batch Inference

To run predictions on a completely new folder of unannotated X-ray patches and generate a CSV report of exact class probabilities:

```bash
set PYTHONPATH=. && python src/test_folder.py --model B --folder path/to/test_images --output batch_results.csv

```

### 3. Running the Leakage Audit

Industrial datasets generated via sliding-window crops (like RIAWELC) are highly susceptible to spatial data leakage, which can artificially inflate accuracy to >99%. Verify your splits before trusting validation metrics.

```bash
set PYTHONPATH=. && python src/leakage_audit.py --config configs/model_b.yaml

```

The audit script will:

* Verify the existence of a dedicated test set.
* Compute exact duplicates (MD5) across train/val/test.
* Compute near-duplicates (pHash) to detect overlapping sub-crops.
* Run a strict, unbiased hold-out evaluation generating a true confusion matrix.

## 📊 Outputs & Tracking

All training runs automatically append their performance metrics (Accuracy, Precision, Recall, F1-Score, Inference Time, Model Size) to `experiment_results.csv` for easy comparative analysis.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

```

```

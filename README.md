# Automatic Classification of Martian Surface Features Using CNNs

A deep learning project that uses a Convolutional Neural Network (CNN) to classify images of Martian surface features into 8 distinct categories, achieving ~90.8% overall accuracy.

---

## 📋 Project Overview

This project builds and trains a CNN model to automatically classify images of the Martian surface. The model is trained on a labelled image dataset and learns to distinguish between 8 geological/surface feature classes.

### Classes
| ID | Class Name |
|----|------------|
| 0 | Other |
| 1 | Crater |
| 2 | Dark Dune |
| 3 | Slope Streak |
| 4 | Bright Dune |
| 5 | Impact Ejecta |
| 6 | Swiss Cheese |
| 7 | Spider |

---

## 🧠 Model Architecture

A custom sequential CNN with 3 convolutional blocks followed by fully connected layers:

```
Input (128×128×3)
  → Conv2D(32) + ReLU + MaxPool  →  64×64×32
  → Conv2D(64) + ReLU + MaxPool  →  32×32×64
  → Conv2D(128) + ReLU + MaxPool →  16×16×128
  → Flatten → Dense(256) + Dropout(0.3)
  → Output Dense(8) + Softmax
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Sparse Categorical Cross-Entropy
- **Regularization:** L2 (λ=0.0001) + Dropout
- **Early Stopping:** patience=5, restores best weights

---

## 📁 Repository Structure

```
├── Deep_learning_midterm_final.ipynb   # Main notebook
├── labels-map-proj-v3.txt              # Raw label file
├── landmarks_map-proj-v3_classmap.csv  # Generated CSV with filenames + classes
├── map-proj-v3/                        # Image dataset directory
├── cnn_architecture.png                # CNN diagram
├── accuracy.png                        # Training vs test accuracy plot
├── loss.png                            # Training vs test loss plot
├── class_accuracy_plot.png             # Per-class accuracy bar chart
├── training_history.csv                # Saved training metrics
├── final_model.keras                   # Saved final model
└── best_model.keras                    # Saved best checkpoint
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib pillow
```

### Running the Notebook

1. Clone the repository and place the `map-proj-v3/` image folder and `labels-map-proj-v3.txt` in the root directory.
2. Open `Deep_learning_midterm_final.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells in order.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Training Accuracy | 90.79% |
| Test Accuracy | 90.85% |
| Loss Difference (Test − Train) | −0.0041 |

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| Bright Dune | 98.6% |
| Other | 91.4% |
| Crater | 61.1% |
| Spider | 47.4% |
| Dark Dune | 28.1% |
| Impact Ejecta | 17.2% |
| Swiss Cheese | 5.3% |
| Slope Streak | 4.3% |

> **Note:** Lower accuracy on minority classes (Slope Streak, Swiss Cheese, Impact Ejecta) reflects dataset class imbalance. Frequent and visually distinct classes like Bright Dune perform significantly better.

---

## 🔧 Key Design Decisions

- **Stratified train/test split (80/20)** to preserve class distribution.
- **Data augmentation** (random flip, rotation ±5°, zoom ±5%) applied to the training set only.
- **L2 regularization + Dropout** to reduce overfitting.
- **Early stopping** to prevent over-training and restore the best checkpoint automatically.

---

## 🔮 Future Improvements

- Address class imbalance via oversampling (SMOTE) or class-weighted loss.
- Experiment with transfer learning (e.g., ResNet50, EfficientNet) pre-trained on ImageNet.
- Increase image resolution beyond 128×128 for finer texture discrimination.
- Apply more aggressive augmentation for underrepresented classes.

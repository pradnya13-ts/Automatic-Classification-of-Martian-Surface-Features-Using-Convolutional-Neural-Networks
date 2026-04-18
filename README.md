<div align="center">

![Mars Banner](https://media.giphy.com/media/xT9IgG50Lg7russbD6/giphy.gif)

# 🔴 Martian Surface Feature Classifier

### Automatic Classification of Mars Orbital Images Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-90.85%25-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20HiRISE-red?style=for-the-badge&logo=nasa)
![Model](https://img.shields.io/badge/Model-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface)

<br/>

*A CNN trained to recognize 8 geological features on the Martian surface from NASA HiRISE orbital imagery.*

</div>

---

## 🌕 What Does This Project Do?

<img src="https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif" width="180" align="right"/>

This project trains a **Convolutional Neural Network (CNN)** on NASA's HiRISE Mars satellite imagery to automatically identify surface features — craters, dunes, spider formations, and more.

The full ML pipeline is covered end to end:

- 🧹 Data preprocessing & label encoding
- 📊 Exploratory data analysis & class distribution
- 🏗️ CNN architecture design
- 🏋️ Model training with augmentation & regularization
- 📈 Evaluation with per-class accuracy breakdown

---

## 🗂️ Repository Structure

```
📦 mars-surface-cnn
 ┣ 📓 CNN.ipynb                        — Main training & evaluation notebook
 ┣ 📄 CNN_report.pdf                   — Full project report
 ┣ 🖼️  cnn_architecture.png            — Model architecture diagram
 ┣ 📊 training_history.csv             — Epoch-by-epoch loss & accuracy log
 ┗ 📖 README.md
```

> 📦 The image dataset and `.keras` model files are hosted externally — see **Downloads** below.

---

## 📦 Downloads

| Resource | Link |
|----------|------|
| 🛰️ NASA HiRISE Mars Image Dataset | [data.nasa.gov](https://data.nasa.gov/dataset/mars-orbital-image-hirise-labeled-data-set-version-3/resource/c93bf426-1eae-4d3b-8afd-548add5e24ce) |
| 🤖 Trained Model (`final_model.keras`) | [huggingface.co/pradnyabhakare829/mars-surface-cnn](https://huggingface.co/pradnyabhakare829/mars-surface-cnn/blob/main/final_model.keras) |

After downloading, arrange files like this before running the notebook:

```
├── CNN.ipynb
├── labels-map-proj-v3.txt
├── landmarks_map-proj-v3_classmap.csv
├── final_model.keras                   ← download from Hugging Face
└── map-proj-v3/                        ← extracted NASA image folder
```

> ⚠️ **Download `final_model.keras` before running** the evaluation cells.

---

## 🧠 Model Architecture

<div align="center">

```
                    ┌─────────────────────┐
                    │   Input 128×128×3   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Conv2D(32) + ReLU  │
                    │    MaxPool(2×2)     │  → 64×64×32
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Conv2D(64) + ReLU  │
                    │    MaxPool(2×2)     │  → 32×32×64
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Conv2D(128) + ReLU  │
                    │    MaxPool(2×2)     │  → 16×16×128
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │       Flatten       │  → 32,768
                    │    Dense(256)       │
                    │    Dropout(0.3)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Output Dense(8)    │
                    │      Softmax        │
                    └─────────────────────┘
```

</div>

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Sparse Categorical Cross-Entropy |
| Regularization | L2 (λ=0.0001) + Dropout |
| Early Stopping | patience=5, restore best weights |
| Image Size | 128×128 px |
| Batch Size | 32 |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/pradnyabhakare829/mars-surface-cnn.git
cd mars-surface-cnn
```

### 2. Install dependencies
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib pillow
```

### 3. Download dataset & model
- Dataset → [NASA HiRISE](https://data.nasa.gov/dataset/mars-orbital-image-hirise-labeled-data-set-version-3/resource/c93bf426-1eae-4d3b-8afd-548add5e24ce)
- Model → [Hugging Face](https://huggingface.co/pradnyabhakare829/mars-surface-cnn)

### 4. Run the notebook
```bash
jupyter notebook CNN.ipynb
```

---

## 📊 Results

<div align="center">

| Metric | Score |
|--------|-------|
| 🏋️ Training Accuracy | 90.79% |
| ✅ Test Accuracy | **90.85%** |
| 📉 Loss Difference (Test − Train) | −0.0041 |

</div>

### 🔬 Per-Class Accuracy

```
Bright Dune    ████████████████████  98.6%
Other          ██████████████████░░  91.4%
Crater         ████████████░░░░░░░░  61.1%
Spider         █████████░░░░░░░░░░░  47.4%
Dark Dune      █████░░░░░░░░░░░░░░░  28.1%
Impact Ejecta  ███░░░░░░░░░░░░░░░░░  17.2%
Swiss Cheese   █░░░░░░░░░░░░░░░░░░░   5.3%
Slope Streak   █░░░░░░░░░░░░░░░░░░░   4.3%
```

> 💡 Class imbalance explains the gap — frequent classes like *Bright Dune* and *Other* dominate training signal, leaving rarer classes like *Slope Streak* and *Swiss Cheese* underrepresented.

---

## 🏷️ Surface Feature Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | 🌑 Other | Background / unclassified terrain |
| 1 | 🕳️ Crater | Impact craters from meteorites |
| 2 | 🌊 Dark Dune | Dark sand dune formations |
| 3 | 〰️ Slope Streak | Dust avalanche streaks on slopes |
| 4 | 🏜️ Bright Dune | Bright sand dune formations |
| 5 | 💥 Impact Ejecta | Material ejected from impacts |
| 6 | 🧀 Swiss Cheese | CO₂ sublimation pits near south pole |
| 7 | 🕷️ Spider | Araneiform terrain patterns |

---

## 🔧 Key Design Decisions

- **Stratified 80/20 split** — preserves class distribution across train/test sets
- **Data augmentation** — random flip, rotation ±5°, zoom ±5% on training only
- **L2 + Dropout** — combats overfitting on imbalanced data
- **Early stopping** — halts training when val_loss stalls, restores best checkpoint

---

## 🔮 Future Improvements

- [ ] Address class imbalance with SMOTE or weighted loss
- [ ] Fine-tune a pretrained ResNet50 / EfficientNet via transfer learning
- [ ] Increase input resolution beyond 128×128
- [ ] Add Grad-CAM visualisation to interpret model decisions

---

<div align="center">

![Space GIF](https://media.giphy.com/media/l4FGCymGGNTZVZwtO/giphy.gif)

*Made with ❤️ and a lot of Martian dust*

</div>

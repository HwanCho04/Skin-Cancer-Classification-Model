# Skin Cancer Classification
A machine learning classification project that predicts **skin cancer status (benign vs malignant)** using demographic, behavioral, biological, and environmental risk factors. The project compares multiple models (GLM, Random Forest, XGBoost) and evaluates performance using **Accuracy and ROC AUC**.

> Slides included: **Skin Cancer Classification Report – Hwan Cho**
---

## Features

### Dataset & Problem Setup
- Uses a skin cancer risk dataset with demographic, behavioral, and clinical features
- Binary classification target:
  - **Benign:** 0
  - **Malignant:** 1
- Train/validation split: **80% train / 20% validation**
- Test set provided without labels for final submission
- Metrics:
  - **Accuracy**
  - **ROC AUC** (threshold-independent performance)

---

### Feature Engineering & Selection
- Missing numeric values:
  - Filled using **median imputation**
  - Missingness indicator flags added
- Missing categorical values:
  - Replaced with `"Unknown"`
- One-hot encoding applied to categorical variables
- **LASSO regularization** used to identify important predictors and reduce noise
- Separate Random Forest trained using **top features only** for comparison

---

## Models Implemented
- **Generalized Linear Model (Logistic Regression)** – baseline, interpretable
- **Random Forest (All Features)**
  - 800 trees
  - Depth constrained to reduce overfitting
- **Random Forest (Important Features Only)**
  - Uses LASSO-selected predictors
- **Regularized XGBoost (Final Model)**
  - L1 regularization
  - Early stopping
  - Extensive hyperparameter tuning

---

### Evaluation & Visualization
- Confusion matrices for each model
- ROC curves for validation performance
- Feature importance plots for tree-based models
- Classification threshold optimization based on validation accuracy

---

### Key Findings
- **Age** is the strongest predictor of malignant skin cancer
- **Family history** significantly increases risk
- **UV exposure and sunburn history** show meaningful interaction effects
- Behavioral factors (e.g., sunscreen use) reduce modeled cancer risk
- Tree-based models outperform linear models in capturing nonlinear medical relationships

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/skin-cancer-classification.git
cd skin-cancer-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python skin_cancer_model.py
```

This will:
- Preprocess the data
- Train models
- Generate ROC curves
- Output prediction files for submission

## Project Structure
skin-cancer-classification/
├── data/
│   ├── SkinCancerTrain.csv
│   └── SkinCancerTestNoY.csv
├── scripts/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── outputs/
│   ├── submission_rf_important_features.csv
│   └── rf_finalsubmission_importantvars.csv
├── slides/
│   └── Skin_Cancer_Classification_Hwan_Cho.pptx
├── README.md
└── LICENSE

## Development

### Reproducibility
- Fixed random seeds for all train/validation splits
- Feature preprocessing fit on training data only
- Threshold selection performed on validation data

### (Optional) Future Refactor
- Modular pipelines using sklearn.Pipeline
- Model ensembling across Random Forest and XGBoost
- Cost-sensitive learning to penalize false negatives

## Contributions
To contribute:
1. Fork the repository
2. Create a feature branch (git checkout -b feature/YourFeature)
3. Commit your changes (git commit -m "Add YourFeature")
4. Push to the branch (git push origin feature/YourFeature)
5. Open a Pull Request

## License
This project is licensed under the MIT License.

## Acknowledgments
- Skin Cancer Foundation and Johns Hopkins Medicine
- scikit-learn for modeling and evaluation
- XGBoost for gradient boosting
- matplotlib and seaborn for visualization


# MetaHydroPred

**MetaHydroPred** is a meta-learning framework for predicting **current density** and **H₂ production rate** in microbial electrochemical cells (MECs). The framework combines predictions from multiple baseline machine-learning models using a stacked meta-model to improve robustness across heterogeneous substrates and experimental conditions.

**Author:** Nayan Dash  
**Date:** 3 Aug 2024  

---

## Supported Predictions
- Current density  
- H₂ production rate  

## Supported Substrate Types
- All-organic  
- Acetate  
- Complex substrates  

---

## Method Summary
- Baseline models are trained on different base feature sets (BF sets).
- Meta-features are generated using LOOCV-based stacking.
- A pre-trained meta-model integrates baseline predictions.
- Predictions are inverse-transformed to the original scale.

---

## Usage

### Current Density Prediction
```bash
python predict_current_density.py \
  --substrate_type all-organic \
  --input_csv_path test.csv \
  --output_csv_path current_density_predictions.csv
```

### H₂ Production Rate Prediction
```bash
python predict_h2_production_rate.py \
  --substrate_type acetate \
  --input_csv_path test.csv \
  --output_csv_path h2_production_rate_predictions.csv
```

---

## Input / Output

**Input**
- Test dataset in CSV format
- Pre-trained baseline models (`.joblib`)
- Pre-trained meta-model (`.joblib`)
- Meta-feature training CSV (for feature alignment)

**Output**
- CSV file containing:
  - `Current density` **or**
  - `H2 production rate`

---

## Requirements
- Python ≥ 3.9  
- NumPy  
- SciPy  
- pandas  
- scikit-learn  
- XGBoost  
- CatBoost  
- LightGBM  
- joblib  

(Exact library versions are printed at runtime.)

---

## Reproducibility
- Fixed random seed used across all models
- LOOCV-based meta-feature generation
- Consistent feature scaling and transformations
- Feature selection based on correlation filtering and tree-based importance scoring

---

## Data and Code Availability

- **Web server:**  
  https://balalab-skku.org/MetaHydroPred/  
  (All processed datasets used in this study are available for download.)

- **Source code:**  
  Publicly available on GitHub:  
  **[GitHub repository URL]**

---

## Citation
If you use this code, please cite the associated manuscript:

> *MetaHydroPred: a meta-learning framework for predicting current density and H₂ production rate in microbial electrochemical cells.*

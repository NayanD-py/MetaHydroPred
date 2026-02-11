# MetaHydroPred

**MetaHydroPred** is a meta-learning framework for predicting **current density** and **H₂ production rate** in microbial electrolysis cells (MECs). The framework combines predictions from multiple baseline machine-learning models using a stacked meta-model to improve robustness across heterogeneous substrates and experimental conditions.

---

## Supported Predictions
- Current density  
- H₂ production rate  

## Supported Substrate Types
- All-organic  
- Acetate  
- Complex substrates  


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

**Output**
- CSV file containing:
  - `Current density` **or**
  - `H2 production rate`

---


- **Web server:**  
  https://balalab-skku.org/MetaHydroPred/  
  (All processed datasets used in this study are available for download.)
---

> *MetaHydroPred: a meta-learning framework for predicting current density and H₂ production rate in microbial electrochemical cells.*

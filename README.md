# Fraud Detection in High-Frequency Trade Streams

This repository provides an **end-to-end workflow** for detecting fraudulent transactions in high-frequency trading data.  
It includes **Exploratory Data Analysis (EDA)** and **Ensemble Model Comparison** to identify the most effective algorithms for fraud detection.

---

## Project Structure

```bash
Fraud-Detection-in-High-Frequency-Trade-Streams/
├── fraud_detection_eda.py                  # Exploratory Data Analysis for IEEE Fraud dataset
├── fraud_model_ensemble_comparison.py      # Compare ensemble models for fraud detection
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── Dataset EDA.zip                         # Contains EDA plots and visualizations
└── Model Results.zip                       # Contains model comparison charts and results
```

---

## Dataset Summary

The dataset is based on the **IEEE-CIS Fraud Detection dataset**, consisting of **590,540 transactions** and multiple numerical and categorical features.

Key feature statistics:

| Feature | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| TransactionID | 3,282,270 | 170,474 | 2,987,000 | 3,577,539 |
| TransactionDT | 7,372,311 | 4,617,224 | 86,400 | 15,811,130 |
| TransactionAmt | 135.03 | 239.16 | 0.25 | 31,937.39 |

---

## Model Performance Comparison

Below is a detailed comparison of multiple ensemble-based models evaluated on the dataset:

| Model | Accuracy | F1-Score | Precision | Recall | AUC | AP |
|--------|-----------|----------|------------|---------|------|------|
| **XGBoost (Boosting)** | 0.9806 | 0.9372 | 0.9847 | 0.8949 | 0.9909 | 0.9768 |
| **LightGBM (Boosting)** | 0.9754 | 0.9359 | 0.9821 | 0.8915 | 0.9803 | 0.9769 |
| **CatBoost (Boosting)** | 0.9751 | 0.9295 | 0.9847 | 0.8465 | 0.9845 | 0.9566 |
| **Stacking (Meta-Ensemble)** | 0.9728 | 0.9411 | 0.9674 | 0.8686 | 0.9825 | 0.9650 |
| **RandomForest (Bagging)** | 0.9648 | 0.8837 | 0.9628 | 0.8033 | 0.9785 | 0.9560 |
| **ExtraTrees (Bagging)** | 0.9330 | 0.7341 | 0.9633 | 0.6058 | 0.9531 | 0.8948 |
| **Hybrid Deep Ensemble** | 0.9378 | 0.7914 | 0.8972 | 0.7080 | 0.9545 | 0.8825 |

**Insights:**
- Boosting algorithms (XGBoost, LightGBM, CatBoost) achieved the highest accuracy and precision.
- Meta-ensemble stacking performed robustly, combining multiple models to balance recall and precision.
- Bagging methods (RandomForest, ExtraTrees) performed well but had lower recall.
- The hybrid deep ensemble demonstrated potential for combining traditional and deep learning methods, though with slightly lower metrics.

---

## Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/waliapriyanshu/Fraud-Detection-in-High-Frequency-Trade-Streams.git
   cd Fraud-Detection-in-High-Frequency-Trade-Streams
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run EDA**
   ```bash
   python fraud_detection_eda.py
   ```

4. **Compare Ensemble Models**
   ```bash
   python fraud_model_ensemble_comparison.py
   ```

5. **Check results**
   - Dataset visualizations → `Dataset EDA.zip`
   - Model performance charts → `Model Results.zip`

---

## Dataset Reference

This project utilizes the **IEEE-CIS Fraud Detection dataset**, available on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection).

---

## Author

**Rashika Ranjan, Sanskriti Mahore, Priyanshu Walia, Ryansh Arora**

---

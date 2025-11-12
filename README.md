# Fraud Detection in High-Frequency Trade Streams

This repository contains an end-to-end workflow for detecting fraudulent transactions in high-frequency trading data. The project includes Exploratory Data Analysis (EDA) and Ensemble Model Comparison to evaluate various fraud detection techniques.

---

## 📂 Project Structure

Fraud-Detection-in-High-Frequency-Trade-Streams/
│
├── fraud_detection_eda.py                  # Exploratory Data Analysis for IEEE Fraud dataset
├── fraud_model_ensemble_comparison.py      # Compare ensemble models for fraud detection
├── /images/eda_plots/                      # Folder for EDA visualizations
├── /images/model_results/                  # Folder for model comparison charts
└── README.md                               # Project documentation

---

## 🚀 Features

- Data cleaning and preprocessing for IEEE-CIS Fraud Detection dataset  
- Comprehensive EDA with feature correlation and trend analysis  
- Comparison of ensemble learning algorithms (Random Forest, XGBoost, LightGBM, etc.)  
- Model evaluation based on accuracy, precision, recall, and ROC-AUC metrics  

---

## 📊 Results

- Insights on the most impactful features for detecting fraud  
- Performance comparison across ensemble models  
- Visualization of feature importances and ROC curves  

Example image embeds (once uploaded):

![EDA Overview](images/eda_plots/feature_distribution.png)
![Model Comparison](images/model_results/roc_auc_comparison.png)

---

## 🛠️ Requirements

To install dependencies, run:

```bash
pip install -r requirements.txt
```

---

## 🧠 Usage

1. Clone the repository  
   ```bash
   git clone https://github.com/waliapriyanshu/Fraud-Detection-in-High-Frequency-Trade-Streams.git
   cd Fraud-Detection-in-High-Frequency-Trade-Streams
   ```

2. Run EDA  
   ```bash
   python fraud_detection_eda.py
   ```

3. Compare Ensemble Models  
   ```bash
   python fraud_model_ensemble_comparison.py
   ```

---

## 📸 Visualizations

All generated plots and charts are stored in:

- `images/eda_plots/` → For exploratory data analysis visualizations  
- `images/model_results/` → For model performance charts  

---

## 📚 Dataset

This project uses the **IEEE-CIS Fraud Detection dataset** (available on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)).

---

## 👨‍💻 Author

**Waliaji (Priyanshu Walia)**  
GitHub: [@waliapriyanshu](https://github.com/waliapriyanshu)

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

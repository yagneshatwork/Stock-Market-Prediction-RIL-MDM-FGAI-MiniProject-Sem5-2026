# 📈 Reliance Industries Stock Price Prediction

> **Mini Project — Semester 5 (2026)**  
> **Domain:** Machine Learning & Deep Learning / Financial Generative AI (FGAI / MDM)  
> **Dataset:** Reliance Industries Limited (RIL) Daily Stock Prices (1996 – 2020)

---

## 📌 Project Overview

This project implements and benchmarks three deep learning architectures—**Multilayer Perceptron (MLP)**, **Long Short-Term Memory (LSTM)**, and **Gated Recurrent Unit (GRU)**—against a **Persistence Baseline** for next-day stock price forecasting of Reliance Industries Ltd. (NSE: `RELIANCE`).

The pipeline enforces strict time-series integrity:
- **No data leakage:** MinMax scaler is fitted strictly on the chronological training split.
- **Sliding-window sequencing:** Uses a 60-day historical window to forecast day $t+1$.
- **Validation integrity:** Chronological validation split without cross-temporal shuffling.
- **Interactive Dashboard:** Complete Streamlit web application with interactive Plotly visualizations.

---

## 📊 Model Architecture Comparison & Results

Evaluated on the out-of-sample test set (last 20% chronological split):

| Model | MAE (₹) | RMSE (₹) | MAPE (%) | Direction Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** ($y_t = y_{t-1}$) | **17.32** | **34.30** | **1.40%** | — |
| **GRU** | **27.90** | **45.32** | **2.05%** | **48.99%** |
| **LSTM** | 172.33 | 235.36 | 11.90% | 48.43% |
| **MLP** | 256.82 | 293.13 | 19.52% | 48.03% |

### Key Insights:
1. **Best Deep Learning Model:** **GRU** significantly outperformed LSTM and MLP, capturing short-to-medium sequential dependencies with lower parameter overhead.
2. **Random Walk Reality:** The persistence baseline achieving the lowest raw MAE illustrates the Efficient Market Hypothesis (EMH)—daily price levels in equities behave closely to a random walk where today's price is the strongest point estimate of tomorrow's price.
3. **Direction Accuracy:** All neural network architectures hover around ~48–49% direction accuracy, confirming that raw price sequence regression alone without sentiment, macroeconomic, or order-book features cannot trivially predict directional momentum.

---

## 🗂️ Project Structure

```text
├── app.py                     # Interactive Streamlit dashboard
├── stock_prediction.ipynb     # Jupyter notebook (end-to-end training & analysis)
├── train_model.py             # Script to train MLP, LSTM, GRU & save artifacts
├── predict_next_day.py        # Next trading day forecast inference script
├── preprocess_data.py         # Data cleaning and feature preprocessing
├── stock_data.csv             # Historical daily stock data (RIL)
├── model_results.csv          # Evaluation metrics across all architectures
├── next_day_prediction.csv    # Next-day predicted close prices
├── requirements.txt           # Project dependencies
├── models/                    # Saved model artifacts (.keras & .pkl)
│   ├── gru_model.keras
│   ├── lstm_model.keras
│   ├── mlp_model.keras
│   └── price_scaler.pkl
└── plots/                     # Output comparison and training plots
    ├── actual_vs_predicted.png
    ├── metric_comparison.png
    ├── price_history.png
    └── training_loss.png
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yagneshatwork/Stock-Market-Prediction-RIL-MDM-FGAI-MiniProject-Sem5-2026.git
cd Stock-Market-Prediction-RIL-MDM-FGAI-MiniProject-Sem5-2026
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Interactive Web Dashboard
Launch the Streamlit app to explore data, test different models, and view next-day predictions:
```bash
streamlit run app.py
```

### Run Jupyter Notebook
Open and execute the step-by-step notebook:
```bash
jupyter notebook stock_prediction.ipynb
```

### Retrain Models via CLI
```bash
python train_model.py
```

### Generate Next-Day Prediction
```bash
python predict_next_day.py
```

---

## 🛡️ Time-Series Methodology & Data Leakage Prevention

1. **Chronological Splitting:** 80% train / 20% test partition without shuffle, respecting the arrow of time.
2. **Scaler Isolation:** `MinMaxScaler` is fit solely on `prices[:split_index]` and applied to transform both splits.
3. **No Lookahead Bias:** Each prediction $y_t$ is strictly conditioned on $[p_{t-60}, \dots, p_{t-1}]$.
4. **Validation Integrity:** Model training uses `validation_split=0.1` with `shuffle=False` and `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`.

---

## 📜 Disclaimer
This project is for educational and research purposes as part of academic coursework. It is **not** financial or trading advice.

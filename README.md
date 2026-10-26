# 📈 Reliance Industries (RIL) Stock Price Prediction

> **Mini Project — Semester 5 (2026)**  
> **Course / Track:** Financial Generative AI (FGAI) / Modern Data Mining (MDM)  
> **Asset:** Reliance Industries Limited (`RELIANCE.NS` on the National Stock Exchange of India)  
> **Historical Range:** January 2020 – October 2026 (1,687 daily trading sessions)  
> **Price Metric:** Auto-Adjusted Close (accounts for splits, dividends, and corporate actions via `yfinance`)

---

## 📌 Project Overview

This project provides an end-to-end deep learning framework and an interactive Streamlit dashboard for predicting next-day adjusted closing prices of Reliance Industries Limited (RIL). 

Three neural network architectures—**Multilayer Perceptron (MLP)**, **Long Short-Term Memory (LSTM)**, and **Gated Recurrent Unit (GRU)**—are benchmarked against a **Persistence Baseline** ($y_t = y_{t-1}$). The project is purposely contained in two primary artifacts for clarity and maintainability:
- [`stock_prediction.ipynb`](stock_prediction.ipynb) — Interactive exploratory data analysis, data preparation, window sequencing, model training, and performance evaluation.
- [`app.py`](app.py) — Production-grade interactive Streamlit dashboard with multi-tab comparative visualization, recursive multi-step forecasting, and historical backtesting.

---

## 🏆 Model Evaluation & Benchmark Results

Evaluated on the out-of-sample test split (the final 15% chronological data: 254 trading days from late 2024 through late 2026):

| Rank | Model Architecture | MAE (₹) | RMSE (₹) | Notes / Characteristics |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 | **Persistence Baseline** ($y_t = y_{t-1}$) | **₹13.83** | **₹18.36** | Best raw error; reflects random-walk behavior |
| 🥈 | **Gated Recurrent Unit (GRU)** | **₹22.04** | **₹27.95** | **Best Deep Learning Model**; lowest error & fastest convergence |
| 🥉 | **Long Short-Term Memory (LSTM)** | ₹40.14 | ₹49.06 | Strong temporal sequence learning, higher parameter count |
| 4 | **Multilayer Perceptron (MLP)** | ₹113.46 | ₹124.91 | Baseline feed-forward network without recurrence |

### 💡 Core Analytical Insights

1. **GRU Outperformed Recurrent Alternatives:** GRU achieved an MAE of **₹22.04** (compared to LSTM's ₹40.14 and MLP's ₹113.46). Its simplified gating structure (reset and update gates) mitigates vanishing gradients and avoids overfitting on shorter daily financial sequences.
2. **The Persistence Benchmark Challenge:** In daily equity price levels, tomorrow's price is heavily correlated with today's price ($y_t \approx y_{t-1}$). The persistence baseline's MAE of ₹13.83 confirms the Efficient Market Hypothesis (EMH)—raw sequential regression models track prices with a lag rather than discovering risk-free forward arbitrage.
3. **Multi-Step Compounding:** The app provides recursive multi-step forecasting (1 month to 2 years). As forecasts feed into subsequent steps, prediction drift and variance naturally expand, demonstrating why machine learning models should be treated as probabilistic indicators rather than guaranteed deterministic forecasts.

---

## 🛡️ Time-Series Integrity & Zero-Leakage Pipeline

Predicting financial time series requires strict methodological rigor to prevent data leakage and lookahead bias:

- **Chronological 70 / 15 / 15 Partition:**
  - **Training Set (70%):** Days $0 \to 1,179$ (1,180 trading days)
  - **Validation Set (15%):** Days $1,180 \to 1,432$ (253 trading days) for EarlyStopping (`patience=10`)
  - **Test Set (15%):** Days $1,433 \to 1,686$ (254 trading days) reserved strictly for unbiased out-of-sample evaluation
- **Strict Scaler Isolation:** The `MinMaxScaler(feature_range=(0, 1))` is fitted **strictly on the 70% training slice** (`prices[:train_end]`). Test and validation data are transformed using this frozen scaler.
- **Sliding-Window Feature Formulation:** A 60-day historical window ($t-60, \dots, t-1$) is used to predict day $t$. No future labels or future feature values enter the sequence generation.
- **Synchronized Metadata:** Split boundaries and window parameters are saved to `models/meta.json` so the Streamlit app dynamically evaluates on the exact identical test window used during training.

---

## 🖥️ Streamlit Interactive Dashboard (`app.py`)

The application offers an interactive financial interface divided into four functional views:

1. **📊 Overview:**
   - Real-time headline metrics: latest closing price, historical high/low, total volume, and date coverage.
   - Interactive Plotly chart with 20-day and 50-day Simple Moving Averages (SMA) and volume bars.
2. **🏆 Model Comparison:**
   - Summary metric table comparing MAE and RMSE across all models.
   - Comparative bar charts for quick visual assessment.
   - Out-of-sample actual vs. predicted plot with dynamic model checkboxes to inspect track fitting.
3. **🔮 Future Forecast:**
   - Multi-step recursive projection for up to 2 years (1M, 3M, 6M, 1Y, 2Y).
   - Dynamic uncertainty cone visualization depicting error widening over time.
   - Clear financial disclaimers and drift warnings.
4. **🔍 Historical Backtest:**
   - Select any historical year (2020 – 2026).
   - Compares actual prices against simulated next-day predictions and persistence baseline for that specific timeframe with on-the-fly calculated MAE/RMSE.

---

## 🗂️ Project Repository Structure

```text
├── stock_prediction.ipynb     # Primary Jupyter notebook (EDA, training, evaluation)
├── app.py                     # Streamlit web application
├── retrain.py                 # Reproducible headless retraining script
├── requirements.txt           # Project dependencies
├── stock_data.csv             # Cleaned daily OHLCV dataset (RELIANCE.NS, 2020–2026)
├── model_results.csv          # Benchmark evaluation metrics (MAE & RMSE)
├── models/                    # Serialized neural networks and artifacts
│   ├── gru_model.keras        # Saved GRU model
│   ├── lstm_model.keras       # Saved LSTM model
│   ├── mlp_model.keras        # Saved MLP model
│   ├── price_scaler.pkl       # Fitted MinMaxScaler (on training split only)
│   └── meta.json              # Dataset partition indices and window metadata
└── .gitignore                 # Excludes cache files and virtual environments
```

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yagneshatwork/Stock-Market-Prediction-RIL-MDM-FGAI-MiniProject-Sem5-2026.git
cd Stock-Market-Prediction-RIL-MDM-FGAI-MiniProject-Sem5-2026
```

### 2. Set Up a Virtual Environment (Recommended)
```powershell
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

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
*The web interface will open in your browser at `http://localhost:8501`.*

### 5. Run the Notebook or Retrain
- Open [`stock_prediction.ipynb`](stock_prediction.ipynb) in Jupyter / VS Code for step-by-step interactive analysis.
- Or execute the automated headless pipeline:
  ```bash
  python retrain.py
  ```

---

## 📦 Tech Stack & Dependencies

- **Data Acquisition:** `yfinance`
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & Deep Learning:** `tensorflow` (Keras 3), `scikit-learn`, `joblib`
- **Interactive UI & Visualization:** `streamlit`, `plotly`, `matplotlib`

---

## 📜 Academic Disclaimer

This project is developed exclusively for educational and academic evaluation as part of the Semester 5 Financial Generative AI / Modern Data Mining curriculum. Financial market predictions carry inherent uncertainties; this repository does **not** constitute financial, investment, or trading advice.

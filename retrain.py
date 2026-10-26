"""
Quick retrain script — runs the same logic as the notebook cells.
This is only needed once; after that, use the notebook for full analysis.
"""
import os, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib, numpy as np, pandas as pd, yfinance as yf, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

TICKER      = "RELIANCE.NS"
START_DATE  = "2020-01-01"
WINDOW_SIZE = 60
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
EPOCHS      = 100
BATCH_SIZE  = 32
DATA_FILE   = "stock_data.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ── Download ──────────────────────────────────────────────────────────────────
print(f"Downloading {TICKER} from {START_DATE} ...")
try:
    raw = yf.Ticker(TICKER).history(start=START_DATE, auto_adjust=True)
    if raw.empty:
        raise ValueError("Empty response from yfinance")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw.reset_index().rename(columns={"index": "Date"})
    keep = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in raw.columns]
    raw = raw[keep].copy()
    raw["Date"]  = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Close"] = pd.to_numeric(raw["Close"], errors="coerce")
    raw = raw.dropna(subset=["Date","Close"])
    raw = raw[raw["Close"] > 0].drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    raw["Date"] = raw["Date"].dt.strftime("%Y-%m-%d")
    raw.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(raw)} rows -> {DATA_FILE}")
except Exception as e:
    print(f"Download failed: {e}. Using existing {DATA_FILE}.")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
print(f"Loaded {len(df)} rows | {df['Date'].min().date()} -> {df['Date'].max().date()}")

prices = df["Close"].to_numpy().reshape(-1, 1)
N = len(prices)
train_end = int(N * TRAIN_RATIO)
val_end   = int(N * (TRAIN_RATIO + VAL_RATIO))

scaler = MinMaxScaler()
scaler.fit(prices[:train_end])
scaled = scaler.transform(prices)

# ── Sequences ─────────────────────────────────────────────────────────────────
X_train, y_train = [], []
X_val,   y_val   = [], []
X_test,  y_test  = [], []

for i in range(WINDOW_SIZE, N):
    seq = scaled[i - WINDOW_SIZE: i]
    tgt = scaled[i]
    if i < train_end:
        X_train.append(seq); y_train.append(tgt)
    elif i < val_end:
        X_val.append(seq);   y_val.append(tgt)
    else:
        X_test.append(seq);  y_test.append(tgt)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_val   = np.array(X_val,   dtype=np.float32)
y_val   = np.array(y_val,   dtype=np.float32)
X_test  = np.array(X_test,  dtype=np.float32)
y_test  = np.array(y_test,  dtype=np.float32)

print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

# ── Models ────────────────────────────────────────────────────────────────────
def build_mlp():
    return Sequential([Input(shape=(WINDOW_SIZE,1)), Flatten(),
                       Dense(64,activation="relu"), Dropout(0.2),
                       Dense(32,activation="relu"), Dense(1)], name="MLP")

def build_lstm():
    return Sequential([Input(shape=(WINDOW_SIZE,1)),
                       LSTM(50,return_sequences=True), LSTM(50), Dense(1)], name="LSTM")

def build_gru():
    return Sequential([Input(shape=(WINDOW_SIZE,1)),
                       GRU(50,return_sequences=True), GRU(50), Dense(1)], name="GRU")

arch = {"MLP": build_mlp, "LSTM": build_lstm, "GRU": build_gru}
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

results = []
for name, build_fn in arch.items():
    print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
    model = build_fn()
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val), shuffle=False,
              callbacks=[early_stop], verbose=1)
    p = model.predict(X_test, verbose=0)
    pred_prices   = scaler.inverse_transform(p).flatten()
    actual_prices = scaler.inverse_transform(y_test).flatten()
    mae  = mean_absolute_error(actual_prices, pred_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    print(f"{name} -> MAE: {mae:.2f}  RMSE: {rmse:.2f}")
    results.append({"Model": name, "MAE (Rs)": mae, "RMSE (Rs)": rmse})
    model.save(f"models/{name.lower()}_model.keras")
    print(f"Saved models/{name.lower()}_model.keras")

# Persistence baseline
prev_test    = prices[val_end - 1: N - 1].flatten()
actual_test  = prices[val_end:].flatten()
bl_mae  = mean_absolute_error(actual_test, prev_test)
bl_rmse = np.sqrt(mean_squared_error(actual_test, prev_test))
results.append({"Model": "Persistence Baseline", "MAE (Rs)": bl_mae, "RMSE (Rs)": bl_rmse})

results_df = pd.DataFrame(results).sort_values("MAE (Rs)").reset_index(drop=True)
print("\n" + "="*40)
print(results_df.round(2).to_string(index=False))

results_df.to_csv("model_results.csv", index=False)
joblib.dump(scaler, "models/price_scaler.pkl")

meta = {
    "train_end_idx": int(train_end),
    "val_end_idx":   int(val_end),
    "window_size":   WINDOW_SIZE,
    "train_ratio":   TRAIN_RATIO,
    "val_ratio":     VAL_RATIO,
    "price_type":    "adjusted_close (yfinance auto_adjust=True)",
    "trained_on":    str(pd.Timestamp.now().date()),
}
with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nAll done. Run: streamlit run app.py")

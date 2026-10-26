
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# 1. File paths
# -----------------------------
DATA_FILE = "stock_data.csv"
MODEL_FILE = os.path.join("models", "gru_model.keras")
SCALER_FILE = os.path.join("models", "price_scaler.pkl")
OUTPUT_FILE = "next_day_prediction.csv"

WINDOW_SIZE = 60

# -----------------------------
# 2. Check required files
# -----------------------------
for file_path in [DATA_FILE, MODEL_FILE, SCALER_FILE]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            "Check the filename and project folder."
        )

# -----------------------------
# 3. Load stock dataset
# -----------------------------
df = pd.read_csv(DATA_FILE)

# Handle CSV files with multi-level column names
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [
        " ".join(str(part) for part in col if str(part) != "nan").strip()
        for col in df.columns
    ]

# Find Date and Close columns
date_col = next(
    (col for col in df.columns
     if str(col).strip().lower() == "date"),
    None
)

close_col = next(
    (col for col in df.columns
     if str(col).strip().lower() == "close"
     or str(col).strip().lower().startswith("close ")),
    None
)

if date_col is None or close_col is None:
    raise ValueError(
        f"Could not find Date and Close columns.\n"
        f"Available columns: {list(df.columns)}"
    )

# Clean and sort data chronologically
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

df = df.dropna(subset=[date_col, close_col])
df = df[df[close_col] > 0]
df = df.drop_duplicates(subset=[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

if len(df) < WINDOW_SIZE:
    raise ValueError(
        f"At least {WINDOW_SIZE} valid closing prices are required."
    )

# -----------------------------
# 4. Load trained GRU and scaler
# -----------------------------
model = load_model(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

print("\nGRU model loaded successfully!")

# -----------------------------
# 5. Select latest 60 prices
# -----------------------------
latest_data = df.tail(WINDOW_SIZE).copy()

latest_prices = latest_data[close_col].to_numpy().reshape(-1, 1)

# Use the same scaler used during training
scaled_prices = scaler.transform(latest_prices)

# Shape: (1 sample, 60 time steps, 1 feature)
X_input = scaled_prices.reshape(1, WINDOW_SIZE, 1)

# -----------------------------
# 6. Predict next closing price
# -----------------------------
scaled_prediction = model.predict(X_input, verbose=0)

predicted_price = scaler.inverse_transform(
    scaled_prediction.reshape(-1, 1)
)[0, 0]

last_date = latest_data[date_col].iloc[-1]
last_close = float(latest_prices[-1, 0])

# This is the next calendar day, not necessarily
# the next stock-market trading day.
print("\n" + "=" * 45)
print("RELIANCE STOCK PRICE PREDICTION")
print("=" * 45)

print(f"Latest data date       : {last_date.date()}")
print(f"Latest closing price   : Rs. {last_close:.2f}")
print(f"Predicted next close   : Rs. {predicted_price:.2f}")

change = predicted_price - last_close
change_percent = (change / last_close) * 100

print(f"Predicted price change : Rs. {change:+.2f}")
print(f"Predicted change       : {change_percent:+.2f}%")

print("\nNote: This is a model estimate, not a guaranteed price.")

# -----------------------------
# 7. Save prediction to CSV
# -----------------------------
result = pd.DataFrame([{
    "Last_Data_Date": last_date.date(),
    "Last_Closing_Price": round(last_close, 2),
    "Predicted_Next_Closing_Price": round(float(predicted_price), 2),
    "Predicted_Change": round(float(change), 2),
    "Predicted_Change_Percent": round(float(change_percent), 2)
}])

result.to_csv(OUTPUT_FILE, index=False)

print(f"\nPrediction saved to: {OUTPUT_FILE}")
print("Prediction completed successfully!")

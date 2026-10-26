
import os
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, LSTM, GRU
)
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================
# 1. CONFIGURATION
# ==================================================
SEED = 42
WINDOW_SIZE = 60
TRAIN_RATIO = 0.80
EPOCHS = 100
BATCH_SIZE = 32

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ==================================================
# 2. LOAD DATASET
# ==================================================
df = pd.read_csv("stock_data.csv")

# Handle datasets with multi-level column headers
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [
        " ".join(
            str(part).strip()
            for part in col
            if str(part).strip() and not str(part).startswith("Unnamed")
        ).strip()
        for col in df.columns
    ]

df.columns = [str(col).strip() for col in df.columns]

print("\nAvailable columns:", df.columns.tolist())

# Find Date and Close columns, including names such as
# "Date RELIANCE.NS" or "Close RELIANCE.NS"
def find_column(columns, keyword):
    for col in columns:
        if col.lower() == keyword.lower():
            return col

    for col in columns:
        if col.lower().startswith(keyword.lower() + " "):
            return col

    return None


date_col = find_column(df.columns, "Date")
close_col = find_column(df.columns, "Close")

if date_col is None or close_col is None:
    raise ValueError(
        "Could not find Date and Close columns. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.rename(columns={
    date_col: "Date",
    close_col: "Close"
})

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df = df.dropna(subset=["Date", "Close"])
df = df[df["Close"] > 0]
df = df.drop_duplicates(subset=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if len(df) <= WINDOW_SIZE + 10:
    raise ValueError("Not enough valid data to train the models.")

print("\nDataset shape:", df.shape)
print("Start date:", df["Date"].min())
print("End date:", df["Date"].max())
print("Total records:", len(df))

# ==================================================
# 3. PREPARE CLOSING PRICES
# ==================================================
prices = df["Close"].to_numpy().reshape(-1, 1)

# Chronological split: first 80% for training,
# remaining 20% for testing
split_index = int(len(prices) * TRAIN_RATIO)

# Fit the scaler only on training prices
scaler = MinMaxScaler()
scaler.fit(prices[:split_index])

scaled_prices = scaler.transform(prices)

# ==================================================
# 4. CREATE SEQUENCES
# Use previous 60 trading days to predict next day
# ==================================================
X_train, y_train = [], []
X_test, y_test = [], []

for i in range(WINDOW_SIZE, len(scaled_prices)):
    sequence = scaled_prices[i - WINDOW_SIZE:i]
    target = scaled_prices[i]

    if i < split_index:
        X_train.append(sequence)
        y_train.append(target)
    else:
        X_test.append(sequence)
        y_test.append(target)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

print("\nTraining input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

# ==================================================
# 5. BUILD MLP MODEL
# ==================================================
def build_mlp():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    return model


# ==================================================
# 6. BUILD LSTM MODEL
# Similar to the LSTM practical you provided
# ==================================================
def build_lstm():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    return model


# ==================================================
# 7. BUILD GRU MODEL
# ==================================================
def build_gru():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        GRU(50, return_sequences=True),
        GRU(50),
        Dense(1)
    ])
    return model


models = {
    "MLP": build_mlp(),
    "LSTM": build_lstm(),
    "GRU": build_gru()
}

# ==================================================
# 8. TRAIN AND EVALUATE MODELS
# ==================================================
results = []
predictions_by_model = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"TRAINING {name} MODEL")
    print("=" * 50)

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1
    )

    # ----------------------------------------------
    # Plot training and validation loss
    # ----------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"{name} Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{name.lower()}_loss.png")
    plt.close()

    # ----------------------------------------------
    # Predict test prices
    # ----------------------------------------------
    predicted_scaled = model.predict(X_test, verbose=0)

    predicted_prices = scaler.inverse_transform(predicted_scaled)
    actual_prices = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(
        mean_squared_error(actual_prices, predicted_prices)
    )

    print(f"\n{name} MAE:  {mae:.2f}")
    print(f"{name} RMSE: {rmse:.2f}")

    predictions_by_model[name] = predicted_prices.flatten()

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    })

    # ----------------------------------------------
    # Save the complete model in Keras format
    # ----------------------------------------------
    model.save(f"models/{name.lower()}_model.keras")

    print(f"{name} model saved successfully!")


# ==================================================
# 9. PERSISTENCE BASELINE
# Predict next day's price as today's price
# ==================================================
baseline_predictions = prices[split_index - 1:-1].flatten()
actual_test_prices = prices[split_index:].flatten()

baseline_mae = mean_absolute_error(
    actual_test_prices, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(actual_test_prices, baseline_predictions)
)

results.append({
    "Model": "Persistence Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse
})

# ==================================================
# 10. COMPARE MODEL PERFORMANCE
# ==================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)
print(results_df.to_string(index=False))

results_df.to_csv("model_results.csv", index=False)

# ==================================================
# 11. PLOT ACTUAL VS PREDICTED PRICES
# ==================================================
plt.figure(figsize=(12, 6))

plt.plot(
    actual_test_prices,
    label="Actual Closing Price"
)

for name, predicted in predictions_by_model.items():
    plt.plot(predicted, label=f"{name} Prediction")

plt.xlabel("Test Day")
plt.ylabel("Closing Price")
plt.title("Reliance Stock: Actual vs Predicted Prices")
plt.legend()
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.close()

# ==================================================
# 12. SAVE SCALER SEPARATELY
# ==================================================
joblib.dump(scaler, "models/price_scaler.pkl")

print("\nTraining completed successfully!")
print("Trained models: models/ folder")
print("Model comparison: model_results.csv")
print("Graphs: plots/ folder")
print("Scaler: models/price_scaler.pkl")

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="Reliance Stock Predictor",
    page_icon="📈",
    layout="wide",
)

DATA_FILE = "stock_data.csv"
MODEL_DIR = "models"
SCALER_FILE = os.path.join(MODEL_DIR, "price_scaler.pkl")
WINDOW_SIZE = 60
TRAIN_RATIO = 0.80
MODEL_NAMES = ["MLP", "LSTM", "GRU"]

COLORS = {
    "Actual": "#7f7f7f",
    "MLP": "#EF553B",
    "LSTM": "#00CC96",
    "GRU": "#AB63FA",
    "Baseline": "#FFA15A",
}


# ==================================================
# DATA + MODEL LOADING
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0]
    df = df.drop_duplicates(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_resource
def load_models():
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_DIR, f"{name.lower()}_model.keras")
        models[name] = load_model(path)
    scaler = joblib.load(SCALER_FILE)
    return models, scaler


def predict_one(model, scaler, closes):
    """closes: array of the last WINDOW_SIZE closing prices."""
    scaled = scaler.transform(np.asarray(closes).reshape(-1, 1))
    x = scaled.reshape(1, WINDOW_SIZE, 1)
    pred = model.predict(x, verbose=0)
    return float(scaler.inverse_transform(pred.reshape(-1, 1))[0, 0])


@st.cache_data
def get_test_predictions():
    """Predictions of all three models on the 20% test split."""
    df = load_data()
    models, scaler = load_models()

    prices = df["Close"].to_numpy().reshape(-1, 1)
    split_index = int(len(prices) * TRAIN_RATIO)
    scaled = scaler.transform(prices)

    idx = range(split_index, len(prices))
    X_test = np.stack([scaled[i - WINDOW_SIZE:i] for i in idx]).astype(np.float32)

    out = pd.DataFrame({
        "Date": df["Date"].iloc[split_index:].to_numpy(),
        "Actual": prices[split_index:, 0],
        "Previous": prices[split_index - 1:-1, 0],
    })
    for name, model in models.items():
        p = model.predict(X_test, verbose=0, batch_size=256)
        out[name] = scaler.inverse_transform(p).flatten()
    out["Baseline"] = out["Previous"]
    return out


def compute_metrics(test_df):
    rows = []
    for name in MODEL_NAMES + ["Baseline"]:
        err = test_df[name] - test_df["Actual"]
        mae = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())
        mape = (err.abs() / test_df["Actual"]).mean() * 100
        actual_dir = np.sign(test_df["Actual"] - test_df["Previous"])
        pred_dir = np.sign(test_df[name] - test_df["Previous"])
        dir_acc = (actual_dir == pred_dir).mean() * 100 if name != "Baseline" else np.nan
        rows.append({
            "Model": name if name != "Baseline" else "Persistence Baseline",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Direction Accuracy (%)": dir_acc,
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


# ==================================================
# GUARD: files must exist
# ==================================================
missing = [p for p in [DATA_FILE, SCALER_FILE] +
           [os.path.join(MODEL_DIR, f"{n.lower()}_model.keras") for n in MODEL_NAMES]
           if not os.path.exists(p)]
if missing:
    st.error("Some required files are missing:\n\n" + "\n".join(f"- `{m}`" for m in missing))
    st.info("Run `train_model.py` (or the notebook) first to create the models, "
            "then start the app from the same folder.")
    st.stop()

df = load_data()
models, scaler = load_models()

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.title("📈 Reliance Predictor")
    st.caption("MLP vs LSTM vs GRU for next-day closing price")
    st.divider()
    st.markdown("**Dataset**")
    st.write(f"Records: **{len(df):,}**")
    st.write(f"From: **{df['Date'].min().date()}**")
    st.write(f"To: **{df['Date'].max().date()}**")
    st.divider()
    st.markdown("**Setup**")
    st.write(f"Window: **{WINDOW_SIZE}** days")
    st.write(f"Train / Test: **{int(TRAIN_RATIO*100)}% / {int((1-TRAIN_RATIO)*100)}%**")
    st.divider()
    st.caption("⚠️ Educational project. Not financial advice.")

st.title("Reliance Industries — Stock Price Prediction")
st.caption("Comparing a feed-forward network (MLP) with two recurrent networks (LSTM, GRU).")

tab_overview, tab_compare, tab_predict, tab_about = st.tabs(
    ["📊 Market Overview", "🏆 Model Comparison", "🔮 Predict", "ℹ️ About"]
)

# ==================================================
# TAB 1: MARKET OVERVIEW
# ==================================================
with tab_overview:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    day_change = last["Close"] - prev["Close"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"₹{last['Close']:,.2f}",
              f"{day_change:+.2f} ({day_change / prev['Close'] * 100:+.2f}%)")
    c2.metric("All-time high (in data)", f"₹{df['Close'].max():,.2f}")
    c3.metric("All-time low (in data)", f"₹{df['Close'].min():,.2f}")
    c4.metric("Trading days", f"{len(df):,}")

    col_a, col_b, col_c = st.columns([2, 2, 3])
    start = col_a.date_input("From", df["Date"].min().date(),
                             min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
    end = col_b.date_input("To", df["Date"].max().date(),
                           min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
    mas = col_c.multiselect("Moving averages", [20, 50, 200], default=[50, 200])
    chart_type = st.radio("Chart type", ["Line", "Candlestick"], horizontal=True)

    view = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)].copy()

    if view.empty:
        st.warning("No data in the selected range.")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.03)
        if chart_type == "Candlestick" and {"Open", "High", "Low"}.issubset(view.columns):
            fig.add_trace(go.Candlestick(x=view["Date"], open=view["Open"], high=view["High"],
                                         low=view["Low"], close=view["Close"], name="Price"),
                          row=1, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False)
        else:
            fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], name="Close",
                                     line=dict(color="#636EFA", width=1.5)), row=1, col=1)
        for m in mas:
            ma = df["Close"].rolling(m).mean().loc[view.index]
            fig.add_trace(go.Scatter(x=view["Date"], y=ma, name=f"MA{m}",
                                     line=dict(width=1.2, dash="dot")), row=1, col=1)
        if "Volume" in view.columns:
            fig.add_trace(go.Bar(x=view["Date"], y=view["Volume"], name="Volume",
                                 marker_color="rgba(120,120,120,0.5)"), row=2, col=1)
        fig.update_layout(height=600, hovermode="x unified",
                          margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.02))
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        st.plotly_chart(fig)

        with st.expander("Daily returns distribution"):
            ret = view["Close"].pct_change().dropna() * 100
            hist = go.Figure(go.Histogram(x=ret, nbinsx=80, marker_color="#636EFA"))
            hist.update_layout(height=300, xaxis_title="Daily return (%)",
                               yaxis_title="Days", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(hist)

# ==================================================
# TAB 2: MODEL COMPARISON
# ==================================================
with tab_compare:
    with st.spinner("Running models on the test set..."):
        test_df = get_test_predictions()
        metrics = compute_metrics(test_df)

    best = metrics.iloc[0]
    st.subheader("Test-set performance")
    st.caption(f"{len(test_df):,} unseen trading days "
               f"({pd.to_datetime(test_df['Date'].iloc[0]).date()} → "
               f"{pd.to_datetime(test_df['Date'].iloc[-1]).date()})")

    nn_metrics = metrics[metrics["Model"].isin(MODEL_NAMES)]
    best_nn = nn_metrics.iloc[0]
    cols = st.columns(3)
    for col, name in zip(cols, MODEL_NAMES):
        row = metrics[metrics["Model"] == name].iloc[0]
        col.metric(f"{name} — MAE", f"₹{row['MAE']:.2f}",
                   "🥇 best model" if name == best_nn["Model"] else None,
                   delta_color="off")

    if best["Model"] == "Persistence Baseline":
        st.info("The simple **persistence baseline** (tomorrow = today) scored the lowest MAE. "
                "This is common for stock prices: they behave close to a random walk, so "
                "beating 'yesterday's price' on raw error is genuinely hard.")
    else:
        st.success(f"**{best['Model']}** beat the persistence baseline on MAE.")

    st.dataframe(
        metrics.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}",
                              "MAPE (%)": "{:.2f}", "Direction Accuracy (%)": "{:.1f}"},
                             na_rep="—")
        .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#14532d55")
        .highlight_max(subset=["Direction Accuracy (%)"], color="#14532d55"),
        hide_index=True,
    )

    m1, m2 = st.columns(2)
    for col, metric in zip([m1, m2], ["MAE", "RMSE"]):
        bar = go.Figure(go.Bar(
            x=metrics["Model"], y=metrics[metric],
            marker_color=[COLORS.get(m, COLORS["Baseline"]) for m in metrics["Model"]],
            text=[f"{v:.2f}" for v in metrics[metric]], textposition="outside"))
        bar.update_layout(title=f"{metric} (lower is better)", height=350,
                          margin=dict(l=10, r=10, t=40, b=10), yaxis_title="₹")
        col.plotly_chart(bar)

    st.subheader("Actual vs predicted")
    left, right = st.columns([3, 2])
    shown = left.multiselect("Models to show", MODEL_NAMES + ["Baseline"], default=MODEL_NAMES)
    n_days = right.slider("Last N test days", 30, len(test_df), min(200, len(test_df)), step=10)

    zoom = test_df.tail(n_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zoom["Date"], y=zoom["Actual"], name="Actual",
                             line=dict(color=COLORS["Actual"], width=2.5)))
    for name in shown:
        fig.add_trace(go.Scatter(x=zoom["Date"], y=zoom[name], name=name,
                                 line=dict(color=COLORS[name], width=1.5)))
    fig.update_layout(height=450, hovermode="x unified", yaxis_title="Close (₹)",
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig)

    with st.expander("Prediction error over time"):
        err_fig = go.Figure()
        for name in MODEL_NAMES:
            err_fig.add_trace(go.Scatter(x=zoom["Date"], y=zoom[name] - zoom["Actual"],
                                         name=name, line=dict(color=COLORS[name], width=1)))
        err_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        err_fig.update_layout(height=320, yaxis_title="Predicted − Actual (₹)",
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(err_fig)

# ==================================================
# TAB 3: PREDICT
# ==================================================
with tab_predict:
    mode = st.radio("Mode", ["Next trading day (latest data)", "Backtest on a past date"],
                    horizontal=True)

    if mode.startswith("Next"):
        window = df.tail(WINDOW_SIZE)
        closes = window["Close"].to_numpy()
        last_date = window["Date"].iloc[-1]
        last_close = float(closes[-1])
        target_label = "next trading day"
        actual_next = None
    else:
        split_index = int(len(df) * TRAIN_RATIO)
        test_dates = df["Date"].iloc[split_index:-1]
        pick = st.date_input("Pick a date (the model sees the 60 days up to and including it)",
                             value=test_dates.iloc[-1].date(),
                             min_value=test_dates.iloc[0].date(),
                             max_value=test_dates.iloc[-1].date())
        pos = int(df.index[df["Date"].dt.date <= pick][-1])
        window = df.iloc[pos - WINDOW_SIZE + 1: pos + 1]
        closes = window["Close"].to_numpy()
        last_date = window["Date"].iloc[-1]
        last_close = float(closes[-1])
        actual_next = float(df["Close"].iloc[pos + 1])
        target_label = f"{df['Date'].iloc[pos + 1].date()}"

    preds = {name: predict_one(model, scaler, closes) for name, model in models.items()}
    ensemble = float(np.mean(list(preds.values())))

    st.markdown(f"Based on data up to **{last_date.date()}**, close ₹{last_close:,.2f}. "
                f"Predicting **{target_label}**.")

    cols = st.columns(4 if actual_next is None else 5)
    for col, (name, p) in zip(cols, preds.items()):
        diff = p - last_close
        col.metric(name, f"₹{p:,.2f}", f"{diff:+.2f} ({diff / last_close * 100:+.2f}%)")
    diff = ensemble - last_close
    cols[3].metric("Average of 3", f"₹{ensemble:,.2f}", f"{diff:+.2f} ({diff / last_close * 100:+.2f}%)")
    if actual_next is not None:
        d = actual_next - last_close
        cols[4].metric("Actual", f"₹{actual_next:,.2f}", f"{d:+.2f} ({d / last_close * 100:+.2f}%)")

    # Chart: last 60 days + predicted points
    x_next = pd.bdate_range(last_date, periods=2)[-1] if actual_next is None \
        else df["Date"].iloc[pos + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["Date"], y=window["Close"], name="Last 60 closes",
                             line=dict(color=COLORS["Actual"], width=2)))
    for name, p in preds.items():
        fig.add_trace(go.Scatter(x=[last_date, x_next], y=[last_close, p], name=name,
                                 mode="lines+markers", line=dict(color=COLORS[name], dash="dot"),
                                 marker=dict(size=[0, 10])))
    if actual_next is not None:
        fig.add_trace(go.Scatter(x=[x_next], y=[actual_next], name="Actual next close",
                                 mode="markers",
                                 marker=dict(size=13, symbol="star", color="gold",
                                             line=dict(width=1, color="black"))))
    fig.update_layout(height=450, hovermode="x unified", yaxis_title="Close (₹)",
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig)

    if actual_next is not None:
        err_df = pd.DataFrame({
            "Model": list(preds.keys()),
            "Predicted (₹)": list(preds.values()),
            "Error (₹)": [p - actual_next for p in preds.values()],
            "Abs error (₹)": [abs(p - actual_next) for p in preds.values()],
        }).sort_values("Abs error (₹)")
        st.dataframe(err_df.style.format({"Predicted (₹)": "{:.2f}", "Error (₹)": "{:+.2f}",
                                          "Abs error (₹)": "{:.2f}"}), hide_index=True)

    if actual_next is None:
        st.caption(f"Note: the dataset ends on {last_date.date()}, so 'next trading day' is "
                   "relative to that date, not today's date.")
    st.warning("These are model estimates from past prices only. They are not financial advice.")

# ==================================================
# TAB 4: ABOUT
# ==================================================
with tab_about:
    st.subheader("Project overview")
    st.write(
        "This project predicts Reliance Industries' next-day closing price from the "
        f"previous {WINDOW_SIZE} closing prices, and compares three neural architectures."
    )
    a, b, c = st.columns(3)
    a.markdown("**MLP**\n\nFlatten → Dense(64) → Dropout(0.2) → Dense(32) → Dense(1). "
               "Treats the 60 days as 60 independent inputs, with no memory of order.")
    b.markdown("**LSTM**\n\n2 stacked LSTM(50) layers → Dense(1). Gated memory cells that "
               "can carry information across the whole window.")
    c.markdown("**GRU**\n\n2 stacked GRU(50) layers → Dense(1). A lighter gated unit "
               "with fewer parameters than LSTM.")
    st.markdown("""
**Method**
1. Clean data, sort by date, keep `Close`.
2. Chronological 80/20 split (no shuffling, to avoid look-ahead leakage).
3. `MinMaxScaler` fitted on training data only.
4. Sliding windows of 60 days → next-day close.
5. Train with Adam + MSE + early stopping.
6. Evaluate with MAE, RMSE, MAPE and direction accuracy against a **persistence baseline**.

**Why a baseline?** Stock prices are close to a random walk, so "tomorrow = today" is a
strong benchmark. Comparing against it keeps the evaluation honest.
""")

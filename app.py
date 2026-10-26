"""
Reliance Industries (RIL) — Stock Price Prediction Dashboard
Streamlit app | Mini-Project Sem 5 2026

Sections
--------
A. Overview    – latest price, date-range chart, key stats
B. Model Comparison – MAE/RMSE table for MLP, LSTM, GRU, Persistence Baseline
C. Future Forecast  – recursive multi-step forecast (1 month to 2 years)
D. Historical Backtest – model vs actual for 2022, 2023, 2024 etc.

Price type: adjusted closes (yfinance auto_adjust=True) — same as used during training.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RIL Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE   = "stock_data.csv"
MODEL_DIR   = "models"
SCALER_FILE = os.path.join(MODEL_DIR, "price_scaler.pkl")
META_FILE   = os.path.join(MODEL_DIR, "meta.json")
MODEL_NAMES = ["MLP", "LSTM", "GRU"]

COLORS = {
    "Actual":               "#374151",
    "MLP":                  "#EF553B",
    "LSTM":                 "#00CC96",
    "GRU":                  "#AB63FA",
    "Persistence Baseline": "#FFA15A",
    "Forecast":             "#2563EB",
}

HORIZON_DAYS = {
    "1 Month":  21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year":   252,
    "2 Years":  504,
}

# ─────────────────────────────────────────────────────────────────────────────
# FILE GUARDS
# ─────────────────────────────────────────────────────────────────────────────
required = (
    [DATA_FILE, SCALER_FILE]
    + [os.path.join(MODEL_DIR, f"{n.lower()}_model.keras") for n in MODEL_NAMES]
)
missing = [p for p in required if not os.path.exists(p)]
if missing:
    st.error("**Required files not found:**\n\n" + "\n".join(f"- `{p}`" for p in missing))
    st.info(
        "Run the notebook `stock_prediction.ipynb` from start to finish "
        "to download data, train the models and save all artifacts. "
        "Then relaunch: `streamlit run app.py`"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0]
    df = df.drop_duplicates(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_models() -> tuple:
    """Load all three Keras models and the scaler.  Returns (models_dict, scaler)."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.models import load_model as _load

    mdls = {}
    errs = []
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_DIR, f"{name.lower()}_model.keras")
        try:
            mdls[name] = _load(path)
        except Exception as e:
            errs.append(f"{name}: {e}")

    scaler = joblib.load(SCALER_FILE)

    if errs:
        st.warning("Some models could not be loaded:\n" + "\n".join(errs))

    return mdls, scaler


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    if os.path.exists(META_FILE):
        with open(META_FILE) as f:
            return json.load(f)
    # Fall back to a default (for models trained before meta.json was added)
    return {"window_size": 60, "train_ratio": 0.80, "val_ratio": 0.00}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_predict(model, x: np.ndarray) -> np.ndarray:
    """Run model.predict and return a (n, 1) float32 array.
    Fixes 'Cannot take the length of shape with unknown rank' that occurs
    when the input dtype or rank is wrong for the loaded Keras 3 model."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, :, np.newaxis]          # (batch, 60) → (batch, 60, 1)
    elif x.ndim == 1:
        x = x[np.newaxis, :, np.newaxis]  # (60,) → (1, 60, 1)
    out = model.predict(x, verbose=0, batch_size=256)
    return out.reshape(-1, 1)


def make_sequences(scaled: np.ndarray, window: int) -> np.ndarray:
    """Return X array of shape (n_samples, window, 1)."""
    X = []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window: i])
    return np.array(X, dtype=np.float32)


def recursive_forecast(model, scaler, seed_closes: np.ndarray, n_steps: int,
                       window: int) -> np.ndarray:
    """Recursive multi-step forecast.

    seed_closes : last `window` actual closing prices (₹), shape (window,)
    Returns     : forecast prices (₹), shape (n_steps,)

    Limitation: each predicted value is fed back into the window, so errors
    accumulate. Uncertainty grows substantially with the horizon.
    """
    buf = scaler.transform(seed_closes.reshape(-1, 1)).flatten().tolist()
    forecasts = []
    for _ in range(n_steps):
        x = np.array(buf[-window:], dtype=np.float32).reshape(1, window, 1)
        y_scaled = model.predict(x, verbose=0)[0, 0]
        # Clamp to [0, 1] so the inverse_transform stays in the training range
        y_scaled = float(np.clip(y_scaled, 0.0, 1.0))
        forecasts.append(y_scaled)
        buf.append(y_scaled)
    forecasts_arr = np.array(forecasts, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(forecasts_arr).flatten()


@st.cache_data(show_spinner=False)
def get_test_predictions(_hash: str) -> pd.DataFrame:
    """Run all models on the test split. _hash ensures cache refreshes if data changes."""
    df      = load_data()
    meta    = load_meta()
    models, scaler = load_models()

    window    = int(meta.get("window_size", 60))
    prices    = df["Close"].to_numpy().reshape(-1, 1)
    N         = len(prices)

    # Determine test start from meta or fall back to 80/20 split
    if "val_end_idx" in meta:
        test_start = int(meta["val_end_idx"])
    else:
        test_start = int(N * float(meta.get("train_ratio", 0.80)))

    scaled = scaler.transform(prices)

    # Build test sequences
    idx    = list(range(test_start, N))
    X_test = np.stack([scaled[i - window: i] for i in idx]).astype(np.float32)

    out = pd.DataFrame({
        "Date":     df["Date"].iloc[test_start:].to_numpy(),
        "Actual":   prices[test_start:, 0],
        "Previous": prices[test_start - 1: N - 1, 0],
    })

    for name, model in models.items():
        p = safe_predict(model, X_test)
        out[name] = scaler.inverse_transform(p).flatten()

    out["Persistence Baseline"] = out["Previous"]
    return out


def compute_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES + ["Persistence Baseline"]:
        if name not in test_df.columns:
            continue
        err  = test_df[name] - test_df["Actual"]
        mae  = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())
        rows.append({"Model": name, "MAE (₹)": mae, "RMSE (₹)": rmse})
    return pd.DataFrame(rows).sort_values("MAE (₹)").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and models …"):
    df            = load_data()
    models, scaler = load_models()
    meta          = load_meta()

WINDOW_SIZE = int(meta.get("window_size", 60))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 RIL Predictor")
    st.caption("Reliance Industries Ltd — NSE: RELIANCE")
    st.divider()

    st.markdown("**Dataset**")
    st.write(f"Rows: **{len(df):,}**")
    st.write(f"From: **{df['Date'].min().date()}**")
    st.write(f"To:   **{df['Date'].max().date()}**")
    st.write(f"Prices: **Adjusted close**")
    st.divider()

    st.markdown("**Model setup**")
    st.write(f"Window: **{WINDOW_SIZE} trading days**")
    tr = int(meta.get("train_ratio", 0.80) * 100)
    vr = int(meta.get("val_ratio",   0.15) * 100)
    te = 100 - tr - vr
    st.write(f"Split: **{tr}% / {vr}% / {te}%** (train/val/test)")
    st.divider()

    st.caption("⚠️ Educational project. Not financial advice.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE TITLE
# ─────────────────────────────────────────────────────────────────────────────
st.title("Reliance Industries (RIL) — Stock Price Prediction")
st.caption("MLP · LSTM · GRU compared against a Persistence Baseline")

tab_overview, tab_compare, tab_forecast, tab_backtest = st.tabs([
    "📊  Overview",
    "🏆  Model Comparison",
    "🔮  Future Forecast",
    "🔍  Historical Backtest",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB A: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    chg   = last["Close"] - prev["Close"]
    chgp  = chg / prev["Close"] * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close",
              f"₹{last['Close']:,.2f}",
              f"{chg:+.2f}  ({chgp:+.2f}%)")
    c2.metric("Date of Latest Data", str(last["Date"].date()))
    c3.metric("All-time High (in data)", f"₹{df['Close'].max():,.2f}")
    c4.metric("All-time Low (in data)",  f"₹{df['Close'].min():,.2f}")

    st.caption(
        "ℹ️ The dataset ends on the date shown above — it is **not** necessarily today's date. "
        "Prices are adjusted closes from Yahoo Finance (`RELIANCE.NS`, `auto_adjust=True`)."
    )
    st.divider()

    # ── Date-range selector ──────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 2, 3])
    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()
    start = col_a.date_input("From", date_min,
                             min_value=date_min, max_value=date_max)
    end   = col_b.date_input("To",   date_max,
                             min_value=date_min, max_value=date_max)
    mas   = col_c.multiselect("Moving averages", [20, 50, 200], default=[50, 200])

    view = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)].copy()

    if view.empty:
        st.warning("No data in the selected range.")
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22], vertical_spacing=0.04,
        )
        fig.add_trace(go.Scatter(
            x=view["Date"], y=view["Close"], name="Close (adjusted)",
            line=dict(color="#2563EB", width=1.5),
        ), row=1, col=1)

        for m in mas:
            ma_vals = df["Close"].rolling(m).mean().reindex(view.index)
            fig.add_trace(go.Scatter(
                x=view["Date"], y=ma_vals, name=f"MA {m}",
                line=dict(width=1.1, dash="dot"),
            ), row=1, col=1)

        if "Volume" in view.columns:
            fig.add_trace(go.Bar(
                x=view["Date"], y=view["Volume"], name="Volume",
                marker_color="rgba(100,100,150,0.4)",
            ), row=2, col=1)

        fig.update_layout(
            height=520, hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.04),
            xaxis_rangeslider_visible=False,
        )
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume",    row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB B: MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
with tab_compare:
    # Use a hash of the data file's mtime to bust the cache when data changes
    data_hash = str(os.path.getmtime(DATA_FILE))

    with st.spinner("Running models on the test split …"):
        test_df = get_test_predictions(data_hash)
        metrics = compute_metrics(test_df)

    test_start_date = pd.Timestamp(test_df["Date"].iloc[0]).date()
    test_end_date   = pd.Timestamp(test_df["Date"].iloc[-1]).date()
    st.subheader("Test-set Performance")
    st.caption(
        f"{len(test_df):,} held-out trading days · "
        f"{test_start_date} → {test_end_date} · "
        "Models were never trained on this period."
    )

    # ── MAE metric cards ────────────────────────────────────────────────────
    best_row    = metrics.iloc[0]
    nn_metrics  = metrics[metrics["Model"].isin(MODEL_NAMES)]
    best_nn_row = nn_metrics.iloc[0] if not nn_metrics.empty else None

    cols = st.columns(len(MODEL_NAMES))
    for col, name in zip(cols, MODEL_NAMES):
        row = metrics[metrics["Model"] == name]
        if row.empty:
            col.metric(name, "N/A")
            continue
        row = row.iloc[0]
        is_best = best_nn_row is not None and name == best_nn_row["Model"]
        col.metric(
            f"{name}",
            f"MAE ₹{row['MAE (₹)']:.2f}",
            "🥇 best neural network" if is_best else f"RMSE ₹{row['RMSE (₹)']:.2f}",
            delta_color="off",
        )

    if best_row["Model"] == "Persistence Baseline":
        st.info(
            "The **Persistence Baseline** (tomorrow ≈ today) achieved the lowest MAE. "
            "This is common for stock prices, which behave close to a random walk. "
            "Beating this baseline on raw error is genuinely difficult."
        )
    else:
        st.success(f"**{best_row['Model']}** achieved the lowest MAE on the test set.")

    # ── Results table ────────────────────────────────────────────────────────
    styled = (
        metrics.style
        .format({"MAE (₹)": "{:.2f}", "RMSE (₹)": "{:.2f}"})
        .highlight_min(subset=["MAE (₹)", "RMSE (₹)"], color="#d1fae5")
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)

    # ── Bar charts ───────────────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    for col, metric in zip([m1, m2], ["MAE (₹)", "RMSE (₹)"]):
        bar = go.Figure(go.Bar(
            x=metrics["Model"],
            y=metrics[metric],
            marker_color=[COLORS.get(m, "#9CA3AF") for m in metrics["Model"]],
            text=[f"₹{v:.1f}" for v in metrics[metric]],
            textposition="outside",
        ))
        bar.update_layout(
            title=f"{metric} — lower is better", height=340,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="₹",
        )
        col.plotly_chart(bar, use_container_width=True)

    # ── Actual vs predicted chart ────────────────────────────────────────────
    st.divider()
    st.subheader("Actual vs Predicted Prices on the Test Set")

    shown  = st.multiselect(
        "Show models", MODEL_NAMES + ["Persistence Baseline"],
        default=MODEL_NAMES,
        key="compare_models",
    )
    n_days = st.slider(
        "Show last N test days",
        min_value=30,
        max_value=len(test_df),
        value=min(200, len(test_df)),
        step=10,
    )

    zoom = test_df.tail(n_days)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=zoom["Date"], y=zoom["Actual"], name="Actual",
        line=dict(color=COLORS["Actual"], width=2.5),
    ))
    for name in shown:
        if name in zoom.columns:
            fig2.add_trace(go.Scatter(
                x=zoom["Date"], y=zoom[name], name=name,
                line=dict(color=COLORS.get(name, "#9CA3AF"), width=1.5),
            ))

    fig2.update_layout(
        height=420, hovermode="x unified",
        yaxis_title="Close (₹)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "⚠️ A model that tracks the actual curve visually may still have a high MAE. "
        "Always refer to the metrics table above for an honest comparison."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB C: FUTURE FORECAST
# ═════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.subheader("Recursive Future Forecast")
    st.caption(
        "The model predicts one step ahead, then feeds the prediction back into the "
        "window to forecast the next step, and so on. **Errors compound with the horizon** — "
        "treat longer forecasts as a rough trend indicator, not a reliable price target."
    )

    col_a, col_b = st.columns(2)
    horizon_label = col_a.selectbox(
        "Forecast Horizon", list(HORIZON_DAYS.keys()), index=2,
    )
    model_name = col_b.selectbox(
        "Model", MODEL_NAMES, index=MODEL_NAMES.index("GRU"),
    )
    n_steps = HORIZON_DAYS[horizon_label]

    if model_name not in models:
        st.error(f"Model '{model_name}' not loaded.")
    else:
        model    = models[model_name]
        seed     = df["Close"].tail(WINDOW_SIZE).to_numpy()       # last 60 actual closes
        last_date = df["Date"].iloc[-1]
        last_close = float(seed[-1])

        with st.spinner(f"Computing {n_steps}-step recursive forecast …"):
            forecast_prices = recursive_forecast(
                model, scaler, seed, n_steps, WINDOW_SIZE
            )

        # Validate outputs
        if not np.all(np.isfinite(forecast_prices)):
            st.error("Forecast contains non-finite values. The model may not be compatible "
                     "with the current scaler range (new data exceeds training range). "
                     "Please retrain the models by running the notebook.")
        else:
            # Generate future trading dates (using business-day approximation)
            future_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1), periods=n_steps
            )

            # Show summary stats first
            fc_end = forecast_prices[-1]
            chg_fc = fc_end - last_close
            chgp_fc = chg_fc / last_close * 100

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Last Actual Close",   f"₹{last_close:,.2f}", str(last_date.date()))
            s2.metric("Forecast Start",      str(future_dates[0].date()))
            s3.metric("Forecast End",        str(future_dates[-1].date()))
            s4.metric(f"Forecast @ {horizon_label}",
                      f"₹{fc_end:,.2f}",
                      f"{chg_fc:+.2f}  ({chgp_fc:+.2f}%)")

            # Weekly / monthly aggregation for long horizons
            if n_steps >= 126:
                fc_series = pd.Series(forecast_prices, index=future_dates)
                agg_label = "Monthly" if n_steps >= 252 else "Weekly"
                fc_agg    = fc_series.resample("ME" if n_steps >= 252 else "W").last()
                with st.expander(f"{agg_label} Forecast Summary"):
                    agg_df = pd.DataFrame({
                        "Period End":    fc_agg.index.date,
                        f"{model_name} Forecast (₹)": fc_agg.values.round(2),
                    })
                    st.dataframe(agg_df, hide_index=True, use_container_width=True)

            # Chart: recent actual + forecast
            n_history = min(120, len(df))
            hist_df   = df.tail(n_history)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=hist_df["Date"], y=hist_df["Close"],
                name="Actual (historical)",
                line=dict(color=COLORS["Actual"], width=2),
            ))
            fig3.add_trace(go.Scatter(
                x=[last_date, future_dates[0]],
                y=[last_close, forecast_prices[0]],
                name="Bridge",
                line=dict(color=COLORS["Forecast"], width=1.5, dash="dot"),
                showlegend=False,
            ))
            fig3.add_trace(go.Scatter(
                x=list(future_dates),
                y=forecast_prices,
                name=f"{model_name} Forecast ({horizon_label})",
                line=dict(color=COLORS["Forecast"], width=2, dash="dash"),
                mode="lines",
            ))
            # Boundary line
            fig3.add_vline(
                x=last_date.isoformat(), line_dash="dash",
                line_color="gray", line_width=1,
                annotation_text="Last actual ↓", annotation_position="top",
            )
            fig3.update_layout(
                height=450,
                hovermode="x unified",
                yaxis_title="Close (₹)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.warning(
                "These values are **model estimates based on historical patterns only**. "
                "Recursive forecasting error accumulates — a 2-year forecast is highly "
                "speculative. This is **not financial advice**."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB D: HISTORICAL BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("Historical Backtest — Actual vs Predicted")
    st.caption(
        "Select a historical year to see what the model would have predicted "
        "for that period, compared to what actually happened. "
        "These are **genuine out-of-sample predictions** — the model is trained on data "
        "up to the start of the selected period and then predicts day-by-day forward."
    )

    # Determine available years from the data
    avail_years = sorted(df["Date"].dt.year.unique().tolist())
    # Only offer years where there is at least WINDOW_SIZE days before them
    first_possible_year = df["Date"].iloc[WINDOW_SIZE]["Date"].year \
        if hasattr(df["Date"].iloc[WINDOW_SIZE], "year") \
        else int(str(df["Date"].iloc[WINDOW_SIZE])[:4])
    avail_years = [y for y in avail_years if y >= first_possible_year and y < df["Date"].max().year]

    col_a, col_b = st.columns(2)
    bt_year       = col_a.selectbox("Year to evaluate", avail_years[::-1], index=0)
    bt_model_name = col_b.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("GRU"),
                                    key="bt_model")

    if bt_model_name not in models:
        st.error(f"Model '{bt_model_name}' not loaded.")
    else:
        bt_model = models[bt_model_name]

        # Select rows for the chosen year
        year_mask  = df["Date"].dt.year == bt_year
        year_df    = df[year_mask].copy()

        if len(year_df) < 10:
            st.warning(f"Not enough data for {bt_year}.")
        else:
            # For each day in the year, build input from the 60 actual days before it
            # This is a proper out-of-sample evaluation using only data up to that point.
            bt_actuals = []
            bt_preds   = []
            bt_dates   = []
            bt_prev    = []

            all_prices_arr = df["Close"].to_numpy().reshape(-1, 1)
            all_dates_arr  = df["Date"].to_numpy()

            for idx_in_df in df.index[year_mask]:
                if idx_in_df < WINDOW_SIZE:
                    continue  # not enough history before this row
                window_prices = all_prices_arr[idx_in_df - WINDOW_SIZE: idx_in_df]
                actual_price  = float(all_prices_arr[idx_in_df, 0])
                prev_price    = float(all_prices_arr[idx_in_df - 1, 0])

                # Scale using the pre-trained scaler (fitted on training data)
                window_scaled = scaler.transform(window_prices)
                x = window_scaled.reshape(1, WINDOW_SIZE, 1).astype(np.float32)
                y_scaled = bt_model.predict(x, verbose=0)[0, 0]
                y_pred   = float(scaler.inverse_transform([[y_scaled]])[0, 0])

                bt_actuals.append(actual_price)
                bt_preds.append(y_pred)
                bt_dates.append(df["Date"].iloc[idx_in_df])
                bt_prev.append(prev_price)

            if len(bt_actuals) == 0:
                st.warning("Could not generate predictions. Check that there is enough data before this year.")
            else:
                bt_actuals = np.array(bt_actuals)
                bt_preds   = np.array(bt_preds)
                bt_prev    = np.array(bt_prev)

                bt_mae  = float(np.mean(np.abs(bt_actuals - bt_preds)))
                bt_rmse = float(np.sqrt(np.mean((bt_actuals - bt_preds) ** 2)))
                bl_mae  = float(np.mean(np.abs(bt_actuals - bt_prev)))
                bl_rmse = float(np.sqrt(np.mean((bt_actuals - bt_prev) ** 2)))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{bt_model_name} MAE",  f"₹{bt_mae:.2f}")
                m2.metric(f"{bt_model_name} RMSE", f"₹{bt_rmse:.2f}")
                m3.metric("Baseline MAE",           f"₹{bl_mae:.2f}")
                m4.metric("Baseline RMSE",          f"₹{bl_rmse:.2f}")

                if bt_mae < bl_mae:
                    st.success(f"{bt_model_name} outperformed the persistence baseline in {bt_year} on MAE.")
                else:
                    st.info(
                        f"The persistence baseline beat {bt_model_name} in {bt_year} on MAE. "
                        "This is common — stock prices are difficult to beat on raw error."
                    )

                # Chart
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=bt_dates, y=bt_actuals, name="Actual",
                    line=dict(color=COLORS["Actual"], width=2),
                ))
                fig4.add_trace(go.Scatter(
                    x=bt_dates, y=bt_preds, name=f"{bt_model_name} Predicted",
                    line=dict(color=COLORS.get(bt_model_name, "#2563EB"), width=1.5,
                              dash="dot"),
                ))
                fig4.add_trace(go.Scatter(
                    x=bt_dates, y=bt_prev, name="Persistence Baseline",
                    line=dict(color=COLORS["Persistence Baseline"], width=1, dash="dash"),
                ))
                fig4.update_layout(
                    title=f"{bt_year} Backtest — Actual vs {bt_model_name} vs Baseline",
                    height=420, hovermode="x unified",
                    yaxis_title="Close (₹)",
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(fig4, use_container_width=True)

                st.caption(
                    "⚠️ The model was trained on data up to its training cutoff. "
                    "If the selected year falls inside the training period, these are "
                    "in-sample predictions and will appear more accurate than out-of-sample. "
                    "The test-set period in the **Model Comparison** tab is the honest "
                    "out-of-sample evaluation."
                )

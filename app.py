"""
Polished Professional Stock Predictor — Delta-based + Multi-feature + Residual-Informed Multi-day Forecast

Polish highlights:
- TF log silencing to reduce noisy TensorArray warnings in logs.
- Model + scaler caching (models/{TICKER}_model.h5, models/{TICKER}_scalers.pkl).
- Sidebar "Retrain model" toggle to force retraining.
- Shorter default training settings for faster deployments.
- Same feature set & MC-dropout + residual sampling retained.
"""

import os
import time
import logging
import math
import pickle
from typing import Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error



import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.set_page_config(page_title="Professional Stock Predictor", layout="wide")

st.title("Stock Predictor")
st.info("TensorFlow training removed for cloud demo. Showing stock data and analysis.")




import warnings
warnings.filterwarnings("ignore")

# ---------------- Determinism ----------------
SEED = 42
np.random.seed(SEED)


# ---------------- Streamlit config ----------------


# ---------------- Styling ----------------
st.markdown("""
<style>
    .main-header { font-size: 2.0rem; font-weight: 700; color: #1f77b4; text-align: left; margin-bottom: 1rem; }
    .subtle { color: #9aa5b1; font-size:0.9rem; margin-bottom: 1rem; }
    .metric-card { background: rgba(255,255,255,0.02); padding: 0.6rem; border-radius: 8px; color: #ddd; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- Indicators ----------------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=1).mean()
    ma_down = down.rolling(window=window, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd.fillna(0.0)

# ---------------- Data fetch & feature creation ----------------
@st.cache_data(ttl=300)
def fetch_stock_data_with_features(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch price history and compute RSI/MACD.
    Returns empty DataFrame on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df['RSI_14'] = compute_rsi(df['Close'], window=14)
        df['MACD'] = compute_macd(df['Close'], fast=12, slow=26, sig=9)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD']].ffill().dropna().reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()

def create_window_features_multi(df: pd.DataFrame, lookback: int = 30, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    if feature_cols is None:
        feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI_14', 'MACD']
    prices = df['Close'].values
    F = df[feature_cols].values
    X, y = [], []
    for i in range(lookback, len(F)):
        win = F[i - lookback:i].astype(float)
        next_close = prices[i]
        last_in_window_close = prices[i - 1]
        delta = next_close - last_in_window_close
        X.append(win)
        y.append(delta)
    return np.array(X), np.array(y)

# ---------------- Model builder ----------------
def build_model_generic(model_type='lstm', input_shape=(30,7), units1=64, units2=32, dropout_rate=0.12, lr=5e-4):
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(units1, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units2))
        model.add(Dropout(dropout_rate))
    else:
        model.add(GRU(units1, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units2))
        model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # predict delta
    model.compile(optimizer=Adam(learning_rate=lr), loss='huber', metrics=['mae'])
    return model

# ---------------- Predictor class ----------------
class StockPredictor:
    def __init__(self, model_type='lstm', lookback=30, n_features=7, units1=64, units2=32, dropout_rate=0.12, lr=5e-4):
        self.model_type = model_type
        self.lookback = lookback
        self.n_features = n_features
        self.model = None
        self.scaler = None
        self.y_scaler = None
        self.is_trained = False
        self.units1 = units1
        self.units2 = units2
        self.dropout_rate = dropout_rate
        self.lr = lr

    def build_model(self, input_shape):
        return build_model_generic(self.model_type, input_shape, self.units1, self.units2, self.dropout_rate, self.lr)

    def train(self, X: np.ndarray, y: np.ndarray, validation_split=0.15, epochs=60, verbose=0):
        n_samples, seq_len, n_features = X.shape
        self.scaler = MinMaxScaler()
        X_flat = X.reshape(-1, n_features)
        X_scaled_flat = self.scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_features)

        self.y_scaler = MinMaxScaler()
        y_2d = y.reshape(-1, 1)
        y_scaled = self.y_scaler.fit_transform(y_2d).reshape(-1)

        self.model = self.build_model((seq_len, n_features))
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
        rl = ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.3, min_lr=1e-6, verbose=0)

        history = self.model.fit(X_scaled, y_scaled, validation_split=validation_split,
                                 epochs=epochs, batch_size=32,
                                 callbacks=[es, rl], verbose=verbose)
        self.is_trained = True
        return history

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled_flat = self.scaler.transform(X_flat)
        return X_scaled_flat.reshape(n_samples, seq_len, n_features)

    def predict_delta(self, X: np.ndarray, mc_dropout=False, n_iter=20) -> np.ndarray:
        Xs = self._scale_X(X)
        if mc_dropout:
            preds = []
            for _ in range(n_iter):
                p_scaled = self.model(Xs, training=True).numpy().reshape(-1, 1)
                p_real = self.y_scaler.inverse_transform(p_scaled).reshape(-1)
                preds.append(p_real)
            return np.array(preds)
        else:
            p_scaled = self.model.predict(Xs, verbose=0).reshape(-1, 1)
            p_real = self.y_scaler.inverse_transform(p_scaled).reshape(-1)
            return p_real

    def predict_price_from_window(self, X_window: np.ndarray, mc_dropout=False, n_iter=20) -> np.ndarray:
        deltas = self.predict_delta(X_window, mc_dropout=mc_dropout, n_iter=n_iter)
        last_closes = X_window[:, -1, 0]
        if mc_dropout:
            return deltas + last_closes[None, :]
        else:
            return deltas + last_closes

# ---------------- Save / load helpers ----------------
def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def model_paths_for(ticker: str):
    model_file = os.path.join("models", f"{ticker}_model.h5")
    scaler_file = os.path.join("models", f"{ticker}_scalers.pkl")
    return model_file, scaler_file

def save_model_and_scalers(ticker: str, predictor: StockPredictor):
    model_file, scaler_file = model_paths_for(ticker)
    try:
        predictor.model.save(model_file)
        with open(scaler_file, "wb") as f:
            pickle.dump({"scaler": predictor.scaler, "y_scaler": predictor.y_scaler}, f)
        logger.info(f"Saved model and scalers for {ticker} -> {model_file}, {scaler_file}")
    except Exception as e:
        logger.warning(f"Failed to save model/scalers for {ticker}: {e}")

def load_model_and_scalers(ticker: str):
    model_file, scaler_file = model_paths_for(ticker)
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        try:
            m = load_model(model_file, compile=False)
            with open(scaler_file, "rb") as f:
                d = pickle.load(f)
            scaler = d.get("scaler")
            y_scaler = d.get("y_scaler")
            pred = StockPredictor()
            pred.model = m
            pred.scaler = scaler
            pred.y_scaler = y_scaler
            pred.is_trained = True
            logger.info(f"Loaded model for {ticker} from {model_file}")
            return pred
        except Exception as e:
            logger.warning(f"Failed to load model/scalers for {ticker}: {e}")
            return None
    return None

# ---------------- Plot helpers ----------------
def create_prediction_figure(dates, actual, predicted, lower=None, upper=None, naive=None, title="Price Prediction with Confidence Bands"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Model Forecast', line=dict(dash='dash', width=2)))
    if naive is not None:
        fig.add_trace(go.Scatter(x=dates, y=naive, mode='lines', name='Naive Baseline', line=dict(dash='dot')))
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(x=dates, y=upper, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=dates, y=lower, line=dict(width=0), fill='tonexty', fillcolor='rgba(0,123,255,0.12)',
                                 name='Forecast Uncertainty (10–90%)', hoverinfo='skip'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', height=520,
                      legend=dict(orientation='v', x=0.92, y=0.95))
    return fig

def plot_training_history_safe(history):
    try:
        hist = history.history if hasattr(history, 'history') else {}
    except Exception:
        hist = {}
    epochs = list(range(1, len(hist.get('loss', [])) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss (Train/Val)', 'MAE (Train/Val)'))
    if 'loss' in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist['loss'], name='Train Loss'), row=1, col=1)
    if 'val_loss' in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist['val_loss'], name='Val Loss'), row=1, col=1)
    if 'mae' in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist['mae'], name='Train MAE'), row=1, col=2)
    if 'val_mae' in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist['val_mae'], name='Val MAE'), row=1, col=2)
    fig.update_layout(height=380)
    return fig

# ---------------- App UI ----------------
def main():
    ensure_dirs()
    st.markdown('<h1 class="main-header">📊 Professional Stock Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Delta-based forecasting with MC-dropout & residual bootstrap uncertainty. Model caching enabled.</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        ticker = st.selectbox("Choose Stock", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], index=0)
        model_type = st.radio("Model", ["LSTM", "GRU"])
        lookback = st.slider("Lookback Days", 10, 60, 30)
        horizon = st.slider("Prediction Horizon (days)", 1, 10, 5)
        period = st.selectbox("History Period", ["1y", "2y", "3y", "5y"], index=1)
        with st.expander("Advanced Settings"):
            epochs = st.slider("Epochs", 10, 200, 60, step=10)
            mc_iters = st.slider("MC Dropout Iterations (CI)", 10, 200, 50, step=5)
            retrain = st.checkbox("Retrain model (ignore cached)", value=False)
            show_naive = st.checkbox("Show Naive Baseline (Yesterday's close)", value=True)
        analyze = st.button("🚀 Analyze", use_container_width=True)

    if not analyze:
        st.info("Adjust settings on the left and click Analyze.")
        return

    # Fetch data + features (use local cache fallback)
    with st.spinner("Fetching data and computing features..."):
        df = fetch_stock_data_with_features(ticker, period)

    if df.empty:
        st.error("No data retrieved. Make sure you have local CSV cached or try 'Fetch latest data' in the repo.")
        return

    # Prepare windows
    feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI_14', 'MACD']
    X, y = create_window_features_multi(df, lookback=lookback, feature_cols=feature_cols)
    if len(X) < 50:
        st.error("Insufficient data for training. Increase period or reduce lookback.")
        return

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build or load predictor
    predictor = None
    if not retrain:
        predictor = load_model_and_scalers(ticker)
        if predictor:
            # ensure architecture matches selected model_type (best-effort)
            predictor.model_type = model_type.lower()
            st.success(f"Loaded cached model for {ticker}. (Toggle 'Retrain model' to retrain.)")
    if predictor is None:
        predictor = StockPredictor(model_type=model_type.lower(), lookback=lookback, n_features=len(feature_cols),
                                   units1=64, units2=32, dropout_rate=0.12, lr=5e-4)

    # Train only if model not trained / retrain requested
    history = None
    trained_here = False
    if not predictor.is_trained or retrain:
        with st.spinner("Training model (this can take a little while)..."):
            history = predictor.train(X_train, y_train, validation_split=0.15, epochs=epochs, verbose=0)
            trained_here = True
            save_model_and_scalers(ticker, predictor)
            st.success("Training finished and model cached.")
    else:
        st.info("Using cached model — skipping training.")

    # If history is None but we loaded model, create empty history for plotting logic
    if history is None:
        class DummyHistory:
            history = {}
        history = DummyHistory()

    # Test predictions (deterministic)
    preds_test_prices = predictor.predict_price_from_window(X_test, mc_dropout=False)

    # Align test dates & true prices
    pred_start_idx = lookback + split
    dates_test = df['Date'].iloc[pred_start_idx: pred_start_idx + len(y_test)].reset_index(drop=True)
    true_prices = df['Close'].iloc[pred_start_idx: pred_start_idx + len(y_test)].values

    # Residuals & metrics
    residuals = true_prices - preds_test_prices
    resid_std = float(np.std(residuals)) if len(residuals) > 0 else 0.0
    resid_bias = float(np.mean(residuals)) if len(residuals) > 0 else 0.0

    # MC-dropout CI on test
    try:
        mc_preds_test = predictor.predict_price_from_window(X_test, mc_dropout=True, n_iter=mc_iters)
        mc_lower = np.percentile(mc_preds_test, 10, axis=0)
        mc_upper = np.percentile(mc_preds_test, 90, axis=0)
        lower_test = np.minimum(mc_lower, preds_test_prices - resid_std)
        upper_test = np.maximum(mc_upper, preds_test_prices + resid_std)
    except Exception:
        lower_test = preds_test_prices - resid_std
        upper_test = preds_test_prices + resid_std

    # Naive baseline
    naive_test = None
    if show_naive:
        naive_idx = np.arange(pred_start_idx - 1, pred_start_idx - 1 + len(true_prices))
        if np.all(naive_idx >= 0) and len(naive_idx) == len(true_prices):
            naive_test = df['Close'].iloc[naive_idx].values

    # Plot test
    st.subheader("📈 Test Prediction Chart")
    fig_test = create_prediction_figure(dates_test, true_prices, preds_test_prices,
                                        lower=lower_test, upper=upper_test, naive=naive_test,
                                        title="Price Prediction with Confidence Bands")
    st.plotly_chart(fig_test, use_container_width=True)

    # Training diagnostics (if we trained here)
    st.subheader("📊 Training Diagnostics")
    if hasattr(history, "history") and history.history:
        st.plotly_chart(plot_training_history_safe(history), use_container_width=True)
    else:
        st.info("No training history available (model loaded from cache).")

    # Performance summary
    try:
        model_mae = float(mean_absolute_error(true_prices, preds_test_prices))
        model_rmse = float(math.sqrt(mean_squared_error(true_prices, preds_test_prices)))
    except Exception:
        model_mae = None
        model_rmse = None

    perf = {"Model MAE": model_mae, "Model RMSE": model_rmse}
    if naive_test is not None and model_mae is not None:
        perf["Naive MAE"] = float(mean_absolute_error(true_prices, naive_test))
        perf["Naive RMSE"] = float(math.sqrt(mean_squared_error(true_prices, naive_test)))

    st.subheader("🧾 Test Performance Summary")
    st.dataframe(pd.Series(perf).to_frame("Value").round(6))

    # -------- Multi-day iterative forecast --------
    st.subheader(f"🔮 Multi-day Forecast ({horizon} day{'s' if horizon>1 else ''})")
    last_window = X[-1:].copy()
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq='B')

    # Deterministic iterative forecast with bias correction
    future_preds = []
    cur_window = last_window.copy()
    for step in range(horizon):
        pred_price = predictor.predict_price_from_window(cur_window, mc_dropout=False)[0]
        pred_price = pred_price + resid_bias
        future_preds.append(pred_price)
        cur = cur_window.reshape(cur_window.shape[1], cur_window.shape[2]).copy()
        cur = np.roll(cur, -1, axis=0)
        cur[-1, 0] = pred_price
        cur_window = cur.reshape(1, cur.shape[0], cur.shape[1])
    future_preds = np.array(future_preds)

    # MC rollouts with residual sampling (bootstrap)
    mc_rollouts = np.zeros((mc_iters, horizon))
    for i in range(mc_iters):
        cur_window = last_window.copy()
        for step in range(horizon):
            try:
                p_price = predictor.predict_price_from_window(cur_window, mc_dropout=True, n_iter=1)[0]
            except Exception:
                p_price = predictor.predict_price_from_window(cur_window, mc_dropout=False)[0]
            noise = float(np.random.choice(residuals)) if len(residuals) > 0 else 0.0
            p_price_noisy = p_price + resid_bias + noise
            mc_rollouts[i, step] = p_price_noisy
            cur = cur_window.reshape(cur_window.shape[1], cur_window.shape[2]).copy()
            cur = np.roll(cur, -1, axis=0)
            cur[-1, 0] = p_price_noisy
            cur_window = cur.reshape(1, cur.shape[0], cur.shape[1])
    mc_lower_future = np.percentile(mc_rollouts, 10, axis=0)
    mc_upper_future = np.percentile(mc_rollouts, 90, axis=0)
    lower_future = np.minimum(mc_lower_future, future_preds - resid_std)
    upper_future = np.maximum(mc_upper_future, future_preds + resid_std)

    # Plot future
    future_fig = go.Figure()
    future_fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', name='Model Forecast'))
    future_fig.add_trace(go.Scatter(x=future_dates, y=upper_future, line=dict(width=0), showlegend=False, hoverinfo='skip'))
    future_fig.add_trace(go.Scatter(x=future_dates, y=lower_future, line=dict(width=0), fill='tonexty',
                                    fillcolor='rgba(0,123,255,0.12)', name='Forecast Uncertainty', hoverinfo='skip'))
    future_fig.update_layout(title='Multi-day Forecast (next business days)', xaxis_title='Date', yaxis_title='Predicted Price', height=360)
    st.plotly_chart(future_fig, use_container_width=True)

    # Future table
    try:
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted": np.round(future_preds, 6),
            "Lower (10%)": np.round(lower_future, 6),
            "Upper (90%)": np.round(upper_future, 6)
        })
        st.dataframe(future_df)
    except Exception:
        st.info("Future forecast table unavailable.")

    st.success("Done — model ready. Use 'Retrain model' to re-fit and update cached model.")

if __name__ == "__main__":
    main()

 # py code beginning

"""
Retail Market Dashboard with LSTM Forecasting
===========================================
• Streamlit dashboard that pulls retail‑related time‑series from FRED
• Resamples **all** series to monthly frequency for consistency
• Normalises features with per‑column Min‑Max scaling for stable LSTM training
• Trains a multi‑feature LSTM with user‑configurable parameters
• Saves model in the app directory (safe for Streamlit Cloud) and offers a download button
• Plots recent actuals + multi‑step forecast
"""

import os
import time
from datetime import date
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from fredapi import Fred
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- Streamlit Page Config ---------
st.set_page_config(page_title="Retail Market Dashboard (USA)", layout="wide")
sidebar = st.sidebar
sidebar.title("🔧 Configuration")

# ---------- Date Range Controls -----------
if "start_date" not in st.session_state:
    st.session_state["start_date"] = date(2015, 1, 1)
if "end_date" not in st.session_state:
    st.session_state["end_date"] = date.today()

start_date = sidebar.date_input("Start date", value=st.session_state["start_date"], key="start_date")
end_date = sidebar.date_input("End date", value=st.session_state["end_date"], key="end_date")

# ---------- FRED API Key ------------------
fred_api_key = sidebar.text_input("FRED API key", type="password")

# ---------- Series Dictionary --------------
SERIES_CODES: Dict[str, str] = {
    "Total Retail Sales (RSAFS, Monthly)": "RSAFS",
    "E-commerce Sales (ECOMSA, Quarterly)": "ECOMSA",
    "Retail Employment (CEU4200000001, Monthly)": "CEU4200000001",
    "Clothing & Accessories Sales": "MRTSSM448USS",
    "Consumer Sentiment (UMCSENT)": "UMCSENT",
    "PCE - Personal Consumption": "PCE",
    "Personal Savings Rate": "PSAVERT",
}

selected_series = sidebar.multiselect("Select series", list(SERIES_CODES.keys()), default=list(SERIES_CODES.keys()))

# ---------- Helper: load FRED series ------
@st.cache_data(show_spinner=False)
def load_series(api_key: str, code: str, start: date, end: date) -> pd.Series:
    fred = Fred(api_key=api_key)
    s = fred.get_series(code, observation_start=start, observation_end=end)
    s.name = code
    return s

# ---------- LSTM Model --------------------
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out).squeeze()

# ---------- Main App ----------------------
if fred_api_key:
    series_list = []
    for label in selected_series:
        code = SERIES_CODES[label]
        try:
            s = load_series(fred_api_key, code, start_date, end_date)
            # Resample quarterly to monthly (and everything else to monthly)
            s = s.resample("M").ffill()
            # Friendly name for ECOMSA
            if code == "ECOMSA":
                s.name = "E-commerce Sales (ECOMSA, Monthly)"
            else:
                s.name = label
            series_list.append(s)
        except Exception as e:
            st.error(f"❌ Failed to download {label} ({code}): {e}")

    if series_list:
        # Build unified monthly DataFrame
        idx = pd.date_range(start=pd.to_datetime(start_date).replace(day=1),
                            end=pd.to_datetime(end_date).replace(day=1),
                            freq="M")
        data = pd.concat(series_list, axis=1).sort_index().reindex(idx)
        data_filled = data.ffill()

        # ---------- KPI Section ------------
        st.title("📊 Retail Market Dashboard (USA)")
        st.write(f"Data range: {data_filled.index.min().date()} → {data_filled.index.max().date()} (monthly)")

        latest = data_filled.iloc[-1]
        first = data_filled.iloc[0]
        kpi_cols = st.columns(len(latest))
        for i, col in enumerate(latest.index):
            delta = "N/A"
            if pd.notna(latest[col]) and pd.notna(first[col]) and first[col] != 0:
                delta = f"{(latest[col]-first[col])*100/first[col]:+.2f}%"
            kpi_cols[i].metric(col, f"{latest[col]:,.2f}" if pd.notna(latest[col]) else "N/A", delta)

        # ---------- Combined Chart ---------
        st.subheader("📈 Combined Time Series Chart")
        df_long = data_filled.reset_index().melt(id_vars="index", var_name="Series", value_name="Value").dropna()
        df_long.rename(columns={"index": "Date"}, inplace=True)
        st.plotly_chart(px.line(df_long, x="Date", y="Value", color="Series"), use_container_width=True)

        # ---------- Correlation Heatmap ----
        st.subheader("🔍 Correlation (Pct Change)")
        corr = data_filled.pct_change().dropna().corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto"), use_container_width=True)

        # ---------- LSTM Forecasting -------
        st.header("🧠 LSTM Forecasting")
        if len(data_filled.dropna()) < 24:
            st.info("Need at least 24 monthly records for forecasting.")
        else:
            target_col = st.selectbox("Target series", data_filled.columns)
            seq_len = st.slider("Sequence length (months)", min_value=3, max_value=12, value=6)
            horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=12, value=3)
            epochs = st.slider("Epochs", 10, 200, 50, 10)
            lr = st.number_input("Learning rate", value=0.001, format="%.4f")

            # --- Normalise each column individually ---
            scalers: Dict[str, MinMaxScaler] = {}
            df_scaled = pd.DataFrame(index=data_filled.index)
            for col in data_filled.columns:
                sc = MinMaxScaler()
                df_scaled[col] = sc.fit_transform(data_filled[[col]])
                scalers[col] = sc

            df_train = df_scaled.dropna()

            # Build sequences
            X_seqs, y_vals = [], []
            for i in range(seq_len, len(df_train) - horizon + 1):
                X_seqs.append(df_train.iloc[i - seq_len:i].values)
                y_vals.append(df_train.iloc[i + horizon - 1][target_col])
            X = torch.tensor(np.array(X_seqs), dtype=torch.float32)
            y = torch.tensor(np.array(y_vals), dtype=torch.float32)

            dataset = TensorDataset(X, y)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = LSTMRegressor(n_features=df_train.shape[1]).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            crit = nn.MSELoss()

            if st.button("🚀 Train LSTM"):
                prog = st.progress(0.0)
                for epoch in range(epochs):
                    model.train()
                    total_loss = 0.0
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optim.zero_grad()
                        preds = model(xb)
                        loss = crit(preds, yb)
                        loss.backward()
                        optim.step()
                        total_loss += loss.item() * xb.size(0)
                    total_loss /= len(train_loader.dataset)
                    prog.progress((epoch + 1) / epochs, text=f"Epoch {epoch+1}/{epochs} | loss {total_loss:.4f}")
                prog.empty()
                st.success(f"Training finished (loss {total_loss:.4f})")

                # Save model locally for Streamlit Cloud session
                model_path = "lstm_model.pth"
                torch.save(model.state_dict(), model_path)
                with open(model_path, "rb") as f:
                    st.download_button("⬇️ Download trained model", f, file_name="lstm_model.pth")

                # ----- Forecast -----
                model.eval()
                recent_seq = torch.tensor(df_train.tail(seq_len).values, dtype=torch.float32).unsqueeze(0).to(device)
                forecasts = []
                with torch.no_grad():
                    seq_arr = recent_seq.clone()
                    for _ in range(horizon):
                        pred = model(seq_arr).item()
                        forecasts.append(pred)
                        # update sequence
                        next_step = seq_arr.cpu().numpy().squeeze(0)[-1].copy()
                        target_idx = df_train.columns.get_loc(target_col)
                        next_step[target_idx] = pred
                        seq_arr = torch.tensor(np.vstack([seq_arr.cpu().numpy().squeeze(0)[1:], next_step]), dtype=torch.float32).unsqueeze(0).to(device)

                # Inverse transform forecast values
                inv_forecasts = [scalers[target_col].inverse_transform([[f]]).flatten()[0] for f in forecasts]

                last_date = df_train.index[-1]
                forecast_idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=horizon, freq="M")
                forecast_df = pd.DataFrame({target_col: inv_forecasts}, index=forecast_idx)

                st.metric("Next predicted value", f"{inv_forecasts[0]:,.2f}")

                # Plot
                plot_actual = data_filled[target_col].tail(24)
                fig = px.line(title=f"{target_col}: last 24 months & forecast")
                fig.add_scatter(x=plot_actual.index, y=plot_actual.values, mode="lines", name="Actual")
                fig.add_scatter(x=forecast_df.index, y=forecast_df[target_col], mode="lines+markers", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("🔑 Enter your FRED API key in the sidebar to begin.")

# ---------- Footer -----------------------
sidebar.markdown("---")
sidebar.markdown(
    """ℹ️ **Tips**  
• All series resampled monthly & forward‑filled  
• Quarterly series (e.g., ECOMSA) auto‑converted to monthly  
• Minimum 24 records required for LSTM training""")


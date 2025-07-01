 # py code beginning

import streamlit as st
from datetime import date
from typing import Dict

import pandas as pd
import numpy as np
import plotly.express as px
from fredapi import Fred

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Page Config ---
st.set_page_config(page_title="Retail Market Dashboard (USA)", layout="wide")
sidebar = st.sidebar
sidebar.title("🔧 Configuration")

# Initialize session_state dates if not exists
if "start_date" not in st.session_state:
    st.session_state["start_date"] = date(2015, 1, 1)
if "end_date" not in st.session_state:
    st.session_state["end_date"] = date.today()

# --- Date Inputs ---------------------------------------------------
start_date = sidebar.date_input("Start date", value=date(2015, 1, 1), key="start_date")
end_date = sidebar.date_input("End date", value=date.today(), key="end_date")

# --- Other sidebar controls ----------------------------------------
fred_api_key = sidebar.text_input("FRED API key", type="password")

SERIES_CODES: Dict[str, str] = {
    "Total Retail Sales (RSAFS, Monthly)": "RSAFS",
    "E-commerce Sales (ECOMSA, Quarterly)": "ECOMSA",
    "Retail Employment (CEU4200000001, Monthly)": "CEU4200000001",
    "Clothing & Accessories Sales": "MRTSSM448USS",
    "Consumer Sentiment (UM)": "UMCSENT",
    "PCE - Personal Consumption": "PCE",
    "Personal Savings Rate": "PSAVERT",
}

selected_series = sidebar.multiselect(
    "Select series to display", list(SERIES_CODES.keys()), default=list(SERIES_CODES.keys())
)

# --- Helper to load series from FRED -------------------------------
@st.cache_data(show_spinner=False)
def load_series(api_key: str, code: str, start: date, end: date) -> pd.Series:
    fred = Fred(api_key=api_key)
    s = fred.get_series(code, observation_start=start, observation_end=end)
    s.name = code
    return s

# --- LSTM model definition ---
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

# --------------------------- Main ----------------------------------
if fred_api_key:
    series_list = []

    for label in selected_series:
        code = SERIES_CODES[label]
        try:
            s = load_series(fred_api_key, code, st.session_state["start_date"], st.session_state["end_date"])

            # Resample quarterly to monthly for ECOMSA
            if code == "ECOMSA":
                s = s.resample("M").ffill()
                s.name = "E-commerce Sales (ECOMSA, Monthly)"

            # Resample all series to monthly frequency and forward fill
            else:
                s = s.resample("M").ffill()
            series_list.append(s)
        except Exception as e:
            st.error(f"❌ Failed to download {label} ({code}): {e}")

    if series_list:
        # Build dataframe on monthly frequency
        full_idx = pd.date_range(
            start=pd.to_datetime(st.session_state["start_date"]).replace(day=1),
            end=pd.to_datetime(st.session_state["end_date"]).replace(day=1),
            freq="M",
        )
        data = pd.concat(series_list, axis=1).sort_index()
        data = data.reindex(full_idx)
        data_filled = data.ffill()

        # Header
        st.title("📊 Retail Market Dashboard (USA)")
        st.write(f"FRED data from {st.session_state['start_date']} to {st.session_state['end_date']} (Monthly)")

        # KPI Cards
        valid = data_filled.dropna(how="all")
        if not valid.empty:
            latest = valid.iloc[-1]
            first = valid.iloc[0]
            kpi_cols = st.columns(len(latest))
            for idx, (label, val) in enumerate(latest.items()):
                first_val = first.get(label, None)
                delta_str = "N/A"
                if pd.notna(val) and pd.notna(first_val) and isinstance(first_val, (int, float)) and first_val != 0:
                    delta_pct = ((val - first_val) / first_val) * 100
                    delta_str = f"{delta_pct:+.2f}% vs first"
                val_str = f"{val:,.2f}" if pd.notna(val) else "N/A"
                kpi_cols[idx].metric(label, val_str, delta_str)

        # Combined Chart
        st.subheader("📈 Combined Time Series Chart")
        plot_columns = list(data.columns)
        if "E-commerce Sales (ECOMSA, Monthly)" in plot_columns and "E-commerce Sales (ECOMSA, Quarterly)" in plot_columns:
            plot_columns.remove("E-commerce Sales (ECOMSA, Quarterly)")
        if plot_columns:
            df_long = (
                data[plot_columns]
                .reset_index()
                .melt(id_vars="index", var_name="Series", value_name="Value")
                .dropna()
                .rename(columns={"index": "Date"})
            )
            fig_combined = px.line(df_long, x="Date", y="Value", color="Series", title="📈 Combined Time Series Data")
            fig_combined.update_layout(xaxis=dict(range=[st.session_state["start_date"], st.session_state["end_date"]]))
            st.plotly_chart(fig_combined, use_container_width=True)

        # Correlation heatmap
        st.subheader("🔍 Correlation (Pct Change)")
        corr = data.pct_change().dropna().corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto"), use_container_width=True)

        # --- LSTM Forecasting ---
        st.subheader("🧠 LSTM Forecasting")

        if len(data_filled.dropna()) < 24:
            st.info("📉 Need at least 24 months of complete data for LSTM forecasting.")
        else:
            target_col = st.selectbox("🎯 Target series", data_filled.columns, key="lstm_target")
            seq_len = st.slider("Sequence length (months)", 3, 12, 6, key="lstm_seq")
            horizon = st.slider("Forecast horizon (months ahead)", 1, 12, 3, key="lstm_horizon")
            epochs = st.slider("Training epochs", 10, 200, 50, 10, key="lstm_epochs")
            lr = st.number_input("Learning rate", value=0.001, format="%.4f", key="lstm_lr")

            df_lstm = data_filled.dropna()

            # Prepare sequences
            X_seqs, y_vals = [], []
            for i in range(seq_len, len(df_lstm) - horizon):
                X_seqs.append(df_lstm.iloc[i - seq_len : i].values)
                y_vals.append(df_lstm.iloc[i + horizon - 1][target_col])
            X_tensor = torch.tensor(X_seqs, dtype=torch.float32)
            y_tensor = torch.tensor(y_vals, dtype=torch.float32)

            if len(X_tensor) == 0:
                st.warning("Not enough sequential data after NaN handling.")
            else:
                dataset = TensorDataset(X_tensor, y_tensor)
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_ds, val_ds = torch.utils.data.random_split(
                    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
                )
                train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
                val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = LSTMRegressor(n_features=df_lstm.shape[1]).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                if st.button("🚀 Train LSTM"):
                    progress = st.progress(0.0, text="Training LSTM...")
                    for epoch in range(epochs):
                        model.train()
                        epoch_loss = 0.0
                        for xb, yb in train_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            optimizer.zero_grad()
                            preds = model(xb)
                            loss = criterion(preds, yb)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item() * xb.size(0)
                        epoch_loss /= len(train_loader.dataset)
                        progress.progress((epoch + 1) / epochs, text=f"Epoch {epoch + 1}/{epochs} - loss {epoch_loss:.4f}")
                    progress.empty()
                    st.success(f"Training complete. Final train loss: {epoch_loss:.4f}")

                    # Save model
                    import time
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    model_path = f"/content/drive/MyDrive/py/lstm_model_{ts}.pth"
                    torch.save(model.state_dict(), model_path)
                    st.success(f"📦 LSTM model saved to: {model_path}")

                    # Validation evaluation
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            preds = model(xb)
                            loss = criterion(preds, yb)
                            val_losses.append(loss.item() * xb.size(0))
                    val_loss = sum(val_losses) / len(val_loader.dataset)
                    st.write(f"Validation loss: {val_loss:.4f}")

                    # Predict next single value (forecast horizon)
                    latest_seq = torch.tensor(df_lstm.tail(seq_len).values, dtype=torch.float32).unsqueeze(0).to(device)
                    pred_value = model(latest_seq).item()
                    st.metric(label=f"📈 Predicted {target_col} ({horizon} months ahead)", value=f"{pred_value:,.2f}")

                    # Multi-step forecast for horizon steps
                    forecast_seq = df_lstm.tail(seq_len).values.copy()
                    forecast_preds = []

                    with torch.no_grad():
                        input_seq = torch.tensor(forecast_seq, dtype=torch.float32).unsqueeze(0).to(device)
                        for _ in range(horizon):
                            pred = model(input_seq).item()
                            forecast_preds.append(pred)
                            # Prepare next input sequence by dropping oldest and appending predicted target
                            next_input = input_seq.cpu().numpy().squeeze(0)[1:]  # drop oldest timestep
                            new_step = next_input[-1].copy()  # copy last known features
                            target_idx = df_lstm.columns.get_loc(target_col)
                            new_step[target_idx] = pred  # replace target col with predicted value
                            next_input = np.vstack([next_input, new_step])
                            input_seq = torch.tensor(next_input, dtype=torch.float32).unsqueeze(0).to(device)

                    # Build forecast index starting from next month after last actual date
                    last_date = df_lstm.index[-1]
                    forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=horizon, freq='M')

                    # Build DataFrame for forecast values
                    forecast_df = pd.DataFrame({target_col: forecast_preds}, index=forecast_index)

                    # Plot recent actual + forecast
                    plot_window = 24  # months of recent actuals to show
                    plot_actual = df_lstm[target_col].tail(plot_window)

                    st.subheader(f"📊 Forecast Plot for {target_col} ({horizon} months ahead)")
                    fig_forecast = px.line(title=f"{target_col} Actual + Forecast")
                    fig_forecast.add_scatter(x=plot_actual.index, y=plot_actual.values, mode='lines', name='Actual')
                    fig_forecast.add_scatter(x=forecast_df.index, y=forecast_df[target_col], mode='lines+markers', name='Forecast')
                    st.plotly_chart(fig_forecast, use_container_width=True)

else:
    st.info("🔑 Please enter your FRED API key in the sidebar.")

# --- Sidebar footer ------------------------------------------------
sidebar.markdown("---")
sidebar.markdown("""
ℹ️ **Tips**

• All data is resampled to monthly frequency for consistency.

• Quarterly series (like ECOMSA) are forward-filled monthly.

• Data from FRED: https://fred.stlouisfed.org

• Add your own series in `SERIES_CODES`.
""")


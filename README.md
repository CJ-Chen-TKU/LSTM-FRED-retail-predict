# LSTM-FRED-retail-predict
# 📊 Retail Market Dashboard with LSTM Forecasting

This interactive Streamlit dashboard allows users to explore key U.S. retail market economic indicators from [FRED](https://fred.stlouisfed.org), visualize trends, analyze correlations, and **train a PyTorch LSTM model to forecast future values**.

## 🚀 Features

- 📈 **Visualize Retail Indicators** from FRED:
  - Total Retail Sales, E-Commerce Sales (ECOMSA), Employment, Sentiment, Consumption, and more.
- 🕒 **Monthly Resampling**: All series are automatically resampled to monthly frequency (including quarterly series like ECOMSA).
- 🔍 **Correlation Heatmap**: See relationships between time series via percent-change correlation matrix.
- 💡 **KPI Cards**: Quick glance of latest values and percentage change over selected time period.
- 🧠 **LSTM Forecasting**:
  - Train an LSTM model on selected series.
  - Choose sequence length, forecast horizon, learning rate, and epochs.
  - Real-time training feedback with forecast plot.
  - Save model to Google Drive automatically.

## 📸 Screenshot

![Retail Dashboard Screenshot](./screenshot.png)  
<sup>Add your own screenshot image here</sup>

---

## 🛠️ Installation

### 1. Clone this repository:

```bash
git clone https://github.com/your-username/retail-market-dashboard.git
cd retail-market-dashboard


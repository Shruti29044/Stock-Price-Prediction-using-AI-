# 📊 Stock Price Trend Predictor with AI

This project builds a simple but powerful **AI-based stock trend prediction app** using:

* 📈 **Moving Averages** (MA5, MA10)
* 🧠 **Random Forest Classifier**
* 💾 **SQLite** for historical data caching
* 🕯️ **Candlestick chart** visualization via Plotly
* 🌐 **Gradio** for an interactive web interface

---

## 🚀 Features

* Predicts 3-day **uptrend or downtrend** for any stock ticker (e.g. AAPL, TSLA, MSFT)
* Uses real historical data fetched via `yfinance`
* Saves downloaded data to **SQLite**, improving speed on future runs
* Visualizes the last 60 days with an interactive **candlestick chart**
* Requires **only a browser** (runs via Gradio)

---

## 🛠️ Installation

Run this in a Jupyter or Google Colab cell:

```python
!pip install -q gradio pandas scikit-learn yfinance plotly
```

---

## 🧠 How It Works

1. **Data Loading**
   Loads stock data from SQLite if available. If not, fetches from Yahoo Finance and saves to SQLite.

2. **Feature Engineering**
   Adds:

   * 5-day Moving Average (MA5)
   * 10-day Moving Average (MA10)
   * Target column for supervised learning (uptrend in 3 days)

3. **Model Training**

   * Splits data into train/test sets
   * Trains a **Random Forest Classifier**
   * Predicts next 3-day trend and calculates test set accuracy

4. **Visualization**

   * Displays a candlestick chart for the last 60 days
   * Shows a table of recent prices and moving averages

---

## 🧪 Example Usage

Just run the final cell. Then enter any valid stock ticker like:

```
AAPL
TSLA
GOOGL
AMZN
MSFT
```

You'll see:

* A prediction (Uptrend/Downtrend)
* Accuracy percentage
* Price & MA table
* Candlestick chart

---

## 📝 Notes

* Requires at least \~30 days of historical stock data to make predictions
* SQLite database is named `stock_data.db` (created automatically)
* No API key required (data from Yahoo Finance)

---

## 📦 Technologies Used

* `pandas`, `numpy`
* `scikit-learn` (Random Forest)
* `yfinance`
* `sqlite3`
* `plotly` (Candlestick)
* `gradio`

---

## 💡 Future Improvements

* Add more indicators (RSI, MACD)
* Use LSTM/GRU models
* Add longer forecast windows (7-day, 14-day)


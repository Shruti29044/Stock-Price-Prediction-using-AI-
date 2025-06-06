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


## 🧩Challenges Faced (In Detail)
Building an AI-powered stock trend prediction system using technical indicators, machine learning, and data storage pipelines posed a variety of real-world engineering and machine learning challenges. Here's a breakdown of the major issues encountered during development:

1. 🔍 Low Initial Model Accuracy
Problem: When starting with basic machine learning models (e.g., Logistic Regression or Decision Trees) using limited features (Close, MA5, MA10), the model struggled to achieve meaningful accuracy.
Solution: I upgraded to a Random Forest Classifier and later explored XGBoost, which is better at capturing nonlinear relationships and temporal patterns. I also experimented with time-based train-test splits instead of random shuffles to reflect real-world constraints.

2. 🧪 Label Engineering
Problem: Defining the "target" for prediction was more nuanced than it seemed. Initially, I used a binary target: “Will the price be higher in 3 days?” However, that label often misclassified sideways movement or slight noise as trends.
Solution: I refined the label logic using actual movement thresholds (like a minimum % change) to make the target more meaningful. I also left room for integrating volatility-based or RSI-based labels in the future.

3. 💾 Working with SQL in Colab
Problem: Integrating SQLite in Colab was tricky. Issues included:

Table creation and overwriting errors

Pandas’ to_sql() and read_sql() occasionally failing due to mismatched columns

Indexing problems with datetime columns
Solution: I added safe wrappers to check table existence, used parse_dates=["Date"] when loading, and ensured the schema was consistent between fetch and save.

4. 📊 Visualizing with Plotly Candlestick
Problem: Plotly’s candlestick chart requires clean and complete OHLC (Open, High, Low, Close) data for every row. Missing or malformed values caused the chart to silently fail.
Solution: I cleaned the last 60 rows of the dataframe explicitly and added input checks. I also isolated the chart function so that issues in prediction wouldn’t break the visualization.

5. 🧠 Overfitting on Limited Data
Problem: With just 6 months of historical stock data, models could memorize the data patterns and not generalize well to future trends.
Solution: I:

Used time-aware train/test splits (no shuffling)

Tuned hyperparameters conservatively

Added technical indicators to reduce reliance on raw prices

6. ⚠️ Gradio Runtime Crashes
Problem: Gradio failed silently when:

A function returned None

Data wasn’t fetched properly from yfinance

The wrong format was passed to a Plotly component
Solution: I added debug print statements inside the prediction function, caught and displayed fallback messages, and used debug=True during testing to surface backend errors.

7. 🌐 Repeated Data Downloads
Problem: Fetching data via yfinance every time a ticker was queried slowed the UI and caused throttling from the API.
Solution: I implemented SQLite caching. On first fetch, data is saved to stock_data.db. On subsequent predictions, it’s loaded locally unless missing or outdated.

8. 🔢 Feature Scaling (Optional Challenge)
Problem: Random Forest doesn’t require feature scaling, but adding other models (like SVM or LSTM) would. Managing this in a multi-model pipeline adds complexity.
Solution: I kept scaling logic modular so it could be plugged in conditionally if switching to models sensitive to scale.

9. 📈 Prediction Interpretability
Problem: Users wanted to know why a trend was predicted. But tree-based models like Random Forest aren’t easily interpretable without extra tools.
Solution: I added indicator values (MA5, MA10, Close) in the UI to help users reason about trends. SHAP or feature importance can be added in future iterations.

10. 🧪 Testing Real-World Use Cases
Problem: Testing the model on random tickers revealed many with missing data, splits, or low volume — which hurt predictions.
Solution: I added pre-checks for data length, column availability, and added fallback messages like “Not enough data for prediction.”



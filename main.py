# Install dependencies
!pip install -q gradio pandas scikit-learn yfinance plotly

import pandas as pd
import numpy as np
import yfinance as yf
import gradio as gr
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import os

# === SQLite DB functions ===
DB_FILE = "stock_data.db"

def save_to_sqlite(df, ticker):
    conn = sqlite3.connect(DB_FILE)
    df.to_sql(ticker, conn, if_exists="replace", index=True)
    conn.close()

def load_from_sqlite(ticker):
    if not os.path.exists(DB_FILE):
        return None
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, index_col="Date", parse_dates=["Date"])
        return df
    except Exception:
        return None
    finally:
        conn.close()

# === Feature Engineering ===
def add_indicators(df):
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

# === Main prediction logic ===
def predict_stock(ticker):
    # Try loading from SQLite, else fetch from yfinance
    df = load_from_sqlite(ticker.upper())
    if df is None or len(df) < 30:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty or len(df) < 30:
            return "Not enough data for prediction", None, None
        save_to_sqlite(df, ticker)

    df = add_indicators(df)
    features = ["Close", "MA5", "MA10"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    latest = X.iloc[-1].values.reshape(1, -1)
    forecast = model.predict(latest)[0]
    trend = "📈 Uptrend expected in 3 days" if forecast == 1 else "📉 Downtrend expected in 3 days"

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index[-60:], open=df["Open"][-60:], high=df["High"][-60:],
        low=df["Low"][-60:], close=df["Close"][-60:]
    )])
    fig.update_layout(title=f"{ticker.upper()} - Last 60 Days", xaxis_rangeslider_visible=False)

    return f"{trend}\nModel Accuracy: {round(acc * 100, 2)}%", df.tail(30)[features], fig

# === Gradio UI ===
def launch_app():
    gr.Interface(
        fn=predict_stock,
        inputs=gr.Textbox(label="Enter Stock Ticker (e.g., AAPL,TSLA,GOOGL,MSFT,AMZN)"),
        outputs=[
            gr.Textbox(label="Prediction Result"),
            gr.Dataframe(label="Recent Prices + MA Indicators"),
            gr.Plot(label="Candlestick Chart")
        ],
        title="📊 Stock Trend Predictor with AI",
        description="Enter a stock ticker to see technical indicator-based prediction. Data is stored in SQLite for faster future access."
    ).launch()

launch_app()

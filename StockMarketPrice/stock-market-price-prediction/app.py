from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import gradio as gr

from src.predict import load_metadata, predict_from_yahoo, predict_next_close
from src.train import MODEL_PATH, train


if not MODEL_PATH.exists():
    train()


def train_model(ticker: str, start: str, end: str, data_file: Optional[str]) -> str:
    uploaded_path = getattr(data_file, "name", data_file)
    data_path = Path(uploaded_path) if uploaded_path else None
    _, metrics = train(
        data_path=data_path,
        ticker=ticker.strip().upper() or "AAPL",
        start=start or "2020-01-01",
        end=end or None,
    )
    return json.dumps(metrics, indent=2)


def predict_from_latest_yahoo(ticker: str, start: str, end: str) -> str:
    result = predict_from_yahoo(
        ticker=ticker.strip().upper() or load_metadata().get("ticker", "AAPL"),
        start=start or "2020-01-01",
        end=end or None,
    )
    return (
        f"Last close: ${result['last_close']:,.2f}\n"
        f"Predicted next close: ${result['predicted_next_close']:,.2f}\n"
        f"Predicted change: ${result['predicted_change']:,.2f} "
        f"({result['predicted_change_percent']:,.2f}%)"
    )


def predict_from_features(
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    lag_1: float,
    lag_2: float,
    lag_5: float,
    ma_5: float,
    ma_10: float,
    return_1: float,
    volatility_5: float,
) -> str:
    prediction = predict_next_close(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_5": lag_5,
            "ma_5": ma_5,
            "ma_10": ma_10,
            "return_1": return_1,
            "volatility_5": volatility_5,
        }
    )
    return f"${prediction:,.2f}"


with gr.Blocks(title="Stock Market Price Prediction") as demo:
    gr.Markdown("# Stock Market Price Prediction")
    gr.Markdown("Train on Yahoo Finance, CSV, or Excel OHLCV data, then predict the next closing price.")

    with gr.Tab("Train and Evaluate"):
        with gr.Row():
            ticker = gr.Textbox(value="AAPL", label="Yahoo ticker")
            start = gr.Textbox(value="2020-01-01", label="Start date")
            end = gr.Textbox(value="", label="End date (optional)")
        data_file = gr.File(label="Optional CSV or Excel file", file_types=[".csv", ".xlsx", ".xls"])
        train_button = gr.Button("Train Model", variant="primary")
        metrics_output = gr.Code(label="Evaluation Metrics", language="json")
        train_button.click(train_model, inputs=[ticker, start, end, data_file], outputs=metrics_output)

    with gr.Tab("Predict From Yahoo"):
        with gr.Row():
            predict_ticker = gr.Textbox(value="AAPL", label="Yahoo ticker")
            predict_start = gr.Textbox(value="2020-01-01", label="Start date")
            predict_end = gr.Textbox(value="", label="End date (optional)")
        yahoo_button = gr.Button("Predict Next Close", variant="primary")
        yahoo_output = gr.Textbox(label="Prediction", lines=4)
        yahoo_button.click(
            predict_from_latest_yahoo,
            inputs=[predict_ticker, predict_start, predict_end],
            outputs=yahoo_output,
        )

    with gr.Tab("Predict From Features"):
        with gr.Row():
            with gr.Column():
                open_price = gr.Number(value=190.0, label="Open")
                high = gr.Number(value=195.0, label="High")
                low = gr.Number(value=188.0, label="Low")
                close = gr.Number(value=192.0, label="Close")
                volume = gr.Number(value=60_000_000, label="Volume")
                lag_1 = gr.Number(value=191.0, label="Previous close")
            with gr.Column():
                lag_2 = gr.Number(value=190.0, label="Close two days ago")
                lag_5 = gr.Number(value=188.0, label="Close five days ago")
                ma_5 = gr.Number(value=190.5, label="5-day moving average")
                ma_10 = gr.Number(value=189.5, label="10-day moving average")
                return_1 = gr.Number(value=0.005, label="1-day return")
                volatility_5 = gr.Number(value=0.018, label="5-day volatility")
        feature_button = gr.Button("Predict", variant="primary")
        feature_output = gr.Textbox(label="Predicted Next Close")
        feature_button.click(
            predict_from_features,
            inputs=[
                open_price,
                high,
                low,
                close,
                volume,
                lag_1,
                lag_2,
                lag_5,
                ma_5,
                ma_10,
                return_1,
                volatility_5,
            ],
            outputs=feature_output,
        )


if __name__ == "__main__":
    demo.launch()

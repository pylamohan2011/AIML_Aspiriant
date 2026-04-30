---
title: Stock Market Price Prediction
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
---

# Stock Market Price Prediction

End-to-end machine learning project for next-day stock closing price prediction. It can train from Yahoo Finance, CSV, or Excel OHLCV data, evaluate multiple regression models, expose a FastAPI endpoint, and run as a Gradio app on Hugging Face Spaces.

## Project structure

```text
stock-market-price-prediction/
  app.py                  # Gradio app for local use and Hugging Face Spaces
  api.py                  # FastAPI app
  data/                   # Optional stock_prices.csv or stock_prices.xlsx
  models/                 # Saved model, metrics, metadata
  src/train.py            # Load data, engineer features, train, evaluate, save
  src/predict.py          # Load model and predict
  src/sample_data.py      # Generates sample OHLCV data for smoke tests
  requirements.txt
```

## Dataset

Use any Yahoo-style OHLCV file with these columns:

```text
Date, Open, High, Low, Close, Volume
```

Place one of these files in `data/`:

```text
data/stock_prices.csv
data/stock_prices.xlsx
```

If no file exists, training tries Yahoo Finance with `yfinance`. If Yahoo/network access is unavailable, it creates a sample dataset so the workflow still runs.

## Train and evaluate

```powershell
cd StockMarketPrice\stock-market-price-prediction
python src/train.py
```

The script trains Ridge, Random Forest, and Histogram Gradient Boosting models, selects the lowest-RMSE model, and writes:

```text
models/stock_price_model.joblib
models/metrics.json
models/metadata.json
```

## Run FastAPI

```powershell
cd StockMarketPrice\stock-market-price-prediction
python -m uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run Gradio

```powershell
cd StockMarketPrice\stock-market-price-prediction
python app.py
```

For a background local process on Windows:

```powershell
python run_gradio_local.py
```

## Deploy to Hugging Face Spaces

Target Space:

```text
https://huggingface.co/spaces/pylamohan2011/StockmarketPrice
```

Use a Gradio Space. Recommended files:

```text
app.py
requirements.txt
src/
models/stock_price_model.joblib
models/metadata.json
models/metrics.json
```

Deploy directly with:

```powershell
$env:HF_TOKEN="hf_your_write_token_here"
python deploy_to_hf.py
```

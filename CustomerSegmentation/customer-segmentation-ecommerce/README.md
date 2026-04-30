---
title: E-commerce Customer Segmentation
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
---

# E-commerce Customer Segmentation

End-to-end machine learning project for customer segmentation using Kaggle Online Retail-style e-commerce transaction data. It builds RFM and sales features, trains and evaluates K-Means clustering models, exposes a FastAPI endpoint, and provides a Gradio web app for Hugging Face Spaces.

## Project structure

```text
customer-segmentation-ecommerce/
  app.py                  # Gradio app for local use and Hugging Face Spaces
  api.py                  # FastAPI app
  data/online_retail.csv  # Kaggle dataset location
  models/                 # Saved model, metrics, metadata
  src/train.py            # Train, evaluate, save clustering model
  src/predict.py          # Load model and predict segment
  src/sample_data.py      # Generates sample Online Retail-like data
  requirements.txt
```

## Dataset

Recommended Kaggle dataset: Online Retail / e-commerce transaction data with columns similar to:

```text
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

Place your file at one of:

```text
data/online_retail.csv
data/online_retail.xlsx
```

If no Kaggle file exists, `src/train.py` creates a sample dataset so the full workflow can run.

## Train and evaluate

```powershell
cd CustomerSegmentation\customer-segmentation-ecommerce
python src/train.py
```

The script tries K values from 2 to 6, selects the best silhouette score, and writes:

```text
models/customer_segmentation_model.joblib
models/metrics.json
models/metadata.json
```

## Run FastAPI

```powershell
cd CustomerSegmentation\customer-segmentation-ecommerce
python -m uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run Gradio

```powershell
cd CustomerSegmentation\customer-segmentation-ecommerce
python app.py
```

For a background local process on Windows:

```powershell
python run_gradio_local.py
```

## Deploy to Hugging Face Spaces

Target Space:

```text
https://huggingface.co/spaces/pylamohan2011/CustomerSegmentation
```

Deploy directly with:

```powershell
$env:HF_TOKEN="hf_your_write_token_here"
python deploy_to_hf.py
```

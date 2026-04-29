---
title: House Price Prediction
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
---

# House Price Prediction

End-to-end machine learning project for Kaggle house price prediction. It trains and evaluates regression models, saves the best pipeline, exposes a FastAPI endpoint, and provides a Gradio web app that can run on Hugging Face Spaces.

## Project structure

```text
house-price-prediction/
  app.py                  # Gradio app for local use and Hugging Face Spaces
  api.py                  # FastAPI app
  data/train.csv          # Kaggle dataset location
  models/                 # Saved model, metrics, metadata
  src/train.py            # Train, evaluate, save model
  src/predict.py          # Load model and predict
  src/sample_data.py      # Generates a local sample dataset
  requirements.txt
```

## Dataset

Use the Kaggle House Prices dataset:

1. Download `train.csv` from Kaggle: `house-prices-advanced-regression-techniques`.
2. Place it at `CapstonProjects/house-price-prediction/data/train.csv`.

If `data/train.csv` is missing, `src/train.py` creates a small Kaggle-like sample dataset so the project can still run end to end.

## Train and evaluate

```powershell
cd CapstonProjects/house-price-prediction
python src/train.py
```

The script trains Ridge, Random Forest, and Gradient Boosting models, selects the model with the lowest RMSE, and writes:

- `models/house_price_model.joblib`
- `models/metrics.json`
- `models/metadata.json`

## Run FastAPI

```powershell
cd CapstonProjects/house-price-prediction
python -m uvicorn api:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/`

Example request:

```json
{
  "OverallQual": 7,
  "GrLivArea": 1500,
  "GarageCars": 2,
  "TotalBsmtSF": 900,
  "1stFlrSF": 900,
  "FullBath": 2,
  "BedroomAbvGr": 3,
  "YearBuilt": 2000,
  "YearRemodAdd": 2010,
  "LotArea": 9000,
  "Neighborhood": "CollgCr",
  "HouseStyle": "1Story"
}
```

## Run Gradio

```powershell
cd CapstonProjects/house-price-prediction
python app.py
```

For a background local process on Windows:

```powershell
python run_gradio_local.py
```

## Deploy to Hugging Face Spaces

Target Space:

```text
https://huggingface.co/spaces/pylamohan2011/MLSpaces
```

Use a Gradio Space.

1. Open your Hugging Face Space.
2. Upload or push these files from `CapstonProjects/house-price-prediction`.
3. Make sure the Space contains `app.py`, `requirements.txt`, `src/`, `models/`, and optionally `data/train.csv`.
4. If `models/house_price_model.joblib` is not present, the app trains automatically from `data/train.csv`; if the Kaggle CSV is also missing, it trains on the sample dataset.

Recommended files to commit to the Space:

```text
app.py
requirements.txt
src/
models/house_price_model.joblib
models/metadata.json
models/metrics.json
```

For best results, train locally with Kaggle `train.csv` first, then upload the saved `models/` files to the Space.

You can deploy directly with:

```powershell
$env:HF_TOKEN="hf_your_write_token_here"
python deploy_to_hf.py
```

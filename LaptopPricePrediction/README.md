---
title: Laptop Price Prediction
emoji: 🖥️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Laptop Price Prediction

This project trains a laptop price regression model, exposes a FastAPI prediction endpoint, and provides a Gradio web app for interactive forecasting.

## Structure

- `src/train.py` - trains the model from `data/laptop_price.csv` or generated sample data.
- `src/predict.py` - loads the saved model and makes predictions.
- `src/sample_data.py` - generates a sample laptop pricing dataset when a real dataset is missing.
- `app.py` - Gradio web interface.
- `api.py` - FastAPI prediction API.
- `run_gradio_local.py` - launches the local Gradio app.
- `deploy_to_hf.py` - deploys the project to Hugging Face Spaces.
- `requirements.txt` - Python dependencies.

## Setup

1. Create and activate your environment in `LaptopPricePrediction`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Using a Kaggle dataset

Place your Kaggle laptop price CSV as:

```bash
LaptopPricePrediction/data/laptop_price.csv
```

Expected fields include:

- `Company`
- `TypeName`
- `Inches`
- `Resolution`
- `Cpu`
- `Ram`
- `Memory`
- `Gpu`
- `OpSys`
- `Weight`
- `Price_euros`

If the file is missing, the project creates a sample dataset automatically.

## Train the model

```bash
cd LaptopPricePrediction
python src/train.py
```

## Run the Gradio app locally

```bash
cd LaptopPricePrediction
python run_gradio_local.py
```

## Run the FastAPI service locally

```bash
cd LaptopPricePrediction
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Then POST to `http://127.0.0.1:8000/predict` with JSON matching the input schema.

## Deploy to Hugging Face Spaces

Set your Hugging Face token and optionally the `HF_SPACE_ID` environment variable:

```bash
set HF_TOKEN=hf_your_token_here
set HF_SPACE_ID=pylamohan2011/laptop-price-prediction
python deploy_to_hf.py
```

If you already created a Space, set `HF_SPACE_ID` to the correct repo name.

---
title: Titanic Survival Prediction
emoji: 🚢
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
---

# Titanic Survival Prediction

End-to-end Titanic survival classification project. It trains and evaluates a model on the Kaggle Titanic dataset, saves the best pipeline, exposes a FastAPI endpoint, and provides a Gradio web app for local use or deployment to Hugging Face Spaces.

## Project structure

```text
titanic/
  app.py                  # Gradio app for local use and Hugging Face Spaces
  api.py                  # FastAPI app
  data/                   # Optional Kaggle Titanic dataset
  deploy_to_hf.py         # Optional deploy helper for Hugging Face Spaces
  models/                 # Saved model and metrics
  requirements.txt
  run_gradio_local.py
  src/
    __init__.py
    sample_data.py
    train.py
    predict.py
```

## Dataset

Use the Kaggle Titanic dataset:

1. Download the Titanic dataset from Kaggle: `titanic/train.csv`
2. Place it at `titanic/data/train.csv`

Alternatively, if you have Kaggle credentials configured, you can download the dataset automatically:

```powershell
cd Titanic
python download_kaggle_data.py --output data
```

Make sure your Kaggle credentials are available as either:

- environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`
- or `~/.kaggle/kaggle.json`

> Note: `Titanic/data/train.csv` is not committed to GitHub. The repository keeps a placeholder file at `Titanic/data/.gitkeep` so the folder structure is preserved, while the actual dataset is downloaded locally.

If `data/train.csv` is missing, the project falls back to a small sample dataset so the app can still run end to end.

## Train and evaluate

```powershell
cd titanic
python src/train.py
```

This trains a classification pipeline and writes:

- `models/titanic_model.joblib`
- `models/metrics.json`
- `models/metadata.json`

## Run FastAPI

```powershell
cd titanic
python -m uvicorn api:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/`

Example request:

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22.0,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S",
  "Name": "Braund, Mr. Owen Harris"
}
```

## Run Gradio

```powershell
cd titanic
python app.py
```

Or run locally in a separate process:

```powershell
python run_gradio_local.py
```

## Deploy to Hugging Face Spaces

1. Create or choose a Hugging Face Space that uses the Gradio SDK.
2. Upload `app.py`, `requirements.txt`, `src/`, and optionally `models/` and `data/train.csv`.
3. If the model file is absent, the app trains automatically on the sample dataset or the Kaggle CSV if available.

To deploy programmatically, set `HF_TOKEN` and optionally `HF_SPACE_ID`:

```powershell
$env:HF_TOKEN="hf_your_write_token_here"
$env:HF_SPACE_ID="your-username/titanic-survival-prediction"
python deploy_to_hf.py
```

## GitHub Actions Deployment

A GitHub Actions workflow is included at `.github/workflows/deploy-titanic-space.yml`.

1. Create GitHub repository secrets:
   - `HF_TOKEN` — your Hugging Face write token
   - `HF_SPACE_ID` — your Hugging Face Space ID, e.g. `pylamohan2011/titanic-survival-prediction`
2. Push changes to the `main` branch.
3. The workflow runs automatically and deploys the `Titanic/` project to your Space.

The workflow uses `python Titanic/deploy_to_hf.py` so deployment is consistent with the local helper script.

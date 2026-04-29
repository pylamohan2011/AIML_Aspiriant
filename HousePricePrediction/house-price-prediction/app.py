from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.predict import MODEL_PATH, predict_price
from src.train import train


NEIGHBORHOODS = ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Somerst", "OldTown"]
HOUSE_STYLES = ["1Story", "2Story", "1.5Fin", "SLvl"]


if not MODEL_PATH.exists():
    train()


def estimate_price(
    overall_qual: int,
    living_area: float,
    garage_cars: int,
    basement_area: float,
    first_floor_area: float,
    full_bath: int,
    bedrooms: int,
    year_built: int,
    year_remodeled: int,
    lot_area: float,
    neighborhood: str,
    house_style: str,
) -> str:
    features = {
        "OverallQual": overall_qual,
        "GrLivArea": living_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": basement_area,
        "1stFlrSF": first_floor_area,
        "FullBath": full_bath,
        "BedroomAbvGr": bedrooms,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remodeled,
        "LotArea": lot_area,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
    }
    price = predict_price(features)
    return f"${price:,.0f}"


with gr.Blocks(title="House Price Prediction") as demo:
    gr.Markdown("# House Price Prediction")
    gr.Markdown("Enter property details to estimate the expected sale price.")
    with gr.Row():
        with gr.Column():
            overall_qual = gr.Slider(1, 10, value=7, step=1, label="Overall Quality")
            living_area = gr.Number(value=1500, label="Above Ground Living Area")
            garage_cars = gr.Slider(0, 5, value=2, step=1, label="Garage Cars")
            basement_area = gr.Number(value=900, label="Basement Area")
            first_floor_area = gr.Number(value=900, label="First Floor Area")
            full_bath = gr.Slider(0, 5, value=2, step=1, label="Full Bathrooms")
        with gr.Column():
            bedrooms = gr.Slider(0, 10, value=3, step=1, label="Bedrooms")
            year_built = gr.Number(value=2000, label="Year Built")
            year_remodeled = gr.Number(value=2010, label="Year Remodeled")
            lot_area = gr.Number(value=9000, label="Lot Area")
            neighborhood = gr.Dropdown(NEIGHBORHOODS, value="CollgCr", label="Neighborhood")
            house_style = gr.Dropdown(HOUSE_STYLES, value="1Story", label="House Style")

    output = gr.Textbox(label="Predicted Price")
    submit = gr.Button("Predict", variant="primary")
    submit.click(
        estimate_price,
        inputs=[
            overall_qual,
            living_area,
            garage_cars,
            basement_area,
            first_floor_area,
            full_bath,
            bedrooms,
            year_built,
            year_remodeled,
            lot_area,
            neighborhood,
            house_style,
        ],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()

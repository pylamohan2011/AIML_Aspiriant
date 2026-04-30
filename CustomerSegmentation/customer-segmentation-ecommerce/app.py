from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import gradio as gr

from src.predict import load_metadata, predict_segment
from src.train import MODEL_PATH, train


if not MODEL_PATH.exists():
    train()


def train_model(data_file: Optional[str]) -> str:
    uploaded_path = getattr(data_file, "name", data_file)
    data_path = Path(uploaded_path) if uploaded_path else None
    _, metrics = train(data_path=data_path)
    return json.dumps(metrics, indent=2)


def predict_customer_segment(
    recency_days: float,
    frequency: float,
    total_quantity: float,
    total_spend: float,
    average_order_value: float,
    unique_products: float,
) -> str:
    result = predict_segment(
        {
            "recency_days": recency_days,
            "frequency": frequency,
            "total_quantity": total_quantity,
            "total_spend": total_spend,
            "average_order_value": average_order_value,
            "unique_products": unique_products,
        }
    )
    return f"{result['segment']} (cluster {result['cluster']})"


example = load_metadata().get("example_customer", {})

with gr.Blocks(title="E-commerce Customer Segmentation") as demo:
    gr.Markdown("# E-commerce Customer Segmentation")
    gr.Markdown("Train on Kaggle Online Retail-style transactions, evaluate clusters, and assign a customer segment.")

    with gr.Tab("Train and Evaluate"):
        data_file = gr.File(label="Optional Kaggle CSV or Excel file", file_types=[".csv", ".xlsx", ".xls"])
        train_button = gr.Button("Train Model", variant="primary")
        metrics_output = gr.Code(label="Evaluation Metrics", language="json")
        train_button.click(train_model, inputs=[data_file], outputs=metrics_output)

    with gr.Tab("Predict Segment"):
        with gr.Row():
            with gr.Column():
                recency_days = gr.Number(value=float(example.get("recency_days", 30)), label="Recency in days")
                frequency = gr.Number(value=float(example.get("frequency", 8)), label="Number of orders")
                total_quantity = gr.Number(value=float(example.get("total_quantity", 160)), label="Total quantity")
            with gr.Column():
                total_spend = gr.Number(value=float(example.get("total_spend", 2500)), label="Total spend")
                average_order_value = gr.Number(
                    value=float(example.get("average_order_value", 312.5)),
                    label="Average order value",
                )
                unique_products = gr.Number(value=float(example.get("unique_products", 20)), label="Unique products")
        predict_button = gr.Button("Predict Segment", variant="primary")
        segment_output = gr.Textbox(label="Segment")
        predict_button.click(
            predict_customer_segment,
            inputs=[
                recency_days,
                frequency,
                total_quantity,
                total_spend,
                average_order_value,
                unique_products,
            ],
            outputs=segment_output,
        )


if __name__ == "__main__":
    demo.launch()

from __future__ import annotations

import gradio as gr

from src.predict import MODEL_PATH, predict_price
from src.train import train

COMPANIES = ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"]
TYPES = ["Ultrabook", "Gaming", "Notebook", "2 in 1", "Workstation"]
RESOLUTIONS = ["1920x1080", "2560x1440", "3840x2160", "1366x768"]
CPUS = ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Core i9", "AMD Ryzen 5", "AMD Ryzen 7"]
RAM_OPTIONS = [4, 8, 16, 32]
MEMORY_OPTIONS = ["256GB SSD", "512GB SSD", "1TB SSD", "256GB SSD + 1TB HDD", "512GB SSD + 1TB HDD"]
GPUS = ["Intel", "Nvidia GTX 1650", "Nvidia RTX 3050", "Nvidia RTX 3060", "Nvidia RTX 4070"]
OSES = ["Windows", "macOS", "Linux", "Chrome OS"]
WEIGHTS = [1.1, 1.3, 1.6, 2.0, 2.5]
INCHES = [13.3, 14.0, 15.6, 16.0, 17.3]


if not MODEL_PATH.exists():
    train()


def estimate_price(
    company: str,
    type_name: str,
    inches: float,
    resolution: str,
    cpu: str,
    ram: int,
    memory: str,
    gpu: str,
    op_sys: str,
    weight: float,
) -> str:
    features = {
        "Company": company,
        "TypeName": type_name,
        "Inches": inches,
        "Resolution": resolution,
        "Cpu": cpu,
        "Ram": ram,
        "Memory": memory,
        "Gpu": gpu,
        "OpSys": op_sys,
        "Weight": weight,
    }
    predicted = predict_price(features)
    return f"€{predicted:,.2f}"


with gr.Blocks(title="Laptop Price Prediction") as demo:
    gr.Markdown("# Laptop Price Prediction")
    gr.Markdown(
        "Use the slider and dropdown controls to estimate the expected laptop price in euros."
    )
    with gr.Row():
        with gr.Column():
            company = gr.Dropdown(COMPANIES, value="Dell", label="Company")
            type_name = gr.Dropdown(TYPES, value="Notebook", label="Type")
            inches = gr.Dropdown(INCHES, value=15.6, label="Screen Size (inches)")
            resolution = gr.Dropdown(RESOLUTIONS, value="1920x1080", label="Resolution")
            cpu = gr.Dropdown(CPUS, value="Intel Core i5", label="CPU")
            ram = gr.Dropdown(RAM_OPTIONS, value=8, label="RAM (GB)")
        with gr.Column():
            memory = gr.Dropdown(MEMORY_OPTIONS, value="512GB SSD", label="Storage")
            gpu = gr.Dropdown(GPUS, value="Intel", label="GPU")
            op_sys = gr.Dropdown(OSES, value="Windows", label="Operating System")
            weight = gr.Dropdown(WEIGHTS, value=1.8, label="Weight (kg)")

    output = gr.Textbox(label="Predicted Price")
    submit = gr.Button("Predict", variant="primary")
    submit.click(
        estimate_price,
        inputs=[
            company,
            type_name,
            inches,
            resolution,
            cpu,
            ram,
            memory,
            gpu,
            op_sys,
            weight,
        ],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()

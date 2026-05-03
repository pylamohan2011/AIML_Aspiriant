from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.predict import MODEL_PATH, predict_survival
from src.train import train

EMBARKED_OPTIONS = ["S", "C", "Q"]
SEX_OPTIONS = ["male", "female"]
CLASS_OPTIONS = [1, 2, 3]

if not MODEL_PATH.exists():
    train()


def estimate_survival(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
    name: str,
) -> str:
    payload = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Name": name,
    }
    prediction = predict_survival(payload)
    prob = prediction["survival_probability"] * 100
    label = "Survived" if prediction["survived"] else "Did not survive"
    return f"{label} ({prob:.1f}% probability)"


with gr.Blocks(title="Titanic Survival Prediction") as demo:
    gr.Markdown("# Titanic Survival Prediction")
    gr.Markdown("Enter passenger details to predict whether they survived the Titanic disaster.")
    with gr.Row():
        with gr.Column():
            pclass = gr.Radio(CLASS_OPTIONS, value=3, label="Passenger Class")
            sex = gr.Radio(SEX_OPTIONS, value="male", label="Sex")
            age = gr.Slider(0, 80, value=30, step=1, label="Age")
            sibsp = gr.Slider(0, 8, value=0, step=1, label="Siblings / Spouses Aboard")
            parch = gr.Slider(0, 6, value=0, step=1, label="Parents / Children Aboard")
        with gr.Column():
            fare = gr.Number(value=7.25, label="Fare")
            embarked = gr.Dropdown(EMBARKED_OPTIONS, value="S", label="Port of Embarkation")
            name = gr.Textbox(value="Braund, Mr. Owen Harris", label="Passenger Name")

    output = gr.Textbox(label="Prediction")
    submit = gr.Button("Predict", variant="primary")
    submit.click(
        estimate_survival,
        inputs=[pclass, sex, age, sibsp, parch, fare, embarked, name],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()

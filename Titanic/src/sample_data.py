from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def create_sample_titanic() -> pd.DataFrame:
    sample = [
        {
            "PassengerId": i + 1,
            "Survived": survived,
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
        }
        for i, (survived, pclass, name, sex, age, sibsp, parch, fare, embarked) in enumerate(
            [
                (0, 3, "Braund, Mr. Owen Harris", "male", 22.0, 1, 0, 7.25, "S"),
                (1, 1, "Cumings, Mrs. John Bradley (Florence Briggs Thayer)", "female", 38.0, 1, 0, 71.2833, "C"),
                (1, 3, "Heikkinen, Miss. Laina", "female", 26.0, 0, 0, 7.925, "S"),
                (1, 1, "Futrelle, Mrs. Jacques Heath (Lily May Peel)", "female", 35.0, 1, 0, 53.1, "S"),
                (0, 3, "Allen, Mr. William Henry", "male", 35.0, 0, 0, 8.05, "S"),
                (0, 3, "Moran, Mr. James", "male", None, 0, 0, 8.4583, "Q"),
                (1, 1, "McCarthy, Mr. Timothy J", "male", 54.0, 0, 0, 51.8625, "S"),
                (0, 3, "Palsson, Master. Gosta Leonard", "male", 2.0, 3, 1, 21.075, "S"),
                (1, 3, "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)", "female", 27.0, 0, 2, 11.1333, "S"),
                (1, 2, "Nasser, Mrs. Nicholas (Adele Achem)", "female", 14.0, 1, 0, 30.0708, "C"),
                (1, 3, "Sandstrom, Miss. Marguerite Rut", "female", 4.0, 1, 1, 16.7, "S"),
                (1, 1, "Bonnell, Miss. Elizabeth", "female", 58.0, 0, 0, 26.55, "S"),
                (0, 3, "Saundercock, Mr. William Henry", "male", 20.0, 0, 0, 8.05, "S"),
                (0, 3, "Andersson, Mr. Anders Johan", "male", 39.0, 1, 5, 31.275, "S"),
                (1, 3, "Vestrom, Miss. Hulda Amanda Adolfina", "female", 14.0, 0, 0, 7.8542, "S"),
                (0, 2, "Hewlett, Mrs. (Mary D Kingcome) ", "female", 55.0, 0, 0, 16.0, "S"),
                (0, 3, "Rice, Master. Eugene", "male", 2.0, 4, 1, 29.125, "Q"),
                (1, 2, "Williams, Mr. Charles Eugene", "male", 28.0, 0, 0, 13.0, "S"),
                (1, 1, "Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)", "female", 31.0, 1, 0, 18.0, "S"),
                (0, 3, "Masselmani, Mrs. Fatima", "female", None, 0, 0, 7.225, "C"),
            ]
        )
    ]
    return pd.DataFrame(sample)

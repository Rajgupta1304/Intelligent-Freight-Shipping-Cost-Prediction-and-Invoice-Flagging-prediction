import joblib
import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "shipping_cost_prediction"
    / "models"
    / "predict_shipping_cost_model.pkl"
)


def load_model(model_path: str = str(MODEL_PATH)):
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def predict_freight_cost(input_data):
    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_shipping_cost"] = (
        model.predict(input_df).round()
    )

    return input_df


if __name__ == "__main__":
    # Example inference run for local testing

    sample_data = {
        "Dollars": [18500]
    }

    prediction = predict_freight_cost(sample_data)

    print(prediction)
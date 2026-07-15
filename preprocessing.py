"""
predict.py
----------
Loads the saved model and encoders, then predicts churn for a new
customer record. Edit `new_data` below or import `predict_churn()`
into your own script / API / app.
"""

import pickle
import pandas as pd

MODEL_PATH = "models/customer_churn.pkl"
ENCODERS_PATH = "models/encoders.pkl"


def load_artifacts(model_path: str = MODEL_PATH, encoders_path: str = ENCODERS_PATH):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)

    return model_data["model"], model_data["features_names"], encoders


def encode_new_data(new_data: dict, encoders: dict) -> pd.DataFrame:
    """Apply the saved LabelEncoders to a single new customer record."""
    df = pd.DataFrame([new_data])

    for column, encoder in encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column])

    return df


def predict_churn(new_data: dict) -> dict:
    """Return the churn prediction and probability for a single new customer."""
    model, feature_names, encoders = load_artifacts()
    df = encode_new_data(new_data, encoders)
    df = df[feature_names]  # ensure correct column order

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    new_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 2050.00,  # ~24 months * ~85/month
    }

    result = predict_churn(new_data)
    print(result)

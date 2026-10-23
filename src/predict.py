import os
import sys
import pandas as pd
import joblib

# ── Ensure src/ is importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    MODEL_PATH,
)


def load_model(model_path=None):
    """Load the trained pipeline from disk."""
    if model_path is None:
        model_path = MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run 'python src/train.py' first to train and save the model."
        )
    return joblib.load(model_path)


def predict_confirmation(
    pipeline,
    class_of_travel: str,
    quota: str,
    age_of_passengers: str,
    booking_channel: str,
    train_type: str,
    holiday_or_peak_season: str,
    has_special_considerations: str,
    number_of_passengers: int,
    travel_distance: int,
    number_of_stations: int,
    travel_time: int,
    wl_number: int,
    days_before_journey: int,
    threshold: float = 0.5,
):
    """
    Predict confirmation status for a single ticket.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained model pipeline loaded via `load_model()`.
    (remaining params) : feature values matching the training schema.

    Returns
    -------
    dict with keys:
        - label : str  ("Likely Confirm" or "Unlikely Confirm")
        - probability : float (0-1, confirmation probability)
        - prediction : int (1=Confirmed, 0=Not Confirmed)
    """
    # Build a single-row DataFrame matching training features
    input_data = pd.DataFrame([{
        "Class of Travel": class_of_travel,
        "Quota": quota,
        "Age of Passengers": age_of_passengers,
        "Booking Channel": booking_channel,
        "Train Type": train_type,
        "Holiday or Peak Season": holiday_or_peak_season,
        "Has Special Considerations": has_special_considerations,
        "Number of Passengers": number_of_passengers,
        "Travel Distance": travel_distance,
        "Number of Stations": number_of_stations,
        "Travel Time": travel_time,
        # NOTE: "Seat Availability" removed — not a model feature (data leakage)
        "WL Number": wl_number,
        "Days Before Journey": days_before_journey,
    }])

    # Ensure column order matches training
    input_data = input_data[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    # Predict probability then apply explicit threshold (default = 0.5)
    probability = float(pipeline.predict_proba(input_data)[0][1])  # P(Confirmed)
    prediction = int(probability >= threshold)

    label = "Likely Confirm" if prediction == 1 else "Unlikely Confirm"

    return {
        "label": label,
        "probability": round(probability, 4),
        "prediction": int(prediction),
    }


# ── Interactive CLI Demo ─────────────────────────────────────────────────────

def main():
    """Quick command-line demo of the prediction pipeline."""
    print("=" * 60)
    print("  PNR Waitlist Confirmation - Prediction Demo")
    print("=" * 60)

    pipeline = load_model()
    print("  [OK] Model loaded successfully\n")

    # Sample prediction with example values
    result = predict_confirmation(
        pipeline,
        class_of_travel="3AC",
        quota="General",
        age_of_passengers="Adult",
        booking_channel="IRCTC Website",
        train_type="Express",
        holiday_or_peak_season="No",
        has_special_considerations="No",
        number_of_passengers=2,
        travel_distance=800,
        number_of_stations=10,
        travel_time=12,
        wl_number=15,
        days_before_journey=30,
    )

    print("  -- Example Prediction --")
    print(f"    Prediction : {result['label']}")
    print(f"    Probability: {result['probability']:.2%}")
    print(f"    Raw class  : {result['prediction']}")
    print()


if __name__ == "__main__":
    main()

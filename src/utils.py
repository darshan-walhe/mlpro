import pandas as pd
import numpy as np
import os

# ─── Constants ───────────────────────────────────────────────────────────────

# Path configurations
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Railway Ticket Confirmation.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pnr_logistic_regression_pipeline.pkl")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Feature columns used for model training (after engineering)
# NOTE: 'Current Status' was REMOVED — it is data leakage
#       (100% correlated with the target variable)
CATEGORICAL_FEATURES = [
    "Class of Travel",
    "Quota",
    "Age of Passengers",
    "Booking Channel",
    "Train Type",
    "Holiday or Peak Season",
    "Has Special Considerations",
]

NUMERIC_FEATURES = [
    "Number of Passengers",
    "Travel Distance",
    "Number of Stations",
    "Travel Time",
    # NOTE: "Seat Availability" REMOVED — it perfectly separates waitlisted (=0)
    # from already-confirmed (>0) rows, causing 100%/0% extremes (data leakage).
    "WL Number",            # extracted numeric waitlist position
    "Days Before Journey",  # engineered: days between booking and journey
]

TARGET_COLUMN = "Confirmation Status"

# Valid values for Streamlit dropdowns
CLASS_OPTIONS = ["Sleeper", "1AC", "2AC", "3AC"]
QUOTA_OPTIONS = ["General", "Tatkal", "Premium Tatkal", "Ladies"]
# CURRENT_STATUS_OPTIONS removed — not used as feature (data leakage)
AGE_GROUP_OPTIONS = ["Adult", "Child", "Senior Citizen"]
BOOKING_CHANNEL_OPTIONS = ["IRCTC Website", "Mobile App", "Counter"]
TRAIN_TYPE_OPTIONS = ["Express", "Superfast", "Rajdhani", "Shatabdi"]
PEAK_SEASON_OPTIONS = ["Yes", "No"]

# ─── Data Loading ────────────────────────────────────────────────────────────

def load_dataset(filepath=None):
    """Load the railway ticket confirmation dataset from CSV."""
    if filepath is None:
        filepath = DATA_PATH
    df = pd.read_csv(filepath)
    return df


# ─── Feature Engineering ─────────────────────────────────────────────────────

def extract_waitlist_number(wl_series):
    """
    Extract the numeric waitlist position from strings like 'WL097'.
    Returns 0 for NaN / non-waitlisted entries (already confirmed or RAC).
    """
    def _parse(val):
        if pd.isna(val):
            return 0
        val = str(val).strip()
        if val.upper().startswith("WL"):
            try:
                return int(val[2:])
            except ValueError:
                return 0
        return 0
    return wl_series.apply(_parse)


def compute_days_before_journey(df):
    """
    Compute the number of days between Booking Date and Date of Journey.
    Both columns should be in a parseable date format.
    """
    journey = pd.to_datetime(df["Date of Journey"], errors="coerce")
    booking = pd.to_datetime(df["Booking Date"], errors="coerce")
    return (journey - booking).dt.days.fillna(0).astype(int)


def engineer_features(df):
    """
    Apply feature engineering to the raw dataframe:
      1. Extract numeric waitlist position
      2. Compute days before journey
      3. Create binary flag for special considerations
    Returns the dataframe with new columns added.
    """
    df = df.copy()

    # 1. Extract numeric WL position
    df["WL Number"] = extract_waitlist_number(df["Waitlist Position"])

    # 2. Days between booking and journey
    df["Days Before Journey"] = compute_days_before_journey(df)

    # 3. Binary flag for special considerations
    df["Has Special Considerations"] = df["Special Considerations"].notna().map(
        {True: "Yes", False: "No"}
    )

    return df


def encode_target(df, target_col=TARGET_COLUMN):
    """
    Encode the target variable:  'Confirmed' → 1,  'Not Confirmed' → 0
    """
    mapping = {"Confirmed": 1, "Not Confirmed": 0}
    return df[target_col].map(mapping)


# ─── Preprocessing Pipeline Helpers ──────────────────────────────────────────

def get_feature_matrix(df):
    """
    Return feature matrix X using only the defined feature columns.
    Assumes `engineer_features()` has already been called.
    """
    return df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def prepare_dataset(filepath=None):
    """
    Full pipeline: load → engineer → split into X, y.
    Returns (X, y, df_engineered)
    """
    df = load_dataset(filepath)
    df = engineer_features(df)
    X = get_feature_matrix(df)
    y = encode_target(df)
    return X, y, df

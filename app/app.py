"""
app.py — Streamlit Live Demo for PNR Waitlist Confirmation Prediction
=====================================================================
A polished, interactive UI for predicting whether a waitlisted Indian
Railways ticket will be confirmed, with a probability score.

Launch:
    streamlit run app/app.py
"""

import os
import sys
import streamlit as st

# ── Resolve import paths ─────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from predict import load_model, predict_confirmation
from utils import (
    CLASS_OPTIONS,
    QUOTA_OPTIONS,
    AGE_GROUP_OPTIONS,
    BOOKING_CHANNEL_OPTIONS,
    TRAIN_TYPE_OPTIONS,
    PEAK_SEASON_OPTIONS,
    MODEL_PATH,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PNR Confirmation Predictor",
    page_icon="🚂",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* --- Global --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* --- Header --- */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-header p {
        color: #6b7280;
        font-size: 1rem;
    }
    
    /* --- Result Cards --- */
    .result-card {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #86efac;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card.negative {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border-color: #fca5a5;
    }
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-prob {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .result-prob.positive { color: #16a34a; }
    .result-prob.negative { color: #dc2626; }
    
    /* --- Info Badge --- */
    .info-badge {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        border: 1px solid #bfdbfe;
    }

    /* --- Divider --- */
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #d1d5db, transparent);
        margin: 1.5rem 0;
    }
    
    /* --- Footer --- */
    .app-footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🚂 PNR Confirmation Predictor</h1>
    <p>Predict whether your waitlisted ticket will get confirmed</p>
    <span class="info-badge">Powered by Logistic Regression</span>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_model_v2():
    """Load and cache the trained pipeline."""
    return load_model()

try:
    pipeline = get_model_v2()
except FileNotFoundError:
    st.error(
        "⚠️ **Model not found!** "
        "Please run `python src/train.py` first to train and save the model."
    )
    st.stop()

# ─── Input Form ──────────────────────────────────────────────────────────────
st.subheader("📋 Enter Ticket Details")

col1, col2 = st.columns(2)

with col1:
    class_of_travel = st.selectbox(
        "Class of Travel", CLASS_OPTIONS, index=2, help="Select the coach class"
    )
    quota = st.selectbox(
        "Quota", QUOTA_OPTIONS, index=0, help="Booking quota type"
    )
    train_type = st.selectbox(
        "Train Type", TRAIN_TYPE_OPTIONS, index=0,
        help="Type of train"
    )

with col2:
    age_group = st.selectbox(
        "Passenger Age Group", AGE_GROUP_OPTIONS, index=0,
        help="Age category of the passenger"
    )
    booking_channel = st.selectbox(
        "Booking Channel", BOOKING_CHANNEL_OPTIONS, index=0,
        help="How the ticket was booked"
    )
    peak_season = st.selectbox(
        "Holiday / Peak Season?", PEAK_SEASON_OPTIONS, index=1,
        help="Is the journey during a holiday or peak season?"
    )

has_special = st.selectbox(
    "Special Considerations?", ["No", "Yes"], index=0,
    help="Senior citizen, defense quota, etc."
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🔢 Numeric Details")

col3, col4, col5 = st.columns(3)

with col3:
    wl_number = st.number_input(
        "Waitlist Position", min_value=0, max_value=500, value=15, step=1,
        help="Your current waitlist number (0 if not waitlisted)"
    )
    num_passengers = st.number_input(
        "Number of Passengers", min_value=1, max_value=10, value=2, step=1,
    )

with col4:
    travel_distance = st.number_input(
        "Travel Distance (km)", min_value=10, max_value=5000, value=800, step=50,
    )
    num_stations = st.number_input(
        "Number of Stations", min_value=1, max_value=50, value=10, step=1,
    )

with col5:
    travel_time = st.number_input(
        "Travel Time (hours)", min_value=1, max_value=72, value=12, step=1,
    )

days_before = st.slider(
    "Days Before Journey (booking advance)",
    min_value=0, max_value=120, value=30,
    help="How many days in advance was the ticket booked?"
)

# ─── Prediction ──────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

predict_btn = st.button("🔮 Predict Confirmation", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("Analyzing…"):
        result = predict_confirmation(
            pipeline,
            class_of_travel=class_of_travel,
            quota=quota,
            age_of_passengers=age_group,
            booking_channel=booking_channel,
            train_type=train_type,
            holiday_or_peak_season=peak_season,
            has_special_considerations=has_special,
            number_of_passengers=num_passengers,
            travel_distance=travel_distance,
            number_of_stations=num_stations,
            travel_time=travel_time,
            wl_number=wl_number,
            days_before_journey=days_before,
        )
        st.session_state['pred_result'] = result

if 'pred_result' in st.session_state:
    result = st.session_state['pred_result']
    prob = result["probability"]
    is_positive = result["prediction"] == 1
    card_class = "" if is_positive else " negative"
    prob_class = "positive" if is_positive else "negative"
    emoji = "✅" if is_positive else "❌"

    st.markdown(f"""
    <div class="result-card{card_class}">
        <div class="result-label">{emoji} {result['label']}</div>
        <div class="result-prob {prob_class}">{prob:.0%}</div>
        <p style="color: #6b7280; margin: 0;">confirmation probability</p>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar visualization
    st.progress(prob)

    # Details expander
    with st.expander("📊 Prediction Details"):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.metric("Raw Probability", f"{prob:.4f}")
            st.metric("Predicted Class", result["prediction"])
        with detail_col2:
            st.metric("Label", result["label"])
            st.metric("Model", "Logistic Regression")
            
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("☁️ Upload Project Data")
    if st.button("📤 Upload to Nextcloud", use_container_width=True):
        with st.spinner("Uploading to Nextcloud..."):
            try:
                import sys
                import io
                from contextlib import redirect_stdout
                from dotenv import load_dotenv
                
                # Make sure the project root is in path to import upload_to_nextcloud
                if PROJECT_ROOT not in sys.path:
                    sys.path.insert(0, PROJECT_ROOT)
                    
                # Load .env variables so NEXTCLOUD_SHARE_TOKEN is available
                load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
                
                from upload_to_nextcloud import upload_to_nextcloud
                
                # Capture the print statements from the upload script
                f = io.StringIO()
                with redirect_stdout(f):
                    upload_to_nextcloud()
                    
                st.success("Successfully uploaded to Nextcloud!")
                with st.expander("Show Logs"):
                    st.code(f.getvalue())
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p>ML Mini Project — PNR Waitlist Ticket Confirmation Prediction</p>
    <p>Built with Streamlit • Scikit-learn • Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

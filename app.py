import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. Load Assets (Cached) ---
@st.cache_resource
def load_artifacts():
    """
    Load the trained XGBoost model and RobustScaler.
    """
    try:
        model_ = joblib.load('fraud_model.pkl')
        scaler_ = joblib.load('scaler.pkl')
        return model_, scaler_
    except FileNotFoundError as e:
        st.error(f"Critical Error: Model files not found. {e}")
        return None, None

model, scaler = load_artifacts()

# --- 3. UI Layout ---
st.title("🛡️ FraudGuard AI")
st.markdown("### Real-time Transaction Risk Assessment")
st.markdown("---")

if model is not None and scaler is not None:

    # --- Input Section ---
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)

    with col2:
        time_val = st.number_input("Time (Seconds from start)", min_value=0.0, value=0.0, step=1.0)

    st.markdown("#### Behavioral Analysis (Simulation)")
    # V14 is the top feature from SHAP analysis
    v14 = st.slider(
        "V14 (Principal Component - Behavioral Score)", 
        min_value=-20.0, 
        max_value=5.0, 
        value=0.0,
        help="Lower values (negative) indicate highly anomalous behavior based on historical fraud patterns."
    )

    # --- Prediction Logic ---
    if st.button("Analyze Transaction", type="primary"):
        
        # 1. Initialize feature vector (30 features)
        # Order: Amount, Time, V1...V28 (based on our training pipeline)
        features = np.zeros(30)

        # 2. Preprocessing
        try:
            # RobustScaler expects 2D array
            scaled_amount = scaler.transform([[amount]])[0][0]
            scaled_time = scaler.transform([[time_val]])[0][0]
        except Exception as e:
            st.error(f"Preprocessing Error: {e}")
            st.stop()

        # 3. Feature Assignment
        # Note: Ensure this index matches your X_train columns order.
        # In our pipeline: 0=Amount, 1=Time, 15=V14 (approximate index for V14)
        features[0] = scaled_amount
        features[1] = scaled_time
        features[15] = v14 

        # 4. Inference
        prediction_prob = model.predict_proba([features])[0][1]
        
        # 5. Result Display
        st.markdown("---")
        st.subheader("Risk Analysis Report")

        # Progress bar (Fixed float type issue)
        st.progress(float(prediction_prob))
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="Fraud Probability", value=f"{prediction_prob:.2%}")

        with col_res2:
            # --- TRAFFIC LIGHT LOGIC ---
            # Thresholds based on Business Cost Analysis
            THRESH_REVIEW = 0.50  # Suspicious
            THRESH_BLOCK = 0.90   # High Certainty
            
            if prediction_prob > THRESH_BLOCK:
                st.error("STATUS: BLOCKED")
                st.markdown("**Critical Risk.** Transaction automatically declined.")
                
            elif prediction_prob > THRESH_REVIEW:
                st.warning("STATUS: MANUAL REVIEW")
                st.markdown("**Suspicious Activity.** 3D-Secure verification sent to user.")
                
            else:
                st.success("STATUS: APPROVED")
                st.markdown("Transaction pattern is normal.")
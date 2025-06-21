import streamlit as st
import pandas as pd
import joblib
import os

# ====================================================================
# 1. APP CONFIGURATION AND UI
# ====================================================================

st.set_page_config(
    page_title="SWRO Performance Predictor",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 SWRO Desalination: Predictive Performance Modeler")
st.write("""
This application predicts the final salt rejection percentage of a Seawater Reverse Osmosis (SWRO) 
system based on its initial operational data. It uses a machine learning model trained on experimental data 
from sinusoidal stress tests.
""")

st.info("""
**Instructions:**
1. Prepare a CSV file containing the first 10-20 rows of operational data.
2. The file must have the 10 data columns in the correct order (e.g., Feed flow, Brine flow, etc.).
3. Upload the CSV file using the button below. The model will use the first 5 valid data points to make a prediction.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your operational data (CSV format)", type=["csv"])

# ====================================================================
# 2. MODEL AND DATA PROCESSING LOGIC
# ====================================================================

# --- Load the trained model ---
# Construct an absolute path to the model file.
# This version assumes 'model.joblib' is in the SAME directory as this 'app.py' script.
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model.joblib')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Error: Model file not found at '{model_path}'. Please ensure the model has been trained and saved.")
    model = None

# --- Define the full data processing pipeline as a function ---
def process_input_data(df):
    """
    Takes a raw DataFrame from a user upload and runs the full
    feature engineering pipeline to prepare it for the model.
    """
    st.write("---")
    st.subheader("Processing Input Data")
    
    # --- 1. Enforce Schema ---
    final_columns = [
        'feed_flow_l_min', 'brine_flow_l_min', 'feed_pressure_bar',
        'brine_pressure_bar', 'permeate_flow_l_min', 'permeate_salinity_ppm',
        'brine_salinity_ppm', 'temp_c', 'recovery_percent', 'salt_rejection_percent'
    ]
    if len(df.columns) < 10:
        st.error(f"Error: The uploaded file must have at least 10 columns. It has {len(df.columns)}.")
        return None

    df_processed = df.iloc[:, :10].copy()
    df_processed.columns = final_columns
    st.write("✅ Renamed columns to standard schema.")

    # --- 2. Fix Data Types ---
    for col in final_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed.dropna(inplace=True)
    st.write("✅ Converted data to numeric types.")

    if len(df_processed) < 5:
        st.error("Error: After cleaning and feature engineering, the file has fewer than 5 valid data rows. Please provide more initial data.")
        return None

    # --- 3. Feature Engineering ---
    df_processed['pressure_diff_bar'] = df_processed['feed_pressure_bar'] - df_processed['brine_pressure_bar']
    window_size = 5
    cols_for_rolling = ['feed_pressure_bar', 'permeate_flow_l_min', 'pressure_diff_bar']
    for col in cols_for_rolling:
        df_processed[f'{col}_roll_mean'] = df_processed[col].rolling(window=window_size).mean()
        df_processed[f'{col}_roll_std'] = df_processed[col].rolling(window=window_size).std()
    
    # Drop rows with NaNs created by rolling features
    df_processed.dropna(inplace=True)
    st.write("✅ Engineered features (pressure differential, rolling stats).")

    if df_processed.empty:
        st.error("Error: Not enough data to create rolling features. Please provide at least 10-15 rows.")
        return None
    
    # --- 4. Flatten into a single vector ---
    # We take the first 5 valid rows after processing, as this represents the history our model was trained on.
    history_df = df_processed.head(5)
    
    # Check if we have exactly 5 rows to flatten
    if len(history_df) < 5:
        st.error(f"Error: Not enough data to create the full 5-step history vector. Only {len(history_df)} rows are available.")
        return None
        
    feature_vector = history_df.values.flatten()
    st.write("✅ Flattened 5-step history into a feature vector for prediction.")
    
    return feature_vector

# ====================================================================
# 3. PREDICTION AND OUTPUT
# ====================================================================

if uploaded_file is not None and model is not None:
    try:
        # Read the uploaded CSV file
        df_input = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df_input.head())

        # Process the data using our pipeline
        feature_vector = process_input_data(df_input)

        if feature_vector is not None:
            # Reshape the vector for a single prediction
            feature_vector = feature_vector.reshape(1, -1)
            
            # Make a prediction
            prediction = model.predict(feature_vector)
            predicted_value = prediction[0]

            # --- Display the result ---
            st.write("---")
            st.subheader("Prediction Result")
            st.metric(
                label="Predicted Final Salt Rejection (%)", 
                value=f"{predicted_value:.4f}"
            )

            # --- Display Interpretation ---
            if predicted_value >= 99.4:
                st.success("✅ **Status: Normal.** The predicted performance is excellent.")
            elif predicted_value >= 99.0:
                st.warning("⚠️ **Status: Warning.** The model predicts moderate performance degradation. Monitoring is advised.")
            else:
                st.error("🚨 **Status: Alert.** The model predicts significant performance degradation. Proactive maintenance may be required.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

elif uploaded_file is not None and model is None:
    st.warning("Cannot proceed with prediction because the model is not loaded.")

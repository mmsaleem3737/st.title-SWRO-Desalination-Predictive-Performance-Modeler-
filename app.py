import streamlit as st
import pandas as pd
import joblib
import os

# ====================================================================
# 1. APP CONFIGURATION & STATIC CONTENT
# ====================================================================

st.set_page_config(
    page_title="SWRO Performance Predictor",
    page_icon="🌊",
    layout="wide"
)

# --- Sidebar Content ---
with st.sidebar:
    st.title("Project Information")
    st.write("""
    This project demonstrates an end-to-end data science workflow to predict performance 
    degradation in Seawater Reverse Osmosis (SWRO) systems.
    """)

    with st.expander("ℹ️ About this Project"):
        st.write("""
        The goal is to provide an early warning for membrane fouling by predicting the final salt rejection 
        percentage based on initial operational data. The project followed an iterative process:
        
        1.  **Initial Classification:** A model with 100% accuracy was found to be too simple, only distinguishing between stable and unstable tests.
        2.  **Generalized Regression:** A model trained on all data failed to generalize due to fundamentally different experimental conditions.
        3.  **Specialized Regression:** The final, successful model is a **hyper-specialized `RandomForestRegressor`** trained exclusively on standard sinusoidal stress test data, making it a robust proof-of-concept for a real-world predictive tool.
        """)

    with st.expander("📈 Model Performance"):
        st.write("""
        The final model was evaluated using **Leave-One-Out Cross-Validation** on the three available sinusoidal experiments.
        - **Metric:** Mean Absolute Error (MAE)
        - **Average MAE:** `0.8322`
        - **Std. Deviation of MAE:** `0.3866`
        
        This means, on average, the model's prediction for the final salt rejection percentage is off by less than 1 percentage point, demonstrating high accuracy for its specific task.
        """)
        
    with st.expander("📚 Data Source"):
        st.write("""
        This project uses the "Performance Data of a SWRO arising from Wave Powered Desalinisation" dataset.
        - **Authors:** Frost, C., Das, T. K.
        - **Institution:** Queen's University Belfast
        - **DOI:** `10.17632/hws49dsfvc.1`
        - [Link to Dataset](https://data.mendeley.com/datasets/hws49dsfvc/1)
        """)
    
    st.info("App developed by [Your Name Here]") # Feel free to change this!

# --- Main Page Content ---
st.title("🌊 SWRO Desalination: Predictive Performance Modeler")
st.write("""
Upload a CSV file with the initial operational data from an SWRO system to predict its final salt rejection performance. 
The model is specialized for systems undergoing **sinusoidal stress tests**.
""")

# --- Create a sample template for download ---
@st.cache_data
def create_template_df():
    template_data = {
        'feed_flow_l_min': [8.0], 'brine_flow_l_min': [7.31], 'feed_pressure_bar': [53.25],
        'brine_pressure_bar': [52.7], 'permeate_flow_l_min': [1.01], 'permeate_salinity_ppm': [123.01],
        'brine_salinity_ppm': [40833], 'temp_c': [25.4], 'recovery_percent': [12.625], 
        'salt_rejection_percent': [99.6264]
    }
    df = pd.DataFrame(template_data)
    return df.to_csv(index=False).encode('utf-8')

template_csv = create_template_df()

# --- Display Instructions and Uploader ---
with st.expander("Instructions & File Format", expanded=True):
    st.write("""
    1.  Click the **Download Template** button to get a CSV file with the required columns.
    2.  Open the template and replace the sample row with at least 10-15 rows of your own data.
    3.  Save the file and upload it below.
    """)

col1, col2 = st.columns([3, 1]) # Make the uploader wider
with col1:
    uploaded_file = st.file_uploader("Upload your operational data", type=["csv"], label_visibility="collapsed")
with col2:
    st.download_button(
        label="Download Template",
        data=template_csv,
        file_name="swro_template.csv",
        mime="text/csv",
    )


# ====================================================================
# 2. MODEL AND DATA PROCESSING LOGIC
# ====================================================================

# --- Load the trained model ---
def load_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'model.joblib')
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure the model has been trained and saved in the same directory as the app.")
        return None

model = load_model()

# --- Define the full data processing pipeline as a function ---
def process_input_data(df):
    st.write("---")
    st.subheader("⚙️ Processing Input Data")
    
    final_columns = [
        'feed_flow_l_min', 'brine_flow_l_min', 'feed_pressure_bar',
        'brine_pressure_bar', 'permeate_flow_l_min', 'permeate_salinity_ppm',
        'brine_salinity_ppm', 'temp_c', 'recovery_percent', 'salt_rejection_percent'
    ]
    if len(df.columns) < 10:
        st.error(f"Error: Uploaded file must have at least 10 columns. It has {len(df.columns)}.")
        return None

    df_processed = df.iloc[:, :10].copy()
    df_processed.columns = final_columns
    
    for col in final_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed.dropna(inplace=True)

    if len(df_processed) < 5:
        st.error(f"Error: After cleaning, file has only {len(df_processed)} valid data rows. At least 5 are required.")
        return None

    df_processed['pressure_diff_bar'] = df_processed['feed_pressure_bar'] - df_processed['brine_pressure_bar']
    window_size = 5
    cols_for_rolling = ['feed_pressure_bar', 'permeate_flow_l_min', 'pressure_diff_bar']
    for col in cols_for_rolling:
        df_processed[f'{col}_roll_mean'] = df_processed[col].rolling(window=window_size, min_periods=1).mean()
        df_processed[f'{col}_roll_std'] = df_processed[col].rolling(window=window_size, min_periods=1).std()
    
    df_processed.dropna(inplace=True)
    st.write("✅ Feature engineering complete.")

    if len(df_processed) < 5:
        st.error("Error: Not enough data to create the full 5-step history vector. Please provide more initial data rows.")
        return None
        
    history_df = df_processed.head(5)
    feature_vector = history_df.values.flatten()
    st.write("✅ Feature vector created for prediction.")
    
    return feature_vector

# ====================================================================
# 3. PREDICTION AND OUTPUT
# ====================================================================

if uploaded_file is not None and model is not None:
    try:
        # Read the uploaded CSV file, assuming it now has a header
        df_input = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df_input.head())

        feature_vector = process_input_data(df_input)

        if feature_vector is not None:
            feature_vector = feature_vector.reshape(1, -1)
            
            prediction = model.predict(feature_vector)
            predicted_value = prediction[0]

            st.write("---")
            st.subheader("📈 Prediction Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Predicted Final Salt Rejection (%)", 
                    value=f"{predicted_value:.4f}"
                )
            
            with col2:
                if predicted_value >= 99.4:
                    st.success("✅ **Status: Normal.** The predicted performance is excellent.")
                elif predicted_value >= 99.0:
                    st.warning("⚠️ **Status: Warning.** Moderate performance degradation predicted. Monitoring is advised.")
                else:
                    st.error("🚨 **Status: Alert.** Significant performance degradation predicted. Proactive maintenance may be required.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
elif uploaded_file is not None and model is None:
    st.warning("Cannot proceed with prediction because the model is not loaded.")

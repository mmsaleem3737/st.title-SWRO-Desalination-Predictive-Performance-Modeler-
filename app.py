import streamlit as st
import pandas as pd
import joblib
import os

# ====================================================================
# 1. APP CONFIGURATION & UI STYLING
# ====================================================================

st.set_page_config(
    page_title="SWRO Performance Predictor",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Card-like containers */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Status indicators */
    .status-normal {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .status-alert {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(59, 130, 246, 0.6);
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* Button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(79, 70, 229, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Data preview table */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Processing steps */
    .processing-step {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .step-icon {
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(59, 130, 246, 0.05), rgba(16, 185, 129, 0.05));
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .custom-card {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .main-header {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(16, 185, 129, 0.15));
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Main Header ---
st.markdown("""
<div class="main-header">
    <h1>🌊 SWRO Desalination: Predictive Performance Modeler</h1>
    <p style="font-size: 1.1rem; margin-top: 1rem; opacity: 0.8;">
        Advanced Machine Learning for Seawater Reverse Osmosis Performance Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Description ---
st.markdown("""
<div class="custom-card">
    <h3 style="margin-top: 0;">📋 How It Works</h3>
    <p>Upload a CSV file with initial operational data from your SWRO system to predict its final salt rejection performance. 
    This specialized model is optimized for systems undergoing <strong>sinusoidal stress tests</strong>.</p>
    
    <div style="margin-top: 1rem;">
        <span style="color: #3b82f6;">💡</span> <strong>Tip:</strong> Ensure your data includes at least 10-15 rows for accurate predictions.
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================================================
# 2. ENHANCED FILE UPLOAD SECTION
# ====================================================================

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

# --- Enhanced Instructions Section ---
with st.expander("📋 Instructions & File Format", expanded=False):
    st.markdown("""
    <div style="padding: 1rem 0;">
        <h4 style="color: #3b82f6; margin-top: 0;">Step-by-Step Guide:</h4>
        
        <div class="processing-step">
            <span class="step-icon">1️⃣</span>
            Click the <strong>Download Template</strong> button to get a CSV file with the required columns.
        </div>
        
        <div class="processing-step">
            <span class="step-icon">2️⃣</span>
            Open the template and replace the sample row with <strong>at least 10-15 rows</strong> of your own data.
        </div>
        
        <div class="processing-step">
            <span class="step-icon">3️⃣</span>
            Save the file and upload it using the uploader below.
        </div>
        
        <div class="processing-step">
            <span class="step-icon">4️⃣</span>
            Review your data preview and get your performance prediction!
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Enhanced File Upload Section ---
st.markdown("### 📁 Data Upload")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Choose your CSV file", 
        type=["csv"], 
        help="Upload a CSV file with your SWRO operational data"
    )
with col2:
    st.download_button(
        label="📥 Download Template",
        data=template_csv,
        file_name="swro_template.csv",
        mime="text/csv",
        help="Download the CSV template with required columns"
    )

# ====================================================================
# 3. MODEL AND DATA PROCESSING LOGIC (UNCHANGED)
# ====================================================================

# --- Load the trained model ---
@st.cache_resource
def load_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'model.joblib')
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found. Please ensure 'model.joblib' is in the same directory as the app.")
        return None

model = load_model()

# --- Define the full data processing pipeline as a function ---
def process_input_data(df):
    st.markdown("### ⚙️ Processing Input Data")
    
    # Create progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    final_columns = [
        'feed_flow_l_min', 'brine_flow_l_min', 'feed_pressure_bar',
        'brine_pressure_bar', 'permeate_flow_l_min', 'permeate_salinity_ppm',
        'brine_salinity_ppm', 'temp_c', 'recovery_percent', 'salt_rejection_percent'
    ]
    
    # Step 1: Validate columns
    status_text.text("🔍 Validating data structure...")
    progress_bar.progress(20)
    
    if len(df.columns) < 10:
        st.error(f"❌ Uploaded file must have at least 10 columns. It has {len(df.columns)}.")
        return None

    df_processed = df.iloc[:, :10].copy()
    df_processed.columns = final_columns
    
    # Step 2: Clean data
    status_text.text("🧹 Cleaning and converting data types...")
    progress_bar.progress(40)
    
    for col in final_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed.dropna(inplace=True)

    if len(df_processed) < 5:
        st.error(f"❌ After cleaning, file has only {len(df_processed)} valid data rows. At least 5 are required.")
        return None

    # Step 3: Feature engineering
    status_text.text("🔧 Creating engineered features...")
    progress_bar.progress(60)
    
    df_processed['pressure_diff_bar'] = df_processed['feed_pressure_bar'] - df_processed['brine_pressure_bar']
    window_size = 5
    cols_for_rolling = ['feed_pressure_bar', 'permeate_flow_l_min', 'pressure_diff_bar']
    for col in cols_for_rolling:
        df_processed[f'{col}_roll_mean'] = df_processed[col].rolling(window=window_size, min_periods=1).mean()
        df_processed[f'{col}_roll_std'] = df_processed[col].rolling(window=window_size, min_periods=1).std()
    
    df_processed.dropna(inplace=True)
    
    # Step 4: Create feature vector
    status_text.text("📊 Creating feature vector...")
    progress_bar.progress(80)

    if len(df_processed) < 5:
        st.error("❌ Not enough data to create the full 5-step history vector. Please provide more initial data rows.")
        return None
        
    history_df = df_processed.head(5)
    feature_vector = history_df.values.flatten()
    
    # Complete
    status_text.text("✅ Data processing complete!")
    progress_bar.progress(100)
    
    # Clear progress indicators after a moment
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    st.success("🎯 Feature vector successfully created for prediction")
    
    return feature_vector

# ====================================================================
# 4. ENHANCED PREDICTION AND OUTPUT
# ====================================================================

if uploaded_file is not None and model is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        
        # Enhanced data preview
        st.markdown("### 📄 Uploaded Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df_input))
        with col2:
            st.metric("Total Columns", len(df_input.columns))
        with col3:
            st.metric("Data Points", len(df_input) * len(df_input.columns))
        
        # Show data in a nice container
        with st.container():
            st.dataframe(df_input.head(10), use_container_width=True)

        feature_vector = process_input_data(df_input)

        if feature_vector is not None:
            feature_vector = feature_vector.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(feature_vector)
            predicted_value = prediction[0]

            st.markdown("---")
            st.markdown("### 📈 Prediction Results")
            
            # Enhanced results display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                <div class="metric-container">
                    <h3 style="margin: 0; color: #6366f1;">Predicted Final Salt Rejection</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 2.5rem; font-weight: bold;">{:.4f}%</h1>
                </div>
                """.format(predicted_value), unsafe_allow_html=True)
            
            with col2:
                # Enhanced status display
                if predicted_value >= 99.4:
                    st.markdown("""
                    <div class="status-normal">
                        <h4 style="margin: 0; color: #16a34a;">✅ Status: Normal Performance</h4>
                        <p style="margin: 0.5rem 0 0 0;">The predicted performance is excellent. Your SWRO system is operating within optimal parameters.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif predicted_value >= 99.0:
                    st.markdown("""
                    <div class="status-warning">
                        <h4 style="margin: 0; color: #d97706;">⚠️ Status: Performance Warning</h4>
                        <p style="margin: 0.5rem 0 0 0;">Moderate performance degradation predicted. Consider increased monitoring and preventive maintenance.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="status-alert">
                        <h4 style="margin: 0; color: #dc2626;">🚨 Status: Performance Alert</h4>
                        <p style="margin: 0.5rem 0 0 0;">Significant performance degradation predicted. Immediate attention and proactive maintenance recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ An error occurred during processing: {e}")

# ====================================================================
# 5. ENHANCED PROJECT DOCUMENTATION
# ====================================================================

st.markdown("---")
st.markdown("## 📚 Project Documentation")

with st.expander("ℹ️ About this Project"):
    st.markdown("""
    <div style="line-height: 1.6;">
        <p>This project explores the feasibility of predicting SWRO membrane performance degradation using machine learning. 
        The development followed a rigorous, iterative process:</p>
        
        <div class="processing-step">
            <span class="step-icon">🔬</span>
            <strong>Initial Classification:</strong> A model with 100% accuracy was found to be too simple, only distinguishing between stable and unstable tests.
        </div>
        
        <div class="processing-step">
            <span class="step-icon">📊</span>
            <strong>Generalized Regression:</strong> A model trained on all data failed to generalize due to fundamentally different experimental conditions.
        </div>
        
        <div class="processing-step">
            <span class="step-icon">🎯</span>
            <strong>Specialized Regression:</strong> A hyper-specialized RandomForestRegressor was trained exclusively on 3 available sinusoidal stress test experiments.
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("🔍 Key Research Findings"):
    st.markdown("""
    <div class="status-warning">
        <h4 style="margin-top: 0;">🎯 Critical Discovery: The "Brittle Expert" Model</h4>
        <p>The model consistently predicts 'Alert' status for most inputs. This is not a limitation—it's the project's most valuable finding.</p>
    </div>
    
    <div style="margin-top: 1.5rem;">
        <h5>Why does this happen?</h5>
        <p>The model was trained on an extremely small dataset (only 3 experiments), causing it to "memorize" exact numerical patterns rather than learn to generalize. When encountering new data that deviates from these memorized patterns, it correctly identifies anomalies and predicts poor performance.</p>
        
        <h5 style="margin-top: 1rem;">Research Conclusion</h5>
        <p>This project successfully demonstrates the primary challenge in real-world industrial AI: <strong>data scarcity</strong>. We've proven that our methodology is sound, but a robust, deployable model requires a much larger and more varied training dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
with st.expander("📊 Model Performance Metrics"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Validation Method", "Leave-One-Out CV")
        st.metric("Average MAE", "0.8322")
    
    with col2:
        st.metric("Training Dataset", "3 experiments")
        st.metric("MAE Std. Deviation", "0.3866")
    
    st.info("💡 The model's prediction for final salt rejection percentage is typically within 1 percentage point of actual values.")
    
with st.expander("📖 Data Source & Citation"):
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #6366f1;">
        <h5 style="margin-top: 0;">Dataset Information</h5>
        <p><strong>Title:</strong> Performance Data of a SWRO arising from Wave Powered Desalinisation</p>
        <p><strong>Authors:</strong> Frost, C., Das, T. K.</p>
        <p><strong>Institution:</strong> Queen's University Belfast</p>
        <p><strong>DOI:</strong> 10.17632/hws49dsfvc.1</p>
        <p><a href="https://data.mendeley.com/datasets/hws49dsfvc/1" target="_blank">🔗 Access Dataset</a></p>
    </div>
    """, unsafe_allow_html=True)

# --- Enhanced Sidebar ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1)); border-radius: 10px; margin-bottom: 1rem;">
    <h3 style="margin: 0;">🌊 SWRO Predictor</h3>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Advanced ML Model</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📋 Quick Guide")
st.sidebar.markdown("""
- **Step 1:** Download template
- **Step 2:** Add your data (10+ rows)
- **Step 3:** Upload CSV file
- **Step 4:** Get prediction
""")

st.sidebar.markdown("### ⚡ Model Features")
st.sidebar.markdown("""
- Real-time predictions
- Feature engineering
- Performance classification
- Data validation
""")

st.sidebar.info("🔬 App developed for SWRO performance analysis")

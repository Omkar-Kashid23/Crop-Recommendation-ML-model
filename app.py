import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -------------------------
# Load Model Function
# -------------------------
@st.cache_resource
def load_model():
    """Load trained ML model from file."""
    model_path = os.path.join(os.getcwd(), "Best_crop_recommendation_system_ML_Model.pkl")
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure 'Best_crop_recommendation_system_ML_Model.pkl' is in the project root.")
        st.stop()
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


# -------------------------
# App Configuration
# -------------------------
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 Smart Crop Recommendation System")
st.markdown("#### Predict the most suitable crop based on soil and environmental conditions.")

model = load_model()


# -------------------------
# Sidebar Info Section
# -------------------------
with st.sidebar:
    st.header("ℹ️ About This App")
    st.write(
        """
        This application uses a **Machine Learning model** trained on agricultural data 
        to recommend the most suitable crop based on soil nutrients and weather parameters.
        """
    )
    st.divider()
    st.header("⚙️ Model Info")
    st.success("✅ Model loaded successfully!")
    st.write(f"**Model Type:** {type(model).__name__}")
    if hasattr(model, "classes_"):
        st.write(f"**Target Classes:** {list(model.classes_)}")
    if hasattr(model, "get_params"):
        st.write("**Key Parameters:**")
        params = model.get_params()
        for k, v in list(params.items())[:6]:  # show top 6 params
            st.write(f"- {k}: `{v}`")


# -------------------------
# Main Input Section
# -------------------------
tab1, tab2 = st.tabs(["📤 Batch input (CSV upload)", "✍️ Manual input (single sample)"])

# ---- Tab 1: CSV Upload ----
with tab1:
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            if st.button("Run prediction", key="csv_predict"):
                preds = model.predict(df)
                df["Predicted Crop"] = preds
                st.success("✅ Predictions complete!")
                st.dataframe(df, use_container_width=True)
                csv_download = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download predictions as CSV",
                    data=csv_download,
                    file_name="crop_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ---- Tab 2: Manual Input ----
with tab2:
    st.subheader("Enter Values Manually")

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
        K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
    with col2:
        P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    with col3:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=100.0)

    if st.button("Predict Crop"):
        try:
            sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(sample)
            st.success(f"🌱 Recommended Crop: **{prediction[0]}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -------------------------
# Footer
# -------------------------
st.divider()
st.caption("🚀 Developed by Omkar | Powered by Streamlit & scikit-learn")

"""
Production-ready Streamlit app to serve a pickled ML model for crop recommendation.
Includes Dockerfile and requirements.txt for deployment.
"""

import io
import os
import sys
import json
import pickle
import logging
from typing import List, Optional, Tuple, Any, Dict

import pandas as pd
import numpy as np
import streamlit as st

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "Best_crop_recommendation_system_ML_Model.pkl")
APP_TITLE = "Crop Recommendation — Production-ready Streamlit"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("crop_recommender_app")


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load model from disk using pickle with proper error handling."""
    logger.info(f"Loading model from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
    except Exception as e:
        logger.exception("Failed to load model via pickle.load")
        raise

    logger.info("Model loaded successfully")
    return model


def try_get_feature_names(model) -> Optional[List[str]]:
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                try:
                    return list(step.get_feature_names_out())
                except Exception:
                    continue
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    if isinstance(model, dict):
        if "feature_names" in model and isinstance(model["feature_names"], (list, tuple)):
            return list(model["feature_names"])

    return None


def validate_dataframe(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    return df[feature_names]


def predict(model, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
    except Exception:
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df.values)
        except Exception:
            probs = None

    try:
        preds = model.predict(df)
    except Exception:
        preds = model.predict(df.values)

    return preds, probs


def render_manual_input(feature_names: List[str]) -> pd.DataFrame:
    st.subheader("Manual input — single sample")
    with st.form(key="manual_input_form"):
        row = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            label = feat
            val = cols[i % 2].number_input(label, value=0.0, format="%.4f", key=f"num_{i}")
            row[feat] = val
        submitted = st.form_submit_button("Predict")

    if submitted:
        df = pd.DataFrame([row])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        return df

    return pd.DataFrame(columns=feature_names)


def render_csv_uploader() -> Optional[pd.DataFrame]:
    st.subheader("Batch input — upload CSV")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df)} rows from uploaded CSV")
            st.dataframe(df.head())
            return df
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return None
    return None


def safe_predict_and_show(model, df: pd.DataFrame, feature_names: Optional[List[str]] = None):
    if df is None or df.empty:
        st.info("No input to predict.")
        return

    if feature_names is not None:
        try:
            df = validate_dataframe(df, feature_names)
        except ValueError as e:
            st.error(str(e))
            return

    try:
        preds, probs = predict(model, df)
    except Exception as e:
        st.exception(e)
        return

    out = pd.DataFrame({"prediction": preds})
    if probs is not None:
        try:
            if probs.ndim == 2:
                out["confidence"] = np.round(probs.max(axis=1), 4)
        except Exception:
            pass

    st.subheader("Predictions")
    st.dataframe(out)
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions as CSV", csv, file_name="predictions.csv", mime="text/csv")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.write("This app loads a pickled model and allows single or batch predictions.")

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    feature_names = try_get_feature_names(model)
    if feature_names is None:
        st.warning("Could not detect feature names automatically.")

    left, right = st.columns([3, 2])

    with left:
        csv_df = render_csv_uploader()
        manual_df = pd.DataFrame()
        if feature_names is not None and len(feature_names) > 0:
            manual_df = render_manual_input(feature_names)

        input_df = csv_df if csv_df is not None else manual_df
        if st.button("Run prediction"):
            safe_predict_and_show(model, input_df, feature_names)

    with right:
        st.header("Model Info")
        st.write(type(model))
        st.write(feature_names)


if __name__ == "__main__":
    main()

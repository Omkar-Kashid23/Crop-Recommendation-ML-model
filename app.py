"""
Production-ready Streamlit app to serve a pickled ML model for crop recommendation.

File: streamlit_crop_recommender_app.py
Path: place next to your model or change MODEL_PATH.

Features:
- Safe, cached model loading
- Dynamic input UI based on model's feature names when available
- CSV upload for batch predictions
- Single-row manual input form for interactive use
- Prediction display with confidence (if available)
- Basic input validation and logging
- Helpful instructions and developer notes for production deployment

Notes:
- This app will attempt to read feature names from the model (feature_names_in_ or pipeline named steps).
- If it cannot detect features, it prompts user to upload a sample CSV or enter feature names manually.

"""

import io
import os
import sys
import json
import logging
from typing import List, Optional, Tuple, Any, Dict

import pandas as pd
import numpy as np
import streamlit as st

# Use joblib to load pickles created with sklearn pipelines/models
import joblib

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/data/Best_crop_recommendation_system_ML_Model.pkl")
APP_TITLE = "Crop Recommendation — Production-ready Streamlit"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("crop_recommender_app")


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load model from disk and handle common pitfalls."""
    logger.info(f"Loading model from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    try:
        model = joblib.load(path)
    except Exception as e:
        logger.exception("Failed to load model via joblib.load")
        raise

    logger.info("Model loaded successfully")
    return model


def try_get_feature_names(model) -> Optional[List[str]]:
    """Attempt to discover feature names from a scikit-learn compatible model/pipeline.

    Returns None if it cannot be inferred.
    """
    # Common attribute used by sklearn: feature_names_in_
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # If pipeline, try to introspect steps
    try:
        # sklearn Pipeline
        if hasattr(model, "named_steps"):
            # check for a transformer exposing feature names
            for name, step in model.named_steps.items():
                if hasattr(step, "get_feature_names_out"):
                    try:
                        return list(step.get_feature_names_out())
                    except Exception:
                        continue
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
    except Exception:
        pass

    # If model is a dict or custom object with metadata
    if isinstance(model, dict):
        if "feature_names" in model and isinstance(model["feature_names"], (list, tuple)):
            return list(model["feature_names"])

    return None


def validate_dataframe(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Ensure dataframe has the expected columns in the expected order.

    If columns are missing, raise a ValueError with details.
    """
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    # Reorder to match expected order
    return df[feature_names]


def predict(model, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return predictions and optional probabilities if available."""
    # Prefer predict_proba when available
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
    except Exception:
        # Some wrapped models may require raw arrays
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df.values)
        except Exception:
            probs = None

    # Predictions
    try:
        preds = model.predict(df)
    except Exception:
        preds = model.predict(df.values)

    return preds, probs


def render_manual_input(feature_names: List[str]) -> pd.DataFrame:
    """Render a form that collects inputs for a single sample and returns a dataframe."""
    st.subheader("Manual input — single sample")
    st.markdown("Enter values for each feature. Use sensible units and ranges.")
    with st.form(key="manual_input_form"):
        row = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            # Heuristic: choose numeric widget for numeric-sounding names, else text
            label = feat
            if any(keyword in feat.lower() for keyword in ["ph", "temp", "rain", "moist", "nitro", "phosph", "potash", "area", "year"]):
                val = cols[i % 2].number_input(label, value=0.0, format="%.4f", key=f"num_{i}")
            else:
                # allow numeric input by default but text fallback
                val = cols[i % 2].text_input(label, value="0", key=f"txt_{i}")
            row[feat] = val

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Convert to dataframe and coerce types to numeric when possible
        df = pd.DataFrame([row])
        for c in df.columns:
            # try numeric conversion
            df[c] = pd.to_numeric(df[c], errors="ignore")
        return df

    return pd.DataFrame(columns=feature_names)


def render_csv_uploader() -> Optional[pd.DataFrame]:
    st.subheader("Batch input — upload CSV")
    st.markdown("Upload a CSV file with rows matching the model's features. First row must be header.")

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
    # If probs exist and are multi-class, add top confidence
    if probs is not None:
        try:
            if probs.ndim == 2:
                top_conf = probs.max(axis=1)
                out["confidence"] = np.round(top_conf, 4)
                # Add probabilities for each class if desirable
                # classes = getattr(model, 'classes_', None)
                # if classes is not None:
                #     for idx, cls in enumerate(classes):
                #         out[f"prob_{cls}"] = probs[:, idx]
        except Exception:
            pass

    st.subheader("Predictions")
    st.dataframe(out)

    # Download link
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions as CSV", csv, file_name="predictions.csv", mime="text/csv")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.write(
        "This app serves a pickled crop recommendation model. It is built to be robust and production-friendly: model caching, CSV batch input, input validation, and graceful error handling."
    )

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        st.error(f"Model file not found. Make sure MODEL_PATH is set correctly. \n{e}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    # Discover feature names where possible
    feature_names = try_get_feature_names(model)
    if feature_names is None:
        st.warning(
            "Could not automatically detect feature names from the model. You can either (A) upload a CSV with the correct header, or (B) manually enter the expected feature names as a JSON list (e.g. [\"N\", \"P\", \"K\", \"temperature\", \"humidity\"])."
        )

        with st.expander("Provide feature names manually (JSON list)"):
            feat_text = st.text_area("Feature names JSON", value="[]", height=80)
            try:
                parsed = json.loads(feat_text)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    feature_names = parsed
                    st.success("Set feature names from your JSON list")
                else:
                    st.info("Provide a JSON list of strings, e.g. [\"N\", \"P\", \"K\"]")
            except json.JSONDecodeError:
                st.info("Invalid JSON yet — leave empty or provide a valid JSON list")

    # Two-column layout: left for inputs, right for model info and logs
    left, right = st.columns([3, 2])

    with left:
        st.header("Input")
        csv_df = render_csv_uploader()
        manual_df = pd.DataFrame()
        if feature_names is not None and len(feature_names) > 0:
            manual_df = render_manual_input(feature_names)

        # Priority: CSV -> manual
        input_df = None
        if csv_df is not None:
            input_df = csv_df
        elif manual_df is not None and not manual_df.empty:
            input_df = manual_df

        if st.button("Run prediction"):
            safe_predict_and_show(model, input_df, feature_names)

    with right:
        st.header("Model & Diagnostics")
        with st.expander("Model metadata"):
            try:
                st.write(type(model))
                # Show a short repr
                st.code(repr(model)[:1000])
            except Exception:
                st.write("Could not render model metadata")

        with st.expander("Model feature names (detected)"):
            st.write(feature_names)

        with st.expander("Developer / Production notes"):
            st.markdown(
                "- Model path: `%s`\n- Model loading is cached to reduce cold-start overhead.\n- For production: serve with Docker + a small reverse proxy (NGINX) or deploy on Streamlit Cloud / Kubernetes.\n- Use HTTPS, enable authentication for private models, and monitor inputs."
                % MODEL_PATH
            )

    st.markdown("---")
    st.write("Helpful tips:")
    st.write(
        "- If your model expects scaled/encoded features, make sure the uploaded CSV or manual inputs are preprocessed exactly the same way.\n- Prefer using a scikit-learn Pipeline that contains preprocessing + estimator so the app only needs the final pipeline for inference."
    )


if __name__ == "__main__":
    main()

"""
Shared UI helpers for the Streamlit dashboard: theming, cached data/model
loaders, and small reusable components used across every page.
"""

import json
import os

import pandas as pd
import streamlit as st

from src import config
from src.preprocessing import preprocess_series

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#1B2340"          # near-navy text / headings
SURFACE = "#F7F8FB"      # app background
CARD = "#FFFFFF"         # card background
BORDER = "#E7E9F3"
ACCENT = "#4C6FFF"       # primary brand accent (indigo-blue)
ACCENT_SOFT = "#EEF1FF"
TEAL = "#22C1A4"         # "low risk" / positive
AMBER = "#F2A93B"        # "moderate risk"
CORAL = "#EF6461"        # "high risk" / negative
MUTED = "#6B7280"


def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {INK};
        }}
        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.01em;
        }}
        .stApp {{
            background-color: {SURFACE};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {CARD};
            border-right: 1px solid {BORDER};
        }}
        .mh-eyebrow {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {ACCENT};
            background: {ACCENT_SOFT};
            padding: 4px 10px;
            border-radius: 999px;
            margin-bottom: 10px;
        }}
        .mh-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 22px 24px;
            box-shadow: 0 1px 3px rgba(27,35,64,0.04);
            margin-bottom: 16px;
        }}
        .mh-disclaimer {{
            background: #FFF7ED;
            border: 1px solid #FCE3C0;
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 0.88rem;
            color: #7C4A03;
        }}
        .mh-demo-banner {{
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-radius: 10px;
            padding: 10px 16px;
            font-size: 0.85rem;
            color: #991B1B;
            margin-bottom: 14px;
        }}
        .mh-risk-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.95rem;
        }}
        .mh-risk-low {{ background: #E7FBF6; color: #0F7A63; }}
        .mh-risk-moderate {{ background: #FFF4E0; color: #92650B; }}
        .mh-risk-high {{ background: #FDEBEB; color: #B3261E; }}
        .mh-metric-label {{
            color: {MUTED};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .mh-metric-value {{
            font-size: 1.6rem;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_css_class(risk_label: str) -> str:
    label = risk_label.lower()
    if "high" in label:
        return "mh-risk-high"
    if "moderate" in label:
        return "mh-risk-moderate"
    return "mh-risk-low"


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    st.markdown(f'<span class="mh-eyebrow">{eyebrow}</span>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p style='color:{MUTED}; font-size:1.02rem;'>{subtitle}</p>", unsafe_allow_html=True)


def render_disclaimer():
    st.markdown(f'<div class="mh-disclaimer">⚠️ <b>Responsible AI notice:</b> {config.DISCLAIMER}</div>', unsafe_allow_html=True)


def render_demo_banner(is_demo: bool):
    # Demo-dataset banner intentionally disabled.
    return


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_dataset():
    if os.path.exists(config.DATASET_PATH):
        return pd.read_csv(config.DATASET_PATH), False
    elif os.path.exists(config.SAMPLE_DATASET_PATH):
        return pd.read_csv(config.SAMPLE_DATASET_PATH), True
    return None, None


@st.cache_data(show_spinner=False)
def load_clean_dataset():
    df, is_demo = load_raw_dataset()
    if df is None:
        return None, None
    df = df.dropna(subset=[config.TEXT_COLUMN, config.TARGET_COLUMN]).copy()
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).str.strip()
    df = df[df[config.TEXT_COLUMN].str.len() > 0]
    df = df.drop_duplicates(subset=[config.TEXT_COLUMN]).reset_index(drop=True)
    df["clean_text"] = preprocess_series(df[config.TEXT_COLUMN])
    return df, is_demo


def _transformer_present() -> bool:
    return os.path.isdir(config.TRANSFORMER_DIR) and os.path.exists(config.TRANSFORMER_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_metadata():
    if not os.path.exists(config.METADATA_PATH):
        return None
    with open(config.METADATA_PATH) as f:
        meta = json.load(f)
    if _transformer_present():
        with open(config.TRANSFORMER_METRICS_PATH) as f:
            name, m = next(iter(json.load(f).items()))
        meta = {
            **meta,
            "model_name": name,
            "best_params": m.get("best_params", {}),
            "best_model_metrics": {k: v for k, v in m.items() if k != "confusion_matrix"},
        }
    return meta


@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(config.METRICS_PATH):
        return None
    with open(config.METRICS_PATH) as f:
        results = json.load(f)
    if _transformer_present():
        with open(config.TRANSFORMER_METRICS_PATH) as f:
            results.update(json.load(f))
    return results


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    from src.predict import load_artifacts
    return load_artifacts()

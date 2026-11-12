"""
Mental Health Risk Prediction Using Social Media Analytics
Streamlit app entry point (Home page).

Run with:
    streamlit run app.py
"""

import streamlit as st

from src import config
from src.ui import inject_theme, load_metadata, page_header, render_disclaimer

st.set_page_config(
    page_title="Mental Health Risk Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

with st.sidebar:
    st.markdown("### 🧠 Mental Health Risk\n#### Prediction Platform")
    st.caption("NLP + Machine Learning research prototype")
    st.divider()
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/1_Prediction.py", label="Prediction", icon="🔍")
    st.page_link("pages/2_Dataset_Analytics.py", label="Dataset Analytics", icon="📊")
    st.page_link("pages/3_Model_Performance.py", label="Model Performance", icon="🏆")
    st.page_link("pages/4_Explainability.py", label="Explainability", icon="🧩")
    st.page_link("pages/5_About.py", label="About Project", icon="ℹ️")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
hero_col, badge_col = st.columns([3, 1])
with hero_col:
    page_header(
        "AI / ML Research Prototype",
        "Mental Health Risk Prediction",
        "AI-powered analysis of social-media text using NLP and machine learning "
        "to surface potential mental-health risk signal patterns.",
    )

st.write("")

metadata = load_metadata()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown('<div class="mh-metric-label">Model in use</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mh-metric-value">{metadata["model_name"] if metadata else "Not trained"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown('<div class="mh-metric-label">Macro F1-score</div>', unsafe_allow_html=True)
    f1 = metadata["best_model_metrics"]["f1_macro"] if metadata else None
    st.markdown(f'<div class="mh-metric-value">{f"{f1:.2%}" if f1 is not None else "—"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown('<div class="mh-metric-label">Training rows</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mh-metric-value">{metadata["dataset_rows"] if metadata else "—"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown('<div class="mh-metric-label">Risk classes</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mh-metric-value">{len(metadata["classes"]) if metadata else "—"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### How it works")
    st.markdown(
        """
1. **You submit** a short piece of social-media-style text.
2. **NLP preprocessing** cleans the text (URLs, mentions, punctuation, stopwords) while preserving negations like *"not happy"*.
3. **TF-IDF** converts the cleaned text into numerical features.
4. A **trained classification model** (selected automatically by macro F1-score during training) predicts the dataset category the text most resembles.
5. The dashboard shows the **prediction, confidence, textual insights, and sentiment** — always labeled as a *risk signal*, never a diagnosis.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### Technology used")
    t1, t2, t3 = st.columns(3)
    t1.markdown("**NLP**\n\nCustom tokenizer, stopword filtering, negation preservation")
    t2.markdown("**Features**\n\nTF-IDF (unigrams + bigrams)")
    t3.markdown("**Models**\n\nLogistic Regression · Naive Bayes · Linear SVM")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### Get started")
    st.write("Try analyzing a sample social-media style post.")
    if st.button("🔍 Go to Prediction →", use_container_width=True):
        st.switch_page("pages/1_Prediction.py")
    if st.button("📊 View Dataset Analytics", use_container_width=True):
        st.switch_page("pages/2_Dataset_Analytics.py")
    if st.button("🏆 View Model Performance", use_container_width=True):
        st.switch_page("pages/3_Model_Performance.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
render_disclaimer()

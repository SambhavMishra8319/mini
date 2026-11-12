import streamlit as st

from src import config
from src.ui import inject_theme, load_raw_dataset, load_clean_dataset, page_header, render_demo_banner
from src.visualization import (
    class_distribution_chart,
    text_length_distribution_chart,
    word_frequency_chart,
)
from src.sentiment import score_sentiment

st.set_page_config(page_title="Dataset Analytics | Mental Health Risk", page_icon="📊", layout="wide")
inject_theme()

page_header(
    "Exploratory Data Analysis",
    "Dataset Analytics",
    "Real statistics and charts computed directly from the training dataset.",
)

raw_df, is_demo = load_raw_dataset()

if raw_df is None:
    st.error(
        "No dataset found. Add a CSV at `data/dataset.csv` (see `data/README.md`), "
        "or generate the demo dataset with `python data/generate_sample_dataset.py`."
    )
    st.stop()

render_demo_banner(is_demo)

clean_df, _ = load_clean_dataset()

n_raw = len(raw_df)
n_missing = raw_df[config.TEXT_COLUMN].isna().sum() + raw_df[config.TARGET_COLUMN].isna().sum()
n_duplicates = raw_df.duplicated(subset=[config.TEXT_COLUMN]).sum()
n_clean = len(clean_df)
n_classes = clean_df[config.TARGET_COLUMN].nunique()
avg_len = clean_df[config.TEXT_COLUMN].astype(str).str.split().apply(len).mean()

m1, m2, m3, m4, m5 = st.columns(5)
for col, label, value in zip(
    [m1, m2, m3, m4, m5],
    ["Total records", "Classes", "Missing values", "Duplicates", "Avg. text length"],
    [n_raw, n_classes, int(n_missing), int(n_duplicates), f"{avg_len:.1f} words"],
):
    with col:
        st.markdown('<div class="mh-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="mh-metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mh-metric-value">{value}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.pyplot(class_distribution_chart(clean_df, config.TARGET_COLUMN), use_container_width=True)
    st.caption(f"After cleaning: {n_clean} rows remain (from {n_raw} raw rows).")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.pyplot(text_length_distribution_chart(clean_df, config.TEXT_COLUMN), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    top_n = st.slider("Top N words", 10, 30, 20)
    st.pyplot(word_frequency_chart(clean_df["clean_text"], top_n=top_n), use_container_width=True)
    st.caption("Computed after NLP preprocessing (stopwords removed, negations preserved).")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("##### Sentiment distribution")
    with st.spinner("Scoring sentiment across the dataset..."):
        sentiments = clean_df["clean_text"].apply(lambda t: score_sentiment(t)[0])
    st.bar_chart(sentiments.value_counts())
    st.caption("Lexicon-based sentiment signal — a supporting feature, not a diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="mh-card">', unsafe_allow_html=True)
st.markdown("##### Class distribution (table)")
st.dataframe(
    clean_df[config.TARGET_COLUMN].value_counts().rename_axis("class").reset_index(name="count"),
    use_container_width=True,
    hide_index=True,
)
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Preview raw dataset sample"):
    st.dataframe(raw_df.head(20), use_container_width=True)

import streamlit as st

from src.ui import inject_theme, page_header, render_disclaimer

st.set_page_config(page_title="About | Mental Health Risk", page_icon="ℹ️", layout="wide")
inject_theme()

page_header("Project Overview", "About This Project", "")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### Problem")
    st.write(
        "Large amounts of social-media text are generated every day. Manually "
        "reviewing all of it for potentially concerning language patterns is "
        "impractical at scale."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### Proposed Solution")
    st.write(
        "An NLP and machine-learning pipeline that automatically analyzes "
        "social-media-style text and identifies patterns associated with the "
        "risk categories present in the training dataset — surfaced as a "
        "supporting *signal*, not a clinical decision."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### Technologies")
    st.markdown(
        "- Python 3.11+\n"
        "- Pandas, NumPy\n"
        "- Scikit-learn (Logistic Regression, Naive Bayes, Linear SVM)\n"
        "- Custom NLP preprocessing (tokenization, stopword removal, negation preservation)\n"
        "- TF-IDF feature extraction\n"
        "- Streamlit dashboard\n"
        "- Matplotlib / Seaborn visualizations\n"
        "- Joblib model persistence"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### Limitations")
    st.markdown(
        "- Social-media text lacks full clinical context.\n"
        "- Predictions depend entirely on training-data quality and coverage.\n"
        "- Models can reflect biases present in their training data.\n"
        "- Sarcasm, irony, and cultural context are difficult for the model to interpret.\n"
        "- Performance may vary across languages and cultural expressions.\n"
        "- The system **cannot** replace a professional clinical assessment."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### Responsible AI")
    st.write(
        "This system is explicitly designed to avoid clinical language. It never "
        "states that a user 'has' a condition — it reports a dataset-associated "
        "risk *signal* only, and every prediction is shown alongside a clear "
        "non-diagnostic disclaimer."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("#### Future Improvements")
    st.markdown(
        "- Transformer / BERT-based models for deeper contextual understanding\n"
        "- Multilingual NLP support\n"
        "- Better probability calibration\n"
        "- Larger, more diverse, and more representative datasets\n"
        "- Human-in-the-loop expert review of flagged content\n"
        "- Privacy-preserving / on-device deployment options"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
render_disclaimer()

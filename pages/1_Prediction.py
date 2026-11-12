import streamlit as st

from src.predict import ArtifactsNotFoundError, predict
from src.ui import (
    inject_theme,
    page_header,
    render_disclaimer,
    risk_css_class,
)
from src.visualization import class_probability_chart

st.set_page_config(page_title="Prediction | Mental Health Risk", page_icon="🔍", layout="wide")
inject_theme()

page_header(
    "Live Analysis",
    "Text Risk Prediction",
    "Enter a social-media-style post below. The model will analyze it and return a "
    "potential risk signal — not a medical diagnosis.",
)

sample_texts = {
    "— Choose a sample (optional) —": "",
    "Stress example": "I have so many deadlines this week and I honestly can't keep up with everything.",
    "Anxiety example": "My heart races and I feel really nervous whenever I have to speak in front of people.",
    "Depression example": "I feel empty and exhausted, nothing brings me joy anymore and I just feel numb.",
    "Normal example": "Had a wonderful dinner with my family tonight, feeling really grateful.",
}

choice = st.selectbox("Try a sample post", list(sample_texts.keys()))
default_text = sample_texts[choice]

with st.form("prediction_form"):
    text_input = st.text_area(
        "Enter a social-media post for analysis...",
        value=default_text,
        height=140,
        max_chars=2000,
        placeholder="e.g. I've been feeling extremely stressed and exhausted lately...",
    )
    submitted = st.form_submit_button("Analyze Text", type="primary", use_container_width=True)

if submitted:
    if not text_input or not text_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            with st.spinner("Running NLP preprocessing and model inference..."):
                result = predict(text_input)
        except ArtifactsNotFoundError:
            st.error(
                "No trained model found. Please run `python -m src.train` from the "
                "project root before using the prediction page."
            )
        except ValueError as e:
            st.warning(str(e))
        else:
            st.write("")
            pred_col, insight_col = st.columns([1, 1])

            # ---- Prediction card ----
            with pred_col:
                st.markdown('<div class="mh-card">', unsafe_allow_html=True)
                st.markdown("#### Prediction")
                badge_class = risk_css_class(result.risk_label)
                st.markdown(
                    f'<span class="mh-risk-badge {badge_class}">{result.risk_label}</span>',
                    unsafe_allow_html=True,
                )
                st.write("")
                st.markdown(f"**Dataset category matched:** `{result.predicted_class.title()}`")
                if result.supports_probability and result.confidence is not None:
                    st.markdown(f"**Model confidence:** {result.confidence:.0%}")
                    st.progress(min(1.0, result.confidence))
                else:
                    st.caption(
                        "This model does not produce calibrated probabilities, "
                        "so no confidence score is shown."
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                if result.class_probabilities:
                    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
                    st.markdown("##### Class probability breakdown")
                    fig = class_probability_chart(result.class_probabilities)
                    st.pyplot(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            # ---- Text analysis card ----
            with insight_col:
                st.markdown('<div class="mh-card">', unsafe_allow_html=True)
                st.markdown("#### Text Analysis")
                s1, s2, s3 = st.columns(3)
                s1.metric("Word count", result.text_stats["word_count"])
                s2.metric("Characters", result.text_stats["character_count"])
                s3.metric("Avg word length", result.text_stats["avg_word_length"])

                st.write("")
                st.markdown(f"**Sentiment:** {result.sentiment_label}  ·  score {result.sentiment_score:+.2f}")
                st.caption("Sentiment is one textual characteristic only — it does not determine mental-health status.")

                if result.top_terms:
                    st.write("")
                    st.markdown("**Influential textual indicators:**")
                    st.markdown(
                        " ".join(f"`{t}`" for t in result.top_terms),
                    )
                    st.caption(
                        "These are the highest-weighted TF-IDF terms in your text for the "
                        "model. Individual words do not, on their own, prove any condition."
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            render_disclaimer()

if not submitted:
    st.write("")
    render_disclaimer()

import pandas as pd
import streamlit as st

from src.explain import explanation_method_name, global_top_features, is_explainable, local_explanation
from src.predict import ArtifactsNotFoundError, load_artifacts
from src.preprocessing import preprocess_text
from src.ui import inject_theme, page_header

st.set_page_config(page_title="Explainability | Mental Health Risk", page_icon="🧩", layout="wide")
inject_theme()

page_header(
    "Model Insights",
    "Explainability",
    "Understand which words most influence the model's predictions — computed from the model's own weights, never fabricated.",
)

try:
    model, vectorizer, label_encoder, metadata = load_artifacts()
except ArtifactsNotFoundError:
    st.error("No trained model found. Run `python -m src.train` from the project root first.")
    st.stop()

if not is_explainable(model):
    st.warning(
        f"The current best model (**{type(model).__name__}**) does not have a "
        "reliable, honest explanation method available in this project. "
        "Rather than generate a misleading explanation, no chart is shown here."
    )
    st.stop()

st.markdown('<div class="mh-card">', unsafe_allow_html=True)
st.markdown(f"**Explanation method:** {explanation_method_name(model)}")
st.caption(
    "Because the trained model is linear, these are the model's exact internal weights — "
    "not an approximation like SHAP/LIME would produce, and not invented."
)
st.markdown('</div>', unsafe_allow_html=True)

classes = list(label_encoder.classes_)

st.markdown("#### Global feature importance")
selected_class = st.selectbox("Class", classes)
class_idx = classes.index(selected_class)

top_n = st.slider("Number of top terms", 5, 25, 15)
features = global_top_features(model, vectorizer, class_idx, top_n=top_n)
feat_df = pd.DataFrame(features, columns=["Term", "Weight"]).set_index("Term")
st.bar_chart(feat_df)
st.caption(
    f"Terms most strongly associated with the **{selected_class}** class, according to the model's own weights."
)

st.divider()

st.markdown("#### Local explanation for a specific text")
text_input = st.text_area(
    "Enter text to explain",
    placeholder="e.g. I feel exhausted and empty, nothing brings me joy anymore",
    height=100,
)

if st.button("Explain this text", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        clean_text = preprocess_text(text_input)
        if not clean_text:
            st.warning("No meaningful words remained after cleaning this text.")
        else:
            pred_idx = model.predict(vectorizer.transform([clean_text]))[0]
            predicted_class = label_encoder.inverse_transform([pred_idx])[0]
            st.markdown(f"**Predicted class:** `{predicted_class}`")

            contributions = local_explanation(model, vectorizer, clean_text, pred_idx, top_n=10)
            if not contributions:
                st.info("No recognized vocabulary terms were found to explain in this text.")
            else:
                contrib_df = pd.DataFrame(contributions, columns=["Term", "Contribution"]).set_index("Term")
                st.bar_chart(contrib_df)
                st.caption(
                    "Positive bars pushed the prediction toward this class; the size reflects "
                    "how strongly that word (weighted by its TF-IDF value in your text) contributed."
                )

st.write("")
st.info(
    "🔎 **Note on reliability:** these are word-level statistical associations learned from "
    "the training data, not proof that any word 'causes' or 'indicates' a mental-health "
    "condition. Explanations should always be read alongside the disclaimer on every page."
)

import pandas as pd
import streamlit as st

from src.ui import inject_theme, load_metadata, load_metrics, page_header, render_demo_banner
from src.visualization import confusion_matrix_chart, metric_group_comparison_chart, model_comparison_chart

st.set_page_config(page_title="Model Performance | Mental Health Risk", page_icon="🏆", layout="wide")
inject_theme()

page_header(
    "Evaluation",
    "Model Performance",
    "All metrics below are computed from real evaluation on a held-out test split — nothing is hardcoded.",
)

metadata = load_metadata()
results = load_metrics()

if metadata is None or results is None:
    st.error(
        "No trained model found. Run `python -m src.train` from the project root, then reload this page."
    )
    st.stop()

render_demo_banner(metadata.get("is_demo_dataset", False))

st.markdown('<div class="mh-card">', unsafe_allow_html=True)
st.markdown(f"##### 🏆 Best model selected: **{metadata['model_name']}**")
st.caption(f"Selection metric: {metadata['selection_metric']} · Trained on {metadata['dataset_rows']} rows · {metadata['training_date']}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("#### Model comparison table")
rows = []
for name, m in results.items():
    rows.append({
        "Model": name,
        "Accuracy": round(m["accuracy"], 3),
        "Precision (macro)": round(m["precision_macro"], 3),
        "Recall (macro)": round(m["recall_macro"], 3),
        "F1 (macro)": round(m["f1_macro"], 3),
        "F1 (weighted)": round(m["f1_weighted"], 3),
        "Supports probability": m.get("supports_probability", False),
    })
comp_df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False)
st.dataframe(
    comp_df.style.highlight_max(subset=["F1 (macro)"], color="#E7FBF6"),
    use_container_width=True,
    hide_index=True,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.pyplot(model_comparison_chart(results, metric="f1_macro"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.pyplot(metric_group_comparison_chart(results), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("#### Confusion matrix & per-class breakdown")
model_names = list(results.keys())
selected_model = st.selectbox("Select a model", model_names, index=model_names.index(metadata["model_name"]))

sel = results[selected_model]
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.pyplot(confusion_matrix_chart(sel["confusion_matrix"], sel["classes"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown("##### Per-class metrics")
    per_class_df = pd.DataFrame(sel["per_class"]).T.round(3)
    st.dataframe(per_class_df, use_container_width=True)
    if sel.get("roc_auc") is not None:
        st.metric("ROC-AUC (binary)", f"{sel['roc_auc']:.3f}")
    else:
        st.caption("ROC-AUC is only shown for binary classification tasks with probability scores.")
    st.markdown('</div>', unsafe_allow_html=True)

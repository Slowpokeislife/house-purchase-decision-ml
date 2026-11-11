"""Optional portfolio demo: run with `streamlit run app.py` after training."""
from pathlib import Path
import json
import streamlit as st
from house_purchase.predict import predict_one

st.set_page_config(page_title="House Purchase | ML Demo", page_icon="🏠", layout="wide")
st.title("🏠 House Purchase Decision Prediction")
st.caption("Classification demo · TensorFlow / scikit-learn · original Kaggle schema")
st.info("Educational model, not financial or lending advice. Use hypothetical inputs, not personal financial records.")

model_options = [p for p in [Path("artifacts/tuned"), Path("artifacts/optimized"),
                                Path("artifacts/original"), Path("artifacts/baseline")]
                 if (p / "input_schema.json").is_file()]
if not model_options:
    st.warning("No locally trained model detected. First run `house-purchase download`, then `house-purchase baseline` or `house-purchase train`.")
    st.stop()
selected = st.selectbox("Trained experiment", model_options, format_func=lambda p:p.name)
schema = json.loads((selected / "input_schema.json").read_text(encoding="utf-8"))
st.subheader("Property and buyer profile")
record = {}
with st.form("prediction"):
    left, right = st.columns(2)
    for i, (name, specs) in enumerate(schema["categorical"].items()):
        with (left if i % 2 == 0 else right):
            options = specs["options"]
            record[name] = st.selectbox(name.replace("_", " ").title(), options,
                                        index=options.index(specs["default"]) if specs["default"] in options else 0)
    with st.expander("Numerical attributes (adjust as needed)", expanded=True):
        cols = st.columns(3)
        for idx, (name, specs) in enumerate(schema["numerical"].items()):
            with cols[idx % 3]:
                record[name] = st.number_input(name.replace("_", " ").title(),
                    value=float(specs["median"]), format="%.4f")
    submitted = st.form_submit_button("Estimate purchase decision", type="primary")
if submitted:
    try:
        result = predict_one(selected, record)
        st.metric("Estimated purchase probability", f"{result['purchase_probability']:.1%}")
        st.write("Prediction:", "Purchase" if result["predicted_purchase"] else "No purchase")
        st.caption("Classification threshold: 0.50. The probability may not be calibrated; do not treat it as a verified real-world likelihood.")
    except Exception as e:
        st.error(f"Prediction could not be completed: {e}")

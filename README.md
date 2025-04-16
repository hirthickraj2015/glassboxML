# 🧠 GlassboxML

> **The World’s First Human-Readable ML Model Format**  
> Build, audit, and deploy models with full transparency.  
> No more black-boxes. Every prediction, explained.

---

## 🚀 What is GlassboxML?

**GlassboxML** is an open-source standard for packaging machine learning models in a format that prioritizes **transparency**, **explainability**, and **traceability**.

In a world of black-box AI, GlassboxML makes your models:
- 🔍 **Interpretable**: See why and how decisions are made.
- 🧾 **Auditable**: Track the lifecycle, inputs, and logic of every model version.
- 👥 **Accessible**: Designed for both developers and non-technical stakeholders.

Think of it as a blend of `ONNX` + `SHAP` + `Model Card` – but human-readable and built for trust.

---

## ❓ Why GlassboxML?

Today’s ML formats (like `joblib`, `pickle`, or even `ONNX`) focus only on **model weights and structure**.

GlassboxML captures **the full picture**:

| Feature | Traditional Formats | GlassboxML |
|--------|----------------------|-------------|
| Model Weights | ✅ | ✅ |
| Feature Contributions | ❌ | ✅ |
| Training Metadata | ❌ | ✅ |
| Assumptions & Thresholds | ❌ | ✅ |
| Human-Readable Prediction Flow | ❌ | ✅ |
| Version & Audit Trail | ❌ | ✅ |

---

## 📦 What’s Inside a GlassboxML File?

A GlassboxML package includes:
- `model_logic.json` → Decision paths, logic rules, or SHAP values
- `prediction_breakdown.json` → Per-prediction feature impact
- `metadata.yaml` → Training context, assumptions, dataset info
- `audit_log.json` → Version history, changes, usage
- Optional: `glassboxml.explain()` → Python API for breakdowns

---

## 🛠️ Example Use Case

```python
from glassboxml import explain

prediction = model.predict(input_data)
glassboxml_view = explain(prediction)

glassboxml_view.visualize()  # Streamlit/HTML view of model logic

from glassboxml.core.prediction_explainer import explain_prediction
from glassboxml.utils.visualizer import visualize_explanation
import matplotlib.pyplot as plt
import numpy as np

def explain(model, X, feature_names):
    explanations = []
    for i in range(len(X)):
        explanation = explain_prediction(model, X[i], feature_names)
        explanations.append(explanation)

        # --- Call text and visual explanation ---
        print(f"\nPrediction {i+1} Explanation")
        print(f"Prediction: {explanation['prediction']:.4f} (Bias: {explanation['bias']:.4f})")
        print("-" * 40)
        for feat in explanation['explanation']:
            print(f"  {feat['feature']}: value={feat['value']}, weight={feat['weight']:.3f}, contrib={feat['contribution']:.3f}")

        # --- Simple visualization inline ---
        visualize_explanation(explanation)

    return explanations

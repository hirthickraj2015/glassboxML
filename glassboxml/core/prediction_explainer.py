from glassboxml.utils.visualizer import formula_view, terminal_prediction_view, aggregated_view
from glassboxml.core.linear_model_parser import LinearRegression

def explain(model, X, feature_names, view='terminal'):
    """
    Explains predictions with different view options:
    - view='formula': shows mini formula
    - view='terminal': per-prediction terminal colored bars
    - view='aggregate': shows aggregated average contributions
    """
    explanations = []

    if view == 'formula':
        formula_view(model, feature_names)

    for i, x in enumerate(X):
        explanation = explain_prediction(model, x, feature_names)
        explanations.append(explanation)
        if view == 'terminal':
            terminal_prediction_view(explanation)

    if view == 'aggregate':
        aggregated_view(explanations)

    return explanations

# --- helper function to match the old API ---
def explain_prediction(model, x, feature_names):
    """
    Produces explanation dictionary for a single prediction.
    """
    y_pred = model.predict([x])[0]
    contribs = []
    for w, f, val in zip(model.weights, feature_names, x):
        contribs.append({
            'feature': f,
            'value': val,
            'weight': w,
            'contribution': w * val
        })
    return {
        'prediction': y_pred,
        'bias': model.bias,
        'explanation': contribs
    }

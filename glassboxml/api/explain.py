import numpy as np

def explain(model, input_data):
    """
    Explain prediction from a Linear Regression model.

    Args:
        model: Trained sklearn LinearRegression model.
        input_data: A dictionary or numpy array with feature values.

    Returns:
        Dict with predicted value, coefficients, and contribution per feature.
    """
    feature_names = model.feature_names_in_
    coefs = model.coef_
    intercept = model.intercept_

    if isinstance(input_data, dict):
        input_values = np.array([input_data[feat] for feat in feature_names])
    else:
        input_values = np.array(input_data)

    contribution = input_values * coefs
    prediction = np.dot(input_values, coefs) + intercept

    return {
        "prediction": prediction,
        "intercept": intercept,
        "coefficients": dict(zip(feature_names, coefs)),
        "contributions": dict(zip(feature_names, contribution))
    }

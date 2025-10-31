from glassboxML.glassboxml.core.linear_model_parser import LinearRegression
import numpy as np

def test_linear_regression():
    X = np.array([[1, 2], [2, 3]])
    y = np.array([3, 5])
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    # Check prediction shape
    assert preds.shape == y.shape

    # Check prediction accuracy (within tolerance)
    np.testing.assert_allclose(preds, y, atol=1e-6)

    # Optionally
    w_expected = np.array([1.0, 1.0])  
    np.testing.assert_allclose(model.weights, w_expected, atol=1e-6)

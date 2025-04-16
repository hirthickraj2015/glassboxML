from glassboxml.api.explain import explain
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def test_glassbox_explainer():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier().fit(X, y)
    prediction = explain(model, X[0:1])
    assert prediction is not None

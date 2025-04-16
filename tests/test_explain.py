import pickle
import json
from glassboxml.api.explain import explain

def test_linear_regression_explain():
    with open("examples/sample_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("examples/sample_input.json") as f:
        input_data = json.load(f)

    result = explain(model, input_data)
    
    assert "prediction" in result
    assert "coefficients" in result
    assert isinstance(result["coefficients"], dict)

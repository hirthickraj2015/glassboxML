import numpy as np
from glassboxml.utils.visualizer import terminal_prediction_view, aggregated_view
from collections import defaultdict

def explain_tree_prediction(tree, x, all_features=None):
    contribs = {f:0.0 for f in (all_features or [f'feature{i+1}' for i in range(len(x))])}
    node = tree.root
    bias = node.value if node.value is not None else 0
    current_value = bias

    while node.feature is not None:
        feat_name = f'feature{node.feature+1}'
        threshold = node.threshold
        child = node.left if x[node.feature] <= threshold else node.right
        child_pred = child.value if child.value is not None else tree._predict_single(x, child)
        contrib = child_pred - current_value
        contribs[feat_name] += contrib
        current_value = child_pred
        node = child

    explanation_list = [{'feature': k, 'value': x[int(k[-1])-1], 'contribution': v} for k, v in contribs.items()]
    return {
        'prediction': tree.predict([x])[0],
        'bias': bias,
        'explanation': explanation_list
    }

def explain_tree(tree, X, view='terminal'):
    explanations = []
    for x in X:
        exp = explain_tree_prediction(tree, x)
        explanations.append(exp)
        if view == 'terminal':
            terminal_prediction_view(exp)
    if view == 'aggregate':
        aggregated_view(explanations)
    return explanations

def visualize_tree_text(node, feature_names=None, depth=0):
    indent = "  " * depth
    if node.feature is None:
        print(f"{indent}Leaf: value = {node.value:.4f}")
        return
    feat_name = feature_names[node.feature] if feature_names else f'feature{node.feature+1}'
    print(f"{indent}{feat_name} <= {node.threshold:.4f}?")
    visualize_tree_text(node.left, feature_names, depth+1)
    print(f"{indent}{feat_name} > {node.threshold:.4f}?")
    visualize_tree_text(node.right, feature_names, depth+1)

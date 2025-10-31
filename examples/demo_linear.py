import numpy as np
from glassboxml.core.tree_parser import DecisionTreeRegressorScratch
from glassboxml.api.tree_explain import explain_tree, visualize_tree_text

# Sample data
X = np.array([[1,10], [2,8], [3,15], [4,12], [5,20]])
y = np.array([12,17,23,28,33])

feature_names = ['feature1', 'feature2']

# Train scratch decision tree
tree = DecisionTreeRegressorScratch(max_depth=13)
tree.fit(X, y)

# Visualize tree structure in terminal
print("\n🌳 Text-based Tree Visualization:")
visualize_tree_text(tree.root, feature_names)

# Explain each prediction in terminal
print("\n🔍 Prediction Explanations:")
explain_tree(tree, X, view='terminal')


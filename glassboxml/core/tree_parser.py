import numpy as np

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # prediction at this node

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.feature_names = [f'feature{i+1}' for i in range(X.shape[1])]
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        # Leaf condition
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=np.mean(y))
        best_feat, best_thresh, best_mse = None, None, float('inf')
        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                mse = np.var(y[left_mask]) * left_mask.sum() + np.var(y[right_mask]) * right_mask.sum()
                if mse < best_mse:
                    best_mse, best_feat, best_thresh = mse, f, t
        if best_feat is None:
            return TreeNode(value=np.mean(y))
        left_node = self._build_tree(X[X[:, best_feat] <= best_thresh], y[X[:, best_feat] <= best_thresh], depth+1)
        right_node = self._build_tree(X[X[:, best_feat] > best_thresh], y[X[:, best_feat] > best_thresh], depth+1)
        # Node value = mean of all samples in node
        node_value = np.mean(y)
        return TreeNode(feature=best_feat, threshold=best_thresh, left=left_node, right=right_node, value=node_value)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.value is None:
            return 0
        if node.feature is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

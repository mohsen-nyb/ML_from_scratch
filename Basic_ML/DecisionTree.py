import numpy as np

class DecisionTree:
    """
    A simple implementation of a binary classification decision tree using Gini impurity for splitting, with a depth-limited approach.
    
    Attributes:
    max_depth (int): The maximum depth of the tree.
    tree (dict or int): The root of the tree or the predicted class label for leaves.
    """
    
    def __init__(self, max_depth=5):
        """
        Initialize the Decision Tree with a specified maximum depth.
        
        Parameters:
        max_depth (int): The maximum depth of the tree (default is 5).
        """
        self.max_depth = max_depth
        self.tree = None

    def gini(self, y):
        """
        Compute the Gini Impurity for a given set of labels.
        
        Parameters:
        y (array-like): The labels of the data points.
        
        Returns:
        float: The Gini Impurity value.
        """
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()  # Probability distribution of classes
        return 1 - np.sum(p ** 2)  # Gini Impurity formula

    def best_split(self, X, y):
        """
        Find the best feature and threshold to split the dataset using Gini Impurity.
        
        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The labels of the data points.
        
        Returns:
        tuple: The best feature index and threshold for the split.
        """
        best_feature, best_threshold, best_gini = None, None, 1  # Initialize best split
        
        # Iterate through each feature to find the best split
        for feature_id in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_id])  # Unique values of the feature
            
            # Iterate through each threshold for the feature
            for threshold in thresholds:
                # Create masks for the left and right splits based on the threshold
                left_mask = X[:, feature_id] <= threshold
                right_mask = ~left_mask
                
                # Compute Gini Impurity for both left and right splits
                gini_left = self.gini(y[left_mask])
                gini_right = self.gini(y[right_mask])
                
                # Weighted Gini Impurity for the current split
                weighted_gini = (left_mask.sum() * gini_left + right_mask.sum() * gini_right) / len(y)
                
                # Update the best split if a lower Gini Impurity is found
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature, best_threshold = feature_id, threshold
        
        return best_feature, best_threshold

    def grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree by splitting the data and creating child nodes.
        
        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The labels of the data points.
        depth (int): The current depth of the tree (default is 0).
        
        Returns:
        dict or int: The tree structure as a dictionary, or the predicted class label for leaf nodes.
        """
        # Stop splitting if max depth is reached or if all labels are the same
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))  # Return the most frequent class label
        
        # Find the best feature and threshold to split on
        feature, threshold = self.best_split(X, y)
        
        # If no valid split is found, return the majority class
        if feature is None:
            return np.argmax(np.bincount(y))
        
        # Create masks for left and right splits
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Recursively grow the left and right subtrees
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.grow_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self.grow_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        """
        Build the decision tree by training on the provided dataset.
        
        Parameters:
        X (array-like): The feature matrix for training.
        y (array-like): The labels for training.
        """
        self.tree = self.grow_tree(X, y)

    def predict_single(self, x, tree):
        """
        Predict the class label for a single sample by traversing the tree.
        
        Parameters:
        x (array-like): A single sample.
        tree (dict or int): The decision tree or a class label (for leaf nodes).
        
        Returns:
        int: The predicted class label.
        """
        # If the current node is a decision node, traverse left or right
        if isinstance(tree, dict):
            if x[tree["feature"]] <= tree["threshold"]:
                return self.predict_single(x, tree["left"])
            else:
                return self.predict_single(x, tree["right"])
        else:
            return tree  # Leaf node, return the class label

    def predict(self, X):
        """
        Predict the class labels for a set of samples.
        
        Parameters:
        X (array-like): The feature matrix for prediction.
        
        Returns:
        array: An array of predicted class labels.
        """
        return np.array([self.predict_single(x, self.tree) for x in X])


    def evaluate(self, X, y):
        """
        Evaluate the performance of the decision tree on the test dataset.
        
        Parameters:
        X (array-like): The feature matrix for testing.
        y (array-like): The true labels for the test set.
        
        Returns:
        float: The accuracy of the model on the test data.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

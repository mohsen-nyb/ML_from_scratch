import numpy as np
from collections import Counter

class KNNClassifier:
    """
    A simple k-Nearest Neighbors (kNN) classifier implemented from scratch.
    """
    def __init__(self, k_neighbors=3):
        """
        Initializes the kNN classifier.
        
        Parameters:
        k (int): The number of nearest neighbors to consider.
        """
        self.k = k_neighbors
        self.x_train = None
        self.y_train = None
    
    def fit(self, x_train, y_train):
        """
        Stores the training dataset.
        
        Parameters:
        x_train (numpy.ndarray): The feature matrix for training.
        y_train (numpy.ndarray): The target labels for training.
        """
        self.x_train = x_train
        self.y_train = y_train
    
    def euclidean_distance(self, x1, x2):
        """
        Computes the Euclidean distance between two data points.
        
        Parameters:
        x1 (numpy.ndarray): The first data point.
        x2 (numpy.ndarray): The second data point.
        
        Returns:
        float: The computed Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def get_most_common_label(self, K_neighbors_labels):
        """
        Determines the most common label among k-nearest neighbors.
        
        Parameters:
        K_neighbors_labels (list): A list of labels of the k-nearest neighbors.
        
        Returns:
        int: The most common class label.
        """
        label_dict = {}
        for label in K_neighbors_labels:
            label_dict[label] = label_dict.get(label, 0) + 1
        most_common_label = max(label_dict.keys(), key=lambda x: label_dict[x])

        #another more efficient way using collection.Counter method 
        #label_counts = Counter(K_neighbors_labels)
        #most_common_label = label_counts.most_common(1)[0][0]

        return most_common_label
    
    def predict_single(self, x):
        """
        Predicts the class label for a single input data point.
        
        Parameters:
        x (numpy.ndarray): The input data point.
        
        Returns:
        int: The predicted class label.
        """
        distances = [self.euclidean_distance(x, x_train) for x_train in self.x_train]
        k_neighbor_indices = np.argsort(distances)[:self.k]
        K_neighbors_labels = [self.y_train[i] for i in k_neighbor_indices]
        return self.get_most_common_label(K_neighbors_labels)
    
    def predict(self, x_test):
        """
        Predicts class labels for a set of input data points.
        
        Parameters:
        x_test (numpy.ndarray): The feature matrix for testing.
        
        Returns:
        numpy.ndarray: The predicted class labels.
        """
        return np.array([self.predict_single(x) for x in x_test])
    
    def evaluate(self, x_test, y_test):
        """
        Evaluates the model's accuracy on a test dataset.
        
        Parameters:
        x_test (numpy.ndarray): The feature matrix for testing.
        y_test (numpy.ndarray): The actual target labels.
        
        Returns:
        float: The accuracy of the model on the test dataset.
        """
        y_pred = self.predict(x_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy




class KNNRegressor:
    """
    A simple k-Nearest Neighbors (kNN) regressor implemented from scratch.
    """
    def __init__(self, k=3):
        """
        Initializes the kNN regressor.
        
        Parameters:
        k (int): The number of nearest neighbors to consider.
        """
        self.k = k
        self.x_train = None
        self.y_train = None
    
    def fit(self, x_train, y_train):
        """
        Stores the training dataset.
        
        Parameters:
        x_train (numpy.ndarray): The feature matrix for training.
        y_train (numpy.ndarray): The target values for training.
        """
        self.x_train = x_train
        self.y_train = y_train
    
    def euclidean_distance(self, x1, x2):
        """
        Computes the Euclidean distance between two data points.
        
        Parameters:
        x1 (numpy.ndarray): The first data point.
        x2 (numpy.ndarray): The second data point.
        
        Returns:
        float: The computed Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict_single(self, x):
        """
        Predicts the target value for a single input data point.
        
        Parameters:
        x (numpy.ndarray): The input data point.
        
        Returns:
        float: The predicted target value.
        """
        distances = [self.euclidean_distance(x, x_train) for x_train in self.x_train]
        k_neighbor_indices = np.argsort(distances)[:self.k]
        K_neighbors_values = [self.y_train[i] for i in k_neighbor_indices]
        return np.mean(K_neighbors_values)
    
    def predict(self, x_test):
        """
        Predicts target values for a set of input data points.
        
        Parameters:
        x_test (numpy.ndarray): The feature matrix for testing.
        
        Returns:
        numpy.ndarray: The predicted target values.
        """
        return np.array([self.predict_single(x) for x in x_test])
    
    def evaluate(self, x_test, y_test):
        """
        Evaluates the model's performance using Mean Squared Error (MSE).
        
        Parameters:
        x_test (numpy.ndarray): The feature matrix for testing.
        y_test (numpy.ndarray): The actual target values.
        
        Returns:
        float: The Mean Squared Error of the model on the test dataset.
        """
        y_pred = self.predict(x_test)
        mse = np.mean((y_pred - y_test) ** 2)
        return mse

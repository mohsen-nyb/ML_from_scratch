import numpy as np

class KMeans:
    """
    A simple k-Means clustering algorithm implemented from scratch.
    """
    def __init__(self, k=5, max_iter=100, tol=1e-4, random_state=42):
        """
        Initializes the k-Means clustering model.
        
        Parameters:
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        random_state (int): Random seed for reproducibility.
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol 
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def init_centroids(self, X):
        """
        Initializes centroids by randomly selecting k data points.
        
        Parameters:
        X (numpy.ndarray): The dataset.
        
        Returns:
        numpy.ndarray: Initial centroids.
        """
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[random_indices]
    
    def init_centroids_plus_plus(self, X):
        """
        Initializes centroids using the k-Means++ method for better convergence.
        
        Parameters:
        X (numpy.ndarray): The dataset.
        
        Returns:
        numpy.ndarray: Initial centroids.
        """
        np.random.seed(self.random_state)
        centroids = [X[np.random.randint(X.shape[0])]]
        
        for _ in range(self.k - 1):
            distances = np.array([min(self.euclidean_distance(x, c) ** 2 for c in centroids) for x in X])
            probabilities = distances / np.sum(distances)
            next_centroid = X[np.random.choice(X.shape[0], p=probabilities)]
            centroids.append(next_centroid)
        
        return np.array(centroids)
    
    def euclidean_distance(self, x1, x2):
        """
        Computes the Euclidean distance between two points.
        
        Parameters:
        x1 (numpy.ndarray): First point.
        x2 (numpy.ndarray): Second point.
        
        Returns:
        float: Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def assign_labels(self, X):
        """
        Assigns each data point to the nearest centroid.
        
        Parameters:
        X (numpy.ndarray): The dataset.
        
        Returns:
        numpy.ndarray: Cluster labels.
        """
        distances = np.array([[self.euclidean_distance(x, c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """
        Computes new centroids as the mean of assigned points.
        
        Parameters:
        X (numpy.ndarray): The dataset.
        labels (numpy.ndarray): Cluster labels.
        
        Returns:
        numpy.ndarray: Updated centroids.
        """
        return np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i] for i in range(self.k)])
    
    def fit(self, X):
        """
        Fits the k-Means clustering model to the data.
        
        Parameters:
        X (numpy.ndarray): The dataset.
        """
        self.centroids = self.init_centroids_plus_plus(X)
        
        for _ in range(self.max_iter):
            labels = self.assign_labels(X)
            new_centroids = self.update_centroids(X, labels)
            
            if np.sum([self.euclidean_distance(self.centroids[i], new_centroids[i]) for i in range(self.k)]) < self.tol:
                break
            
            self.centroids = new_centroids
            self.labels = labels
    
    def predict(self, X):
        """
        Predicts cluster labels for new data points.
        
        Parameters:
        X (numpy.ndarray): The dataset.
        
        Returns:
        numpy.ndarray: Cluster labels.
        """
        return self.assign_labels(X)
    
    def evaluate(self, X):
        """
        Computes the inertia (sum of squared distances to closest centroid).
        
        Parameters:
        X (numpy.ndarray): The dataset.
        
        Returns:
        float: Inertia score.
        """
        labels = self.assign_labels(X)
        return np.sum([self.euclidean_distance(X[i], self.centroids[labels[i]]) ** 2 for i in range(X.shape[0])])

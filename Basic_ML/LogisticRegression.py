import numpy as np
from tqdm import tqdm

class LogisticRegression:
    """
    A simple logistic regression model implemented from scratch using gradient descent.
    """
    def __init__(self, learning_rate=0.001, num_epochs=200):
        """
        Initializes the Logistic Regression model.
        
        Parameters:
        learning_rate (float): The step size for gradient descent.
        num_epochs (int): The number of iterations for training.
        """
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.weight = None
        self.bias = None
    
    def init_params(self, X):
        """
        Initializes model parameters (weights and bias) randomly.
        
        Parameters:
        X (numpy.ndarray): The input feature matrix.
        """
        self.weight = np.random.randn(X.shape[1])
        self.bias = np.random.randn(1)
    
    def get_normalized(self, X):
        """
        Normalizes the input features using mean and standard deviation.
        
        Parameters:
        X (numpy.ndarray): The input feature matrix.
        
        Returns:
        numpy.ndarray: The normalized feature matrix.
        """
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (X - np.mean(X, axis=0)) / std
    
    def sigmoid(self, z):
        """
        Computes the sigmoid activation function.
        
        Parameters:
        z (numpy.ndarray): The input array.
        
        Returns:
        numpy.ndarray: The sigmoid of the input.
        """
        return 1 / (1 + np.exp(-z))
    
    def compute_bce_loss(self, y_pred, y_true):
        """
        Computes the binary cross-entropy loss.
        
        Parameters:
        y_pred (numpy.ndarray): The predicted probabilities.
        y_true (numpy.ndarray): The actual target values.
        
        Returns:
        float: The computed loss.
        """
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def compute_gradient(self, X, y_pred, y_true):
        """
        Computes the gradients of the loss with respect to weights and bias.
        
        Parameters:
        X (numpy.ndarray): The input feature matrix.
        y_pred (numpy.ndarray): The predicted probabilities.
        y_true (numpy.ndarray): The actual target values.
        
        Returns:
        tuple: Gradients (dw, db) for weight and bias.
        """
        m = X.shape[0]
        dw = (1 / m) * np.dot(X.T, (y_pred - y_true))
        db = (1 / m) * np.sum(y_pred - y_true)
        return dw, db
    
    def update_params(self, dw, db):
        """
        Updates the model parameters using gradient descent.
        
        Parameters:
        dw (numpy.ndarray): Gradient of the loss w.r.t. weights.
        db (float): Gradient of the loss w.r.t. bias.
        """
        self.weight -= self.lr * dw
        self.bias -= self.lr * db
    
    def predict_proba(self, X):
        """
        Predicts probability values for given input features.
        
        Parameters:
        X (numpy.ndarray): The input feature matrix.
        
        Returns:
        numpy.ndarray: The predicted probabilities.
        """
        X = self.get_normalized(X)
        return self.sigmoid(np.dot(X, self.weight) + self.bias)
    
    def predict(self, X):
        """
        Predicts binary class labels for given input features.
        
        Parameters:
        X (numpy.ndarray): The input feature matrix.
        
        Returns:
        numpy.ndarray: The predicted class labels (0 or 1).
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def fit(self, X, Y):
        """
        Trains the model using gradient descent.
        
        Parameters:
        X (numpy.ndarray): The input feature matrix.
        Y (numpy.ndarray): The actual target values.
        """
        X = self.get_normalized(X)
        self.init_params(X)
        
        for epoch in tqdm(range(self.num_epochs)):
            # Forward propagation
            Y_pred = self.predict_proba(X)
            loss = self.compute_loss(Y_pred, Y)
            
            # Backward propagation
            dw, db = self.compute_gradient(X, Y_pred, Y)
            self.update_params(dw, db)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch + 1} / {self.num_epochs} - Loss: {loss:.6f}")
    
    def evaluate(self, X, Y):
        """
        Evaluates the model performance on a test dataset.
        
        Parameters:
        X (numpy.ndarray): The input feature matrix.
        Y (numpy.ndarray): The actual target values.
        
        Returns:
        float: The binary cross-entropy loss on the test dataset.
        """
        Y_pred = self.predict_proba(X)
        return self.compute_loss(Y_pred, Y)

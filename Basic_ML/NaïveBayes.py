# Gaussian Naïve Bayes (For Continuous Data)

import numpy as np

class GaussianNB:
  def fit(self, X, y):
    """Train the Gaussian Naïve Bayes model."""
    self.classes = np.unique(y)
    self.means = {}
    self.variences = {}
    self.priors = {}

    for c in self.classes:
      X_c = X[y == c]
      self.means[c] = np.mean(X_c, axis = 0)
      self.variences[c] = np.var(X_c, axis = 0) + 1e-6 #avoiding zero devision
      self.priors[c] = len(X_c) / len(X)

  def Guassian_pdf(self, x, mean, var):
    """Compute the probability density function of Gaussian distribution."""
    coef = 1 / np.sqrt(2 * np.pi * var)
    exp = np.exp(- (x - mean) ** 2 / (2 * var))
    return coef * exp

  def predict_single(self, x):
    """Predict the class label for a single sample."""
    posteriors = {}
    for c in self.classes:
      prior = np.log(self.priors[c])
      likelihood = np.sum(np.log(self.Guassian_pdf(x, self.means[c], self.variences[c])))
      posterior = prior + likelihood
      posteriors[c] = posterior

    return max(posteriors, key=posteriors.get)

  def predict(self, X):
    """Predict the class labels for a set of samples."""
    return np.array([self.predict_single(x) for x in X])



# Example Usage:
X_train = np.array([[1.5, 2.1], [2.3, 3.1], [1.2, 2.0], [5.4, 6.7], [5.9, 6.1], [6.0, 6.2]])
y_train = np.array([0, 0, 0, 1, 1, 1])

model = GaussianNB()
model.fit(X_train, y_train)
X_test = np.array([[2.0, 2.5], [5.5, 6.0]])
print(model.predict(X_test))  # Expected output: [0, 1]



class MultinomialNB:
  def fit(self, X, y):
    """Train the Multinomial Naïve Bayes model."""
    self.classes = np.unique(y)
    self.word_counts = {}
    self.class_totals = {}
    self.priors = {}

    for c in self.classes:
      X_c = X[y == c]
      self.word_counts[c] = np.sum(X_c, axis=0) + 1  # Laplace smoothing
      self.class_totals[c] = np.sum(self.word_counts[c])
      self.priors[c] = np.sum(y == c) / len(y)

  def predict_single(self, x):
    """Predict a single sample."""
    posteriors = {}
    for c in self.classes:
      likelihood = np.sum(np.log(self.word_counts[c] / self.class_totals[c]) * x)
      prior = np.log(self.priors[c])
      posterior = prior + likelihood
      posteriors[c] = posterior

    return max(posteriors, key=posteriors.get)

  def predict(self, X):
    """Predict multiple samples."""
    return np.array([self.predict_single(x) for x in X])



# Example Usage:
X_train = np.array([[2, 1, 0, 0], [3, 0, 1, 0], [0, 2, 1, 3], [0, 0, 2, 4]])
y_train = np.array([0, 0, 1, 1])

model = MultinomialNB()
model.fit(X_train, y_train)
X_test = np.array([[1, 5, 5, 5], [5, 5, 5, 3]])
print(model.predict(X_test))  # Expected output: [0, 1]


class BernoulliNB:
    def fit(self, X, y):
        """Train the Bernoulli Naïve Bayes model."""
        self.classes = np.unique(y)
        self.feature_probs = {}
        self.priors = {c: np.log(np.sum(y == c) / len(y)) for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]
            self.feature_probs[c] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)  # Laplace smoothing
    
    def predict(self, X):
        """Predict class labels for input samples."""
        y_pred = [self.predict_single(x) for x in X]
        return np.array(y_pred)
    
    def predict_single(self, x):
        """Predict a single sample."""
        posteriors = {}

        for c in self.classes:
            probs = self.feature_probs[c]
            likelihood = np.sum(np.log(probs) * x + np.log(1 - probs) * (1 - x))
            posteriors[c] = self.priors[c] + likelihood
        
        return max(posteriors, key=posteriors.get)

# Example Usage:
X_train = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
y_train = np.array([0, 0, 1, 1])
model = BernoulliNB()
model.fit(X_train, y_train)
X_test = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
print(model.predict(X_test))  # Expected output: [0, 1]

import numpy as np # type: ignore
import matplotlib.pyplot as plt #type: ignore
from gcv import find_optimal_alpha, plot_gcv
from utils import ensure_intercept

class RidgeRegression:
  def __init__(self):
    self.theta = None
    self.X_b = None
    
  def fit(self, X, y, alphas_to_test):
    # Add a column of ones to X for intercept if required
    self.X_b = ensure_intercept(X)
    
    # Get lambda by finding value that minimises GCV (start with logspace, then use linspace)
    lmbda, gcv_scores = find_optimal_alpha(X, y, alphas_to_test)
    
    # Plot GCV scores for further refinement
    plot_gcv(gcv_scores, alphas_to_test, lmbda)
    
    # Use the normal equation
    self.theta = np.linalg.pinv(self.X_b.T @ self.X_b + lmbda * np.identity(X.shape[1])) @ self.X_b.T @ y
    
    return self.theta
    
  def predict(self, X):
    # Predict values
    predictions_vector = self.X_b @ self.theta
    return predictions_vector
  
  
class LinearRegression:
  def __init__(self):
    self.theta = None
    self.X_b = None
    
  def fit(self, X, y):
    # Add a column of ones to X for intercept if required
    self.X_b = ensure_intercept(X)
    
    # Calculate the coefficients using normal equation
    self.theta = np.linalg.pinv(self.X_b.T @ self.X_b) @ self.X_b.T @ y
    
    return self.theta
    
  def predict(self, X): 
    # Predict values
    predictions_vector = self.X_b @ self.theta
    
    return predictions_vector
  
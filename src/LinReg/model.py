import numpy as np # type: ignore
import matplotlib.pyplot as plt #type: ignore

from .gcv import find_optimal_alpha, plot_gcv
from .utils import ensure_intercept

class RidgeRegression:
  def __init__(self):
    self.theta = None
    self.X_b = None
    self.y = None
    self.samples = None
    self.predictors_num = None
    self.order = None
    
  def add_polynomial_features(self, X, order):
    """
    Expands matrix X to include polynomial terms up to the specified degree. If X has multiple columns, it expands each one
    """
    
    X_poly = X.copy()
    if X_poly.ndim == 1:
      X_poly = X_poly.reshape(-1, 1)
            
    for d in range(2, order + 1): 
      X_poly = np.c_[X_poly, X ** d]
    return X_poly
        
    
  def fit(self, X, y, alphas_to_test, order = 1):
    # Update sample size and number of predictors
    self.samples = np.size(y)
    try:
      self.predictors_num = np.shape(X)[1]
    except IndexError:
      self.predictors_num = 1
    
    # Update self.y with training data
    self.y = y
    
    # Update self.order with order passed into function
    self.order = order
    
    # Expand features if order > 1
    if self.order > 1:
      features = self.add_polynomial_features(X, order)
    else:
      features = X
    
    # Add a column of ones to X for intercept if required
    self.X_b = ensure_intercept(features)
    
    # Get lambda by finding value that minimises GCV (start with logspace, then use linspace)
    lmbda, gcv_scores = find_optimal_alpha(features, y, alphas_to_test)
    
    # Plot GCV scores for further refinement
    plot_gcv(gcv_scores, alphas_to_test, lmbda)
    
    # Use the normal equation
    identity = np.identity(self.X_b.shape[1])
    identity[0, 0] = 0 # Don't penalise intercept
    self.theta = np.linalg.pinv(self.X_b.T @ self.X_b + lmbda * identity) @ self.X_b.T @ y
    
    return self.theta
    
  def predict(self, X):
    # Add polynomial features if order > 1
    if self.order > 1:
      X_p = self.add_polynomial_features(X, self.order)
    else:
      X_p = X
    
    # Add a column of ones to X for intercept if required
    X_b_prediction = ensure_intercept(X_p)
    
    # Predict values
    predictions_vector = X_b_prediction @ self.theta
    return predictions_vector
  
  
class LinearRegression:
  def __init__(self):
    self.theta = None
    self.X_b = None
    self.y = None
    self.samples = None
    self.predictors_num = None
    
  def fit(self, X, y):
    # Update sample size and number of predictors
    self.samples = np.size(y)
    try:
      self.predictors_num = np.shape(X)[1]
    except IndexError:
      self.predictors_num = 1
    
    # Update self.y with training data 
    self.y = y
    
    # Add a column of ones to X for intercept if required
    self.X_b = ensure_intercept(X)
    
    # Calculate the coefficients using normal equation
    self.theta = np.linalg.pinv(self.X_b.T @ self.X_b) @ self.X_b.T @ y
    
    return self.theta
    
  def predict(self, X): 
    # Add column of ones if not included
    X_b = ensure_intercept(X)
    
    # Predict values
    predictions_vector = X_b @ self.theta
    
    return predictions_vector
  
  
class PolynomialRegression:
  def __init__(self):
    self.theta = None
    self.X_b = None
    self.y = None
    self.samples = None
    self.predictors_num = None
    self.order = None
    
  def add_polynomial_features(self, X, order):
    """
    Expands matrix X to include polynomial terms up to the specified degree. If X has multiple columns, it expands each one
    """
    
    X_poly = X.copy()
    if X_poly.ndim == 1:
      X_poly = X_poly.reshape(-1, 1)
    
    # Concatenate columns with polynomial terms
    for d in range(2, order + 1): 
        X_poly = np.c_[X_poly, X ** d]
        
    return X_poly
    
  def fit(self, X, y, order = 2):
    # Update sample size and number of predictors
    self.samples = np.size(y)
    try:
      self.predictors_num = np.shape(X)[1]
    except IndexError:
      self.predictors_num = 1
    
    # Updata self.y with training data
    self.y = y
    
    # Update order if given
    self.order = order
    
    # Add polynomial features
    X_poly = self.add_polynomial_features(X, self.order)
    
    # Add a column of ones to X for intercept if required 
    self.X_b = ensure_intercept(X_poly)
    
    # Calculate the coefficients using normal equation
    self.theta = np.linalg.pinv(self.X_b.T @ self.X_b) @ self.X_b.T @ y
    
    return self.theta
  
  def predict(self, X):
    # Add polynomial features
    X_poly = self.add_polynomial_features(X, self.order)
    
    # Add a column of ones to X for intercept if required
    X_b = ensure_intercept(X_poly)
  
    # Predict values
    predictions_vector = X_b @ self.theta
    
    return predictions_vector
  
  
  
class RegressionStatistics:
  def __init__(self, model):
    self.model = model
      
    # Call function to calculate statistics
    self.calculate_regression_statistics()
      
  def __str__(self):
    return (
        f"Regression Statistics:\n"
        f"----------------------\n"
        f"MSE:                   {self.MSE:.4f}\n"
        f"Residual Std Error:    {self.RSE:.4f}\n"
        f"Multiple R^2:          {self.multiple_r_squared:.4f}\n"
        f"Adjusted R^2:          {self.adjusted_r_squared:.4f}\n"
        f"F-Statistic:           {self.f_statistic:.4f}"
    )

    
  def calculate_regression_statistics(self):
    # Get key statistics/fitted values from model
    y_true = self.model.y
    y_mean = np.mean(y_true)
    n = self.model.samples # Total sample
    k = self.model.predictors_num # Number of predictors
    y_fitted = self.model.predict(self.model.X_b)
     
    # Calculate key sum of squares 
    ss_residuals = np.sum((y_true - y_fitted) ** 2)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_regression = np.sum((y_fitted - y_mean) ** 2)
       
    # Calculate MSE
    self._MSE = ss_residuals / (n - k -1)
    
    # Calculate RSE (residual standard error)
    self._RSE = np.sqrt(self.MSE)
      
    # Calculate mutliple_r_squared
    self._multiple_r_squared = 1 - (ss_residuals / ss_total)
    
    # Calculate adjusted r_squared
    self._adjusted_r_squared = 1 - ((1 - self.multiple_r_squared) * (n - 1) / (n - k - 1))
     
    # Calculate F-Statistics
    df = n - k - 1 # degrees of freedom
    MSR = (ss_regression) / k
    
    self._f_statistic = MSR / self._MSE
      
  # Getters to access statistics
  @property
  def MSE(self):
    return self._MSE
      
  @property
  def RSE(self):
    return self._RSE 
      
  @property
  def multiple_r_squared(self):
    return self._multiple_r_squared
    
  @property
  def adjusted_r_squared(self):
    return self._adjusted_r_squared
        
  @property
  def f_statistic(self):
    return self._f_statistic  
  
  
  

      
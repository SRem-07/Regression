import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utils import ensure_intercept


def ridge_regression_gcv_score(X, y, alpha):
  """
  Calculates the Generalised Cross-Validation (GCV) score

  Args: 
    X (np.ndarray): The feature matrix (n_samples, n_features)
    y (np.ndarray): The target vector (n_samples)
    alpha (float): The ridge regularisation parameter (lambda)

  Returns:
    float: The GCV score for the given alpha
  """
  # Check if a column of ones exist, if not add ones to start of X
  X = ensure_intercept(X)
  
  # Get dimensions of the matrix n (rows/samples), p (columns/features)
  n, p = X.shape
  
  # Compute the coefficients (beta_hat) using the normal equation
  XTX = X.T @ X
  # Add regularisation term
  XTX_reg = XTX + alpha * np.eye(p)
  # Compute the inverse
  try: 
    inv_XTX_reg = np.linalg.pinv(XTX_reg)
  except np.linag.LinAlgError:
    return np.inf
  
  beta_hat = inv_XTX_reg @ X.T @ y
  
  # Calculate predicted values and residuals
  y_pred = X @ beta_hat
  residuals = y_pred - y
  
  # Calculate Residual Sum of Squares (RSS)
  rss = np.sum(residuals ** 2)
  
  # Calculate the trace(H)
  trace_H = np.trace(inv_XTX_reg @ XTX)
  
  # Calculate the GCV score
  gcv_score = (rss / n) / (1 - trace_H / n)**2
  
  # Return the GCV score
  return gcv_score

def plot_gcv(gcv_scores, alphas_to_test, lmbda):
  """
  Plot the GCV scores against alphas in order to further refine alpha value
  
  Parameters:
    gcv_scores (np.ndarray): Array of all gcv scores for all alphas tested
    alphas_to_test (np.ndarray): Array of all alphas tested to minimise gcv
    lmbda (float): Value that minimises gcv the furthers
    
  Returns:
    None
      The function shows a plot.  
  """
  plt.figure(figsize=(10, 6))
  plt.plot(alphas_to_test, gcv_scores, 'b-', linewidth=2, label='GCV Score')
  
  # Use 'lmbda' (the argument), not 'best_a'
  plt.axvline(lmbda, color='red', linestyle='--', label=f'Optimal $\lambda$ = {lmbda:.2f}')
  
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Regularization Parameter (Alpha)')
  plt.ylabel('GCV Score')
  plt.title('GCV Optimization for Ridge Regression')
  plt.legend()
  plt.grid(True, which="both", ls="-", alpha=0.5)
  plt.show() # This ensures the plot opens immediately

  print(f"The optimal alpha found is: {lmbda:.4f}")
  


def find_optimal_alpha(X, y, alphas_to_test):
  """
  Finds the optimal alpha by minimizing the GCV score across a list of alphas.

  Args:
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target vector.
    alphas (list or np.ndarray): A list of alpha values to test.

  Returns:
    optimal_alpha (float): The optimal alpha value.
  """
  
  gcv_scores = []
  
  # Iterate through alphas to test
  for alpha in alphas_to_test:
    score = ridge_regression_gcv_score(X, y, alpha)
    gcv_scores.append(score)
  
  # Make an array 
  gcv_scores = np.array(gcv_scores)
  
  # Find alpha value that minimises gcv
  optimal_alpha = alphas_to_test[np.argmin(gcv_scores)]
  
  return optimal_alpha, gcv_scores
    
  
  
if __name__ == '__main__':
  # 1. Generate Synthetic Data
  np.random.seed(42)
  n_samples, n_features = 100, 10
  X_raw = np.random.randn(n_samples, n_features)
  # True weights + intercept
  true_beta = np.random.randn(n_features + 1) 
  X_with_ones = np.c_[np.ones((n_samples, 1)), X_raw]
  y = X_with_ones @ true_beta + np.random.normal(0, 1.5, size=n_samples)

  # 2. Define Alphas (log scale is best for regularisation)
  alphas = np.logspace(-3, 5, 50)

  # 3. Find Optimal Alpha
  def find_optimal_alpha(X, y, alphas_to_test):
      scores = [ridge_regression_gcv_score(X, y, a) for a in alphas_to_test]
      scores = np.array(scores)
      best_alpha = alphas_to_test[np.argmin(scores)]
      return best_alpha, scores

  best_a, all_scores = find_optimal_alpha(X_raw, y, alphas)


  
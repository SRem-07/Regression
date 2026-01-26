import numpy as np
import pytest 
from LinReg import LinearRegression, RidgeRegression, PolynomialRegression, RegressionStatistics, Plot

def test_linear_fit():
  """
    Test if the model can recover a simple linear relationship
  """
  x = np.array([[1], [2], [3], [4], [5]])
  y = np.array([2, 4, 6, 8, 10]) # y = 3x
  
  # Fit model using RidgeRegression
  model = RidgeRegression()
  model.fit(x, y, alphas_to_test = [1e-10]) # Small alpha so it behaves like OLS
  
  prediction = model.predict([[6]])
  
  assert np.isclose(prediction[0], 12, atol = 1e-5)
  
  
def test_polynomial_order():
  """
    Test if the model correctly identifies higher-order features
  """
  X = np.array([[1], [2], [3]])
  model = PolynomialRegression()
  poly_features = model.add_polynomial_features(X, order = 3)
  
  # Check fi the shape is 3x3 -? [x, x^2, x^3]
  assert poly_features.shape == (3, 3)
  # Check if the second row is [2, 4, 8]
  assert np.array_equal(poly_features[1], [2, 4, 8])
  
def test_gcv_selection():
  """
  Test if GCV successfully selects an alpha from the list
  """
  X = np.random.randn(10, 1)
  y = 2 * X.flatten() + 1
  alphas = [0.1, 1.0, 10.0]
  
  model = RidgeRegression()
  model.fit(X, y, alphas_to_test = alphas)
  
  # Check if theta was calculated
  assert model.theta is not None
  assert len(model.theta) == 2 # [intercept, x1]
  
  
def test_polynomial_fit():
  """
    Test if the model can fit a simple polynomial pattern (cubic)
  """
  x = np.linspace(1, 100, 50)
  y = 2 * (x ** 3) 
  
  model = PolynomialRegression()
  model.fit(x, y , 3)
  
  prediction = model.predict([[5]])
  
  assert np.isclose(prediction[0], 250, atol = [1e-10])

  
def test_regression_statistics_perfect_fit():
  """
    Test statistics on a perfectly linaer relationship
  """
  # y = 2x + 5
  x = np.array([[1], [2], [3], [4], [5]])
  y = 2 * x + 5
  
  model = LinearRegression()
  model.fit(x, y)
  
  stats = RegressionStatistics(model)
  
  assert np.isclose(stats.multiple_r_squared, 1.0), "R^2 should be 1.0 for a perfect fit"
  assert np.isclose(stats.MSE, 0.0, atol = 1e-10), "MSE should be 0 for perfect fit"
  assert stats.f_statistic > 1e10, "F-statistic should be extremely high for perfect fit"
  
def test_regression_statistics_noisy():
  """
    Test statistics on noisy data to ensure MSE and R^2 are sensible
  """
  np.random.seed(42)
  x = np.linspace(0, 20, 100).reshape(-1, 1)
  # y = 3x + 2 + noise
  noise = np.random.normal(0, 1, 100)
  y = 3 * x.flatten() + 2 + noise
  
  model = LinearRegression()
  model.fit(x, y)
  
  stats = RegressionStatistics(model)
  
  assert 0.9 < stats.multiple_r_squared < 1.0
  assert stats.adjusted_r_squared < stats.multiple_r_squared
  assert 0.5 < stats.MSE < 1.5
  
  
def test_adjusted_r_squared_penalisation():
  """
    Test if Adjusted R^2 correctly penalises useless predictors
  """
  np.random.seed(42)
  X = np.random.randn(40, 1)
  # y = 5x + noise
  y = 5 * X.flatten() + np.random.normal(0, 1, 40)
  
  # Fit model 1 with just the real predictor
  model1 = LinearRegression()
  model1.fit(X, y)
  stats1 = RegressionStatistics(model1)
  
  # Fit model 2 with 5 columns of random noise
  X_with_noise = np.c_[X, np.random.randn(40, 15)]
  model2 = LinearRegression()
  model2.fit(X_with_noise, y)
  stats2 = RegressionStatistics(model2)
  
  # Adjusted R^%2 should be lower for the noisy model because extra features don't explain variance
  assert stats2.adjusted_r_squared < stats1.adjusted_r_squared
  print(f"Clean Adj R2: {stats1.adjusted_r_squared:.4f}")
  print(f"Noise Adj R2: {stats2.adjusted_r_squared:.4f}")
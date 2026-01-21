from plot_regression import Plot
from model import LinearRegression
import numpy as np

# 1. Create dummy data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Fit model
model = LinearRegression()
coefficients = model.fit(X, y)

# 3. Plot with one call
Plot.plot_2D_linear_trend(X, y, coefficients, )
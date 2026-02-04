# LinReg Library

A specialized Python library for **Linear, Polynomial, and Ridge Regression**. This package features built-in **Generalised Cross-Validation (GCV)** to automatically find the optimal regularization parameter ($\lambda$).

## Features
* **Polynomial Expansion**: Transform linear features into higher-order terms automatically.
* **Ridge Regularization**: Prevents overfitting by adding an $L_2$ penalty: $\|y - X\theta\|^2_2 + \lambda \|\theta\|^2_2$.
* **Automated Tuning**: Uses GCV to select the optimal $\lambda$ without the computational overhead of standard cross-validation.
* **Diagnostics & Visualization**: Built-in tools for 2D/3D plotting, Residual analysis, and Q-Q plots.

## Tech Stack
* NumPy
* Matplotlib
* SciPy

## Overview
This linear regression library solely uses NumPy to perform regressions, as well as Matplotlib to create all plots. This project was developed as my first major coding project to bridge the gap between statistical theory, my current area of study, mathematics, and programming.

This project developed from a simple coding exercise and I applied and learnt some new regression techniques including: the Normal Equation, Ridge Regression, Polynomial Regression and Generalised Cross-Validation. 

## Usage
### Cloning the Repository
If you would like to contribute or run the tests, clone the repository using:
```bash
git clone https://github.com/SRem-07/Regression.git
cd LinReg-Library
pip install -e .
```

## Quick Start
``` Python
import numpy as np
from LinReg import RidgeRegression, RegressionStatistics

# Generate data
X = np.random.randn(100, 1)
y = 2 * X.flatten() + np.random.normal(0, 1, 100)

# Fit model with GCV
model = RidgeRegression(order=1)
model.fit(X, y, alphas_to_test=np.logspace(-3, 3, 20))

# View Statistics
stats = RegressionStatistics(model)
print(stats)
```
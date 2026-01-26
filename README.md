# LinearReg Library

A specialized Python library for **Linear, Polynomial, and Ridge Regression**. This package features built-in **Generalised Cross-Validation (GCV)** to automatically find the optimal regularization parameter ($\lambda$).

## Features
* **Polynomial Expansion**: Easily transform linear features into higher-order terms.
* **Ridge Regularization**: Prevents overfitting by adding an $L_2$ penalty: $\|y - X\beta\|^2 + \lambda \|\beta\|^2$.
* **Automated Tuning**: Uses GCV to select $\lambda$ without needing a separate validation set.
* **Visualization**: 2D and 3D plotting tools to visualize regression surfaces and trendlines.

## Installation
From the project root, install in editable mode:
```bash
pip install -e .
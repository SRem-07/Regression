import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import scipy.stats as stats

class Plot:
  @staticmethod
  def exploratory_plot_2D(X, y):
    """
    Create a 2D plot for inital data exploration
    
    Parameters:
      X (np.ndarray): The feature matrix (n_samples)
      y (np.ndarray): The target vector (n_samples)
      
    Returns: 
      None
        The function displays a plot but does not return a value
    """
    # Create and show a scatter plot
    plt.figure(figsize = (8, 5))
    plt.scatter(X, y, color = 'teal', label = "Data", alpha = 0.6)
    plt.xlabel("Explanatory Variable")
    plt.ylabel("Response Variable")
    plt.legend()
    plt.show()
    
  
  @staticmethod
  def exploratory_plot_3D(X, y):
    """
    Create a 3D plot for inital data exploration
    
    Parameters:
      X (np.ndarray): The feature matrix (n_samples, n_features)
      y (np.ndarray): The target vector (n_samples)
      
    Returns: 
      None
        The function displays a plot but does not return a value
  """
    
    # Slice X into two arrays
    X1, X2 = X[:, 0], X[:, 1]
    
    # Creeate and show 3D scatter plot
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111, projection = '3d')
    
    ax.scatter(X1, X2, y, color = 'teal', alpha = 0.6)
    ax.set_xlabel('Explanatory Variable 1')
    ax.set_ylabel('Explanatory Variable 2')
    ax.set_zlabel('Response Variable')
    plt.show()
    
    
  @staticmethod
  def plot_2D_linear_trend(X, y, model, plot_title = "Scatter Plot", x_label = "Explanatory Variable", y_label = "Response Variable"):
    """
    Create a 2D plot with regression line
    
    Parameters:
      X (np.ndarray): The feature matrix (n_samples)
      y (np.ndarray): The target vector (n_samples)
      beta_hat (np.ndarray): Vector of coefficients from linear regression
      plot_title (string): Title of plot, defaults to 'Scatter Plot'
      x_label (string): label for x-axis, defaults to 'Explanatory Variable'
      y_label (string): Title for y-axis, defaults to 'Response Variable'
      
    Returns: 
      None
        The function displays a plot but does not return a value
  """
    
    # Get range of X and create trendline
    x_range = np.linspace(np.min(X) - 0.5, np.max(X) + 0.5, 100).reshape(-1, 1) # Ensure x_range is a column vector
    y_trendline = model.predict(x_range)
    
    # Create and show scatter plot with trendline
    plt.figure(figsize = (8, 5))
    plt.scatter(X, y, color = 'teal', label = "Data", alpha = 0.6)
    plt.plot(x_range, y_trendline, color = 'red', linestyle = '-', linewidth = 2, label = f"Trendline")
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    
    
  @staticmethod
  def plot_3D_linear_trend(X, y, model, plot_title = "Scatter Plot", exp1_title = "Explanatory Variable 1", exp2_title = "Explanatory Variable 2", predictor_title = "Response Variable"):
    """
    Create a 3D plot for with regression plot
    
    Parameters:
      X (np.ndarray): The feature matrix (n_samples, n_features)
      y (np.ndarray): The target vector (n_samples)
      beta_hat (np.ndarray): Vector of coefficients from linear regression
      plot_title (string): Title of plot, defaults to 'Scatter Plot'
      exp1_title (string): label for x-axis, defaults to 'Explanatory Variable 1'
      exp2_title (string): label for y-axis, defaults to 'Explanatory Variable 2'
      predictor_title (string): Title for z-axis, defaults to 'Response Variable'
      
    Returns: 
      None
        The function displays a plot but does not return a value
  """
  
    # Slice X array into two arrays
    X1, X2 = X[:, 0], X[:, 1]
    
    # Get range of both explanatory variables
    x1_range = np.linspace(np.min(X1), np.max(X2), 20)
    x2_range = np.linspace(np.min(X2), max(X2), 20)
    
    # Create grids and get prediction grid
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
  
    # Flatten grid and predict
    grid_points = np.c_[X1_grid.ravel(), X2_grid.ravel()]
    y_predicted_grid = model.predict(grid_points).reshape(X1_grid.shape)
    
    # Create and show scatter plot with trendline
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111, projection = '3d')
    
    ax.scatter(X1, X2, y, color = 'teal', label = 'Data', alpha = 0.6)
    
    surf = ax.plot_surface(X1_grid, X2_grid, y_predicted_grid, cmap = 'RdPu', alpha = 0.4, linewidth = 0, antialiased = True)
    
    ax.set_title(plot_title)
    ax.set_xlabel(exp1_title)
    ax.set_ylabel(exp2_title)
    ax.set_zlabel(predictor_title)
    
    plt.show()
  
  @staticmethod
  def plot_diagnostics(model):
    """"
      Generates diagnostic plots to check regression assumptions
    """
    # Get data from model
    y_true = model.y
    y_fitted = model.X_b @ model.theta
    
    # Calculate residuals
    residuals = y_true - y_fitted
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    
    # Create residual v fitted plot
    ax1.scatter(y_fitted, residuals, alpha = 0.5, edgecolors = 'k')
    ax1.axhline(0, color = 'red', linestyle = '--')
    ax1.set_title("Residals vs Fitted")
    ax1.set_xlabel("Fitted")
    ax1.set_ylabel("Residuals")
    
    # Create QQ plot
    stats.probplot(residuals, dist = 'norm', plot = ax2)
    ax2.set_title('Normal Q-Q')
    
    plt.tight_layout()
    plt.show()
    
    
    
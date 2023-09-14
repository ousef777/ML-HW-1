import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import display

file_path = 'D3.csv'
sample = pd.DataFrame(pd.read_csv(file_path))

x1 = np.array(sample["X1"])
x2 = np.array(sample["X2"])
x3 = np.array(sample["X3"])
y = np.array(sample["Y"])

def compute_cost(X, y, theta):
    """
    Compute cost for linear regression.

    Parameters:
    X : 2D array where each row represents the training example and each column represent the feature
        m = number of training examples
        n = number of features (including X_0 column of ones)
    y : 1D array of labels/target values for each training example. dimension(m)
    theta : 1D array of fitting parameters or weights. Dimension (n)

    Returns:
    J : Scalar value, the cost
    """
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * len(y)) * np.sum(sqrErrors)
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    """
    Compute the optimal parameters using gradient descent for linear regression.

    Parameters:
    X : 2D array where each row represents the training example and each column represents the feature
        m = number of training examples
        n = number of features (including X_0 column of ones)
    y : 1D array of labels/target values for each training example. dimension(m)
    theta : 1D array of fitting parameters or weights. Dimension (n)
    alpha : Learning rate (scalar)
    iterations : Number of iterations (scalar)

    Returns:
    theta : Updated values of fitting parameters or weights after 'iterations' iterations. Dimension (n)
    cost_history : Array containing the cost for each iteration. Dimension (iterations)
    """

    m = len(y)  # Number of training examples
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors)
        theta -= sum_delta
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

def main(x, title):
    #We walk through the initial steps of building a linear regression model from scratch using NumPy. Let's break down what you're doing:
    #X_0 = np.ones((m, 1)): We're creating a column vector of ones. This will be used as the "bias" term for the linear regression model.
    #X_1 = X.reshape(m, 1): You're reshaping features (X) to make it a 2D array suitable for matrix operations.
    #X = np.hstack((X_0, X_1)): We're horizontally stacking X_0 and X_1 to create final feature matrix X.

    m = len(y)
    X_0 = np.ones((m, 1))
    X_0[:5]

    X_1 = x.reshape(m, 1)
    X_1[:10]

    # Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
    # This will be our final X matrix (feature matrix)
    X = np.hstack((X_0, X_1))
    X[:5]

    theta = np.zeros(2)


    # Lets compute the cost for theta values
    cost = compute_cost(X, y, theta)
    print('The cost for given values of theta_0 and theta_1 =', cost)

    theta = [0., 0.]
    iterations = 200 #1500
    alpha = 0.09    #0.01

    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
    print('Final value of theta =', theta)
    print('cost_history =', cost_history)

    # Assuming that X, y, and theta are already defined
    # Also assuming that X has two columns: a feature column and a column of ones

    # Scatter plot for the training data
    plt.scatter(X[:, 1], y, color='red', marker='+', label='Training Data')

    # Line plot for the linear regression model
    plt.plot(X[:, 1], X.dot(theta), color='green', label='Linear Regression')

    # Plot customizations
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.grid(True)
    plt.xlabel(title)
    plt.ylabel('Y')
    plt.title('Linear Regression Fit')
    plt.legend()

    # Show the plot
    plt.show()

    # Convergence of gradient descent
    plt.plot(range(1, iterations + 1), cost_history, color='blue')
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.grid(True)

    plt.xlabel('Number of iterations')
    plt.ylabel('Cost (J)')
    plt.title('Convergence of gradient descent')

    # Show the plot
    plt.show()

if __name__=="__main__":
    main(x1, 'X1')
    #main(x2, 'X2')
    #main(x3, 'X3')

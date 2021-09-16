import numpy as np
import matplotlib.pyplot as plt

def _derivative(func, x: float, dx: float = 0.001) -> float:
    """
    Estimate the derivative of a function at a given x
    """
    return (func(x + dx) - func(x))/((x+dx) - x)

def hill(x: float, n: int = 2) -> float:
    """
    Evaluate the hill equation at certain x
    """
    return ((x**n)/(1+x**n) - 0.65)

def newton_raphson(func, x: float, tolerance: float = 0.001) -> float:
    """
    Use newton raphson method to estimate the root of an equation
    """
    # estimate zero
    h = func(x) / _derivative(func, x)
    
    # while error greater than tolerance
    while abs(h) > tolerance:
        h = func(x) / _derivative(func, x)
        x = x - h
    
    return x

if __name__ == '__main__':
    root = newton_raphson(hill, 1.5)
    print(f"The root was found to be: {root}")
    
    # plot the function
    x_min = 0
    x_max = 4
    x_vals = np.linspace(x_min, x_max, num=1000)
    y_vals = [hill(x) for x in x_vals]
    
    plt.plot(x_vals, y_vals, 'r-')
    plt.plot(root, 0, 'bo')
    plt.title("Hill function")
    plt.show()
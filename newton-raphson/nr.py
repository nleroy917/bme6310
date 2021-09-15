def _derivative(func, x: float, dx: float = 0.001) -> float:
    """
    Estimate the derivative of a function at a given x
    """
    return (func(x + dx) - func(x))/((x+dx) - x)

def hill(x: float, n: int=4) -> float:
    """
    Evaluate the hill equation at certain x
    """
    return ((x**4)/(1+x**4) - 0.65)

def newton_raphson(func, x: float, tolerance: float = 0.001) -> float:
    """
    Use newton raphson method to estimate the root of an equation
    """
    h = func(x) / _derivative(func, x)
    while abs(h) > tolerance:
        h = func(x) / _derivative(func, x)
        x = x - h
    
    return x

if __name__ == '__main__':
    root = newton_raphson(hill, 1.5)
    print(f"The root was found to be: {root}")
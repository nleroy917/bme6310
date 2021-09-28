import math

# define golden ratio
GOLDEN_RATIO = (math.sqrt(5) - 1) / 2

def f(x: float) -> float:
    """
    function for f(x) = (10/ x)*sin(x)
    """
    return (10/x) * math.sin(x)

def gss(func, x_low: float, x_up: float, tolerance: float = 0.0001, loc='max') -> float:
    """
    Implement the golden section search algorithm
    """
    
    # calculate the distance
    # multiplied by the golden
    # ratio
    dist = (x_up-x_low)*GOLDEN_RATIO
    
    # remove this calculated value
    # from the upper value
    # and add to the lower value
    x_2 = x_up - dist
    x_1 = x_low + dist
    
    while abs(x_up - x_low) > tolerance:
        if func(x_1) > func(x_2):
            x_low = x_2
        else:
            x_up = x_1
            
        # calculate the distance
        # multiplied by the golden
        # ratio again
        dist = (x_up - x_low)*GOLDEN_RATIO
        
        # recalculate the bounds
        x_2 = x_up - dist
        x_1 = x_low + dist
        
    # return average between 
    # the two bounds
    return (x_up + x_low) / 2

if __name__ == '__main__':
    maximum = gss(f, 7, 9)
    print(f"Maximum found: {round(maximum, 2)}")
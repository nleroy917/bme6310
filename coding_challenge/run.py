import math

def calc_time_to_confluency(n_cells, d_cell, rate, d_well):
    """
    Calculate the time needed for cells to reach
    confluency inside a 96-well plate given an
    initial cell count, the diameter of a cell (microns),
    the growth rate (hours), and the diameter of a well
    
    :param: n_cells - int - starting cell count
    :param: d_cell - float - diameter of a cell
    :param: rate - int - growth rate in hours
    :param: d_well - float - diameter of the well in mm
    
    returns the time to confluency in minutes
    """
    
    # convert d_well to microns
    d_well = d_well*1000
    
    # calculate the time in hours
    time = rate*(math.log10((1/n_cells)*(d_well/d_cell))**2)/(math.log10(2))
    
    # return time in mintues
    return time * 60

# initial calc
time_1 = calc_time_to_confluency(1, 20, 24, 5)

# start with more cells and make them bigger
time_2 = calc_time_to_confluency(10, 40, 24, 5)

# print results
print(f"{round(time_1, 2)} minutes")
print(f"{round(time_2, 2)} minutes")

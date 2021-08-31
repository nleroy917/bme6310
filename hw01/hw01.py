"""
Author: Nathan LeRoy
Date: 08/30/2021
"""
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math

def new_matrices(n: int) -> list[npt.ArrayLike, npt.ArrayLike]:
    # create ones array
    A = np.ones(shape=(n,n), dtype="int")
    
    # create equivalent zeros array
    A_zeros = np.zeros(shape=(n,n), dtype="int")
    
    # block arrays rogether
    A_new = np.block(
        [
            [A, A_zeros], 
            [A_zeros, A]
        ]
    )
    
    return [A, A_new]

def f_x(x: float) -> float:
    """
    f(x) = 3xcos^2(x) – 2x 
    """
    return 3*x*(math.cos(x)**2) - 2*x

def f_prime_x(x: float) -> float:
    """
    f'(x) = 3cos^2(x) - 6xsin(x)cos(x) - 2
    """
    return 3*(math.cos(x)**2) - 6*x*math.sin(x)*math.cos(x) - 2

def aoritc_valve_area(pg: float, Q: float) -> float:
    """
    Function to estimate the area of the aortic valve.
    
    Av = Q/sqrt(PG), where
    
    Q is the cardiac output and PG is
    the difference between the left 
    ventricular systolic pressure and 
    the aortic systolic pressure.
    """
    return Q/math.sqrt(pg)

def _read_fasta_file(filename: str) -> str:
    """
    Will read in fasta file and convert it to one long string and return
    said string
    """
    with open(filename, 'r') as f:
        seq = ""
        lines = f.readlines()
        for line in lines:
            seq += line.rstrip()
    
    return seq

def cpg_island_detect(filename: str, window: int = 200, verbose: bool = False) -> list[dict]:
    """
    CpG island detector. Will take an input .fasta file and
    search the sequence space for CpG islands using the method 
    described by Gardiner-Garden and Frommer (1987).
    
    Searches the sequence in windows of 200 bp moving forward in 1bp
    increments. CpG islands are defined as sequence ranges where the 
    Obs/Exp value is greater than 0.6 and the GC content is greater 
    than 50%.
    
    The expected number of CpG dimers in a window is calculated as the 
    number of 'C's in the window multiplied by the number of 'G's in 
    the window, divided by the window length. CpG islands are often 
    found in the 5' regions of vertebrate genes, therefore this program 
    can be used to highlight potential genes in genomic sequences.
    
    :param: - filename (str) - path to .fasta file
    :param: - window (int) - optional. window length to search
    
    :return: - list of dictionaries that contain CpG island location 
               (start and end bp), GC content, and Obs/Exp value
    
    """
    cpg_islands: list[dict] = []
    
    # extract sequence from file
    seq = _read_fasta_file(filename)
    
    for i in range(len(seq)+1 - window):
        # get sequence window
        seq_window = seq[i:i+window]
        
        # calc parameters
        obs_exp = seq_window.count('cg')/(seq_window.count('c')* \
            seq_window.count('g'))*window
        gc_content = (seq_window.count('c') + seq_window.count('g'))/window
        
        # check parameters
        if obs_exp > 0.6 and gc_content > 0.5:
            # append new detected CpG island detection
            cpg_islands.append({
                'start': i+1,
                'end': i+window+1,
                'gc_content': gc_content,
                'obs_exp': obs_exp
            })
            
            # output results
            if verbose:
                print('-----> CpG Island Detected!')
                print(f"-----> \tStart: {i+1}")
                print(f"-----> \tEnd: {i+window+1}")
                print(f"-----> \tGC Content: {round(gc_content*100,2)}%")
                print(f"-----> \tObs/Exp Val: {round(obs_exp,2)}")
                print("")
            
    return cpg_islands if len(cpg_islands) > 0 else None
     

if __name__ == '__main__':
    ##
    # PROBLEM 1
    ##
    [A, A_new] = new_matrices(6)
    
    print(A)
    print(A_new)
    
    ##
    # PROBLEM 2
    ##
    points = 10000 #Number of points
    xmin = math.pi*-2
    xmax = math.pi*2
    
    # calculate
    xlist = np.linspace(xmin, xmax, num=points)
    ylist = [f_x(x) for x in xlist]
    y_prime_list = [f_prime_x(x) for x in xlist]
    
    # plot and annotate
    plt.plot(xlist, ylist, 'b')
    plt.plot(xlist, y_prime_list, 'r--')
    plt.title("f(x) and its derivative")
    plt.xlabel("Radians")
    plt.ylabel("f(x), f'(x)")
    plt.legend(['f(x)', "f'(x)"])
    plt.show()
    
    ##
    # PROBLEM 3
    ##
    points: int = 10000 #Number of points
    PGmin: float = 2
    PGmax: float = 60
    
    # calculate
    xlist = np.linspace(PGmin, PGmax, num=points)
    q_4 = [aoritc_valve_area(pg, 4) for pg in xlist]
    q_5 = [aoritc_valve_area(pg, 5) for pg in xlist]
    
    # plot and annotate
    plt.plot(xlist, q_4, 'b')
    plt.plot(xlist, q_5, 'r--')
    plt.xlabel("Ventrivular and Aoritc Systolic Pressure Difference (mmHg)")
    plt.ylabel("Estimated Aortic Valve Area (cm^2)")
    plt.legend(['Q=4 L/min', 'Q=5 L/min'])
    plt.show()
    
    ##
    # PROBLEM 4
    ##
    cpg_islands = cpg_island_detect('data/test_dna2.fasta')
    
    
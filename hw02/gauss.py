from helpers import _print_highlighted_matrix
import numpy as np
import numpy.typing as npt

def _verify_matrix(M: npt.ArrayLike) -> bool:
    n: int = M.shape[0]
    return all([M[i][i] != 0 for i in range(n)])
        
def _pivot_matrix(M: npt.ArrayLike, row: int, col: int) -> npt.ArrayLike:
    M[[row, col]] = M[[col, row]]
    return M

def gaussian_elim(A: npt.ArrayLike, b: npt.ArrayLike, verbose: bool=False) -> npt.ArrayLike:
    # recast arrays as floats for 
    # potential decimal calculations 
    # ahead
    A = A.astype('float64')
    b = b.astype('float64')
    
    # concatenate arrays to create
    # an augmented matrix
    aug_A = np.insert(A, len(A), b, axis=1)
    if verbose:
        print(aug_A)
        
    # extract num rows and num cols
    rows: int = aug_A.shape[0]
    cols: int = aug_A.shape[1]
    
    # check that our matrix is valid
    while not _verify_matrix(aug_A):
        # swap rows to achieve valid matrix
        for i in range(rows):
            if aug_A[i][i] == 0:
                print(f"0 pivot element found. Swapping rows {i} and {i-1}")
                aug_A = _pivot_matrix(aug_A, i, i-1)
            
    
    # for each column in the matrix (minus solution set)
    for i in range(cols-1):
        # iterate over each row
        # but only need to start at row below triangle
        # we are creating... i.e. full_col + 1
        for j in range(i+1,rows):
            
            # calculate the ratio used in
            # the row matrix calculations
            ratio = aug_A[j][i]/aug_A[i][i]
            
            # apply calculations to each
            # col in the matrix in row j
            for k in range(cols):
                aug_A[j][k] += -1 * ratio * aug_A[i][k]
    
    # if prompted, print
    # the row reduced
    # augmented matrix
    if verbose:
        print(aug_A)
        
    # with a row-reduced matrix, we can back propegate
    # to get the solution set
    # init new array to store solutions
    sol = np.zeros(rows)
    
    # starting from the back run a
    # backsweep to calculate the solutions
    #
    # formula pulled from Numerical Recipes
    # in C (Press, et, al,  1992)
    for i in range(rows-1, -1, -1):
        # init backsweep sum
        b_sum = 0
        for j in range(i+1, rows):
            b_sum += aug_A[i][j]*sol[j]
        sol[i] = (1 / aug_A[i][i]) * (aug_A[i][-1] - b_sum)
    
    return sol
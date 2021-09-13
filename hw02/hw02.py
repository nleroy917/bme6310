import numpy as np
import numpy.typing as npt

# import algorithm
from gauss import gaussian_elim

if __name__ == '__main__':
    
    ##
    # PROBLEM 1
    ##
    
    # test data
    test_A = np.array([
        [4, -2, 3],
        [1, 3, -4],
        [3, 1, 2],
    ])

    test_b = np.array([1, -7, 5])

    # perform gaussian elimination
    solution = gaussian_elim(test_A, test_b, verbose=False)
    
    ##
    # PROBLEM 4
    ##
    U = np.array([
        [0.40, -0.78, 0.47],
        [0.37, -0.33, -0.87],
        [-0.84, -0.52, -0.16]
    ], dtype="float64")
    
    S = np.array([
        [7.10, 0, 0],
        [0, 3.10, 0],
        [0, 0, 0]
    ], dtype="float64")
    
    Vh = np.array([
        [0.30, -0.51, -0.81],
        [0.76, 0.64, -0.12],
        [0.58, -0.58, 0.58]
    ], dtype="float64")
    
    # multiple SVD matrices to get
    # original A matrix back
    A = U @ S @ Vh
    
    # confirm rank of the A matrix
    rank_A = np.linalg.matrix_rank(A)
    
    null_A = gaussian_elim(A, np.zeros(len(A)), verbose=False)
    
    ##
    # PROBLEM 5
    ##
    A = np.array([
        [4, -2],
        [2, -1],
        [0, 0]
    ])
    
    # run svd using numpy
    [U, S, Vh] = np.linalg.svd(A, full_matrices=True)
    print(U)
    print(S)
    print(Vh)
from termcolor import colored
import numpy.typing as npt

def _print_highlighted_matrix(M: npt.ArrayLike, row: int, col: int):
    # extract num rows and num cols
    rows = M.shape[0]
    cols = M.shape[1]
    print("[")
    for r in range(rows):
        print("[", end="")
        for c in range(cols):
            if r == row and c == col:
                print(colored(str(M[r][c]), 'red', attrs=['bold']), end=" ")
            else:
                print(M[r][c], end=" ")
        print("]")
    print("]")
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import sys

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
    print("Problem #1 Test Function:")
    print(f"x = {solution}", end="\n\n")
    
    ##
    # PROBLEM 4
    ##
    k1 = 1
    k2 = 0.001
    k3 = 10
    k4 = 0.1
    A_0 = 10
    
    K = np.array([
        [-1*(k2+k4), k3],
        [k3, -1*k4]
    ])
    
    b = np.array([k1*A_0,0])
    
    conc = gaussian_elim(K, b)
    
    print("Problem #2 Test Function:")
    print("The steady-state concentrations were found to be:")
    print(f"conc = {conc}", end="\n\n")
    
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
    
    print("Problem #3 A calculation:")
    print(f"A = {A}")
    
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
    
    ##
    # PROBLEM 6
    ##
    
    # read in the data
    df = pd.read_csv('data.csv')

    # drop the diagnosis and id columns
    df = df.drop(columns=['id', 'diagnosis'])

    # scale the data in the datafram
    df_scaled = preprocessing.scale(df)

    # init a PCA object and fit the data
    pca = PCA(n_components=30)
    pca.fit(df_scaled)
    
    # trasnform our original matrix
    # and round explained variance to nearest 10 %
    pca_data = pca.transform(df_scaled)
    per_vars = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    
    labels = ['pc' + str(i+1) for i in range(len(per_vars))]
    
    # plot the PCA components 
    plt.bar(range(len(per_vars)), height=per_vars, tick_label=labels)
    plt.title('Explained variance in principle components')
    plt.xlabel('Component #')
    plt.ylabel("Percentage of Explained Variance")
    plt.show()
    
    # extract highest variance and plot
    df_pca = pd.DataFrame(df_scaled, columns=labels)
    plt.scatter(df_pca.pc1, df_pca.pc2)
    plt.title('Scree plot with PC1 and PC2')
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    plt.show()
    
    # organize by feature and PC
    df_pca = pd.DataFrame(pca.components_, columns=df.columns, index = labels)
    df_pca.to_csv('pca_out.csv')

    
    
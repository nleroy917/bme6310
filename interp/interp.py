import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

## image interpolation

head = plt.imread('head_frozen.jpeg')

fig, ax = plt.subplots(1,3, figsize=(12,8))
head_sparse = np.zeros((45, 35))

for i in range(45):
    for j in range(35):
        head_sparse[i,j] = head[i*10,j*10]


ax[0].imshow(head, cmap='gray')
ax[0].set_title('head frozen')

ax[1].imshow(head_sparse, cmap='gray')
ax[1].set_title('head sparse')

# Interpolate
Xi, Yi = np.arange(0, 350, 1), np.arange(0, 450, 1)
X, Y = np.arange(0, 350, 10), np.arange(0, 450, 10)

print(X.shape, Y.shape, head_sparse.shape)
f = interpolate.interp2d(X, Y, head_sparse, kind='cubic')
head_sparse_i = f(Xi, Yi)

ax[2].imshow(head_sparse_i, cmap='gray')

ax[2].set_title('head sparse interpolated')

plt.show()

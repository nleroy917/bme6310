import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv

# Set up subplots
fig, ax = plt.subplots(1, 3, figsize=(12, 8))

# Read in image
I = plt.imread('david.png') # pout.png or cell.png
# or myster.jpeg --> rgb2gray
#I = plt.imread("mystery.jpeg")
I = np.dot(I[..., :3], [0.299, 0.587, 0.114])  # Converts rgb to grayscale

ax[0].imshow(I, cmap='gray')  # Plot image on first subplot

I2 = I[..., :]  # Make a copy of I

u, s, v = svd(I2, full_matrices=False)  # Perform SVD

# Show cumulative sum of singular values
ax[1].scatter(range(len(s)), np.cumsum(s) / np.sum(s), facecolors='none', edgecolor='blue')

print("Waiting for button press...")
plt.waitforbuttonpress()  # Wait for button press

# Loop through each singular value
s_modes = np.zeros((len(s), len(s)))
for i in range(len(s)):
    s_modes[i, i] = s[i]  # Get i-th mode
    image_modes = np.dot(u, np.dot(s_modes, v))  # Calculate resulting image
    ax[2].imshow(image_modes, cmap='gray')  # Plot image
    ax[2].set_title("mode # {}".format(i))  # Set plot title

    fig.canvas.draw()
    print("Waiting for button press...")
    try:
        plt.waitforbuttonpress()  # Wait for button press
    except KeyboardInterrupt:
        print('Goodbye.')
        sys.exit(0)

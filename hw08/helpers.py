import numpy as np
import matplotlib.pyplot as plt

def ft(a):
    """
    performs an FFT with the origin of the function
    in both domains in the center of the vector.
    """
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(a)))


def ftn(a):
    """
    performs an n-dimensional FFT with the origin of the function
    in both domains in the center.
    """

    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a)))

def ift(a):
    """
    performs an inverse FFT with the origin of the function
    in both domains in the center of the vector.
    """
    
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(a)))

def iftn(a):
    """
    performs an n-dimenstional inverse FFT with the origin of the function
    in both domains in the center of the vector.
    """
    
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(a)))    


def pc(data, xlabel=None, ylabel=None, title=None):
    """
    pc(data) plots the real and imaginary parts of data on a color display.
    """
    
    n = np.max(data.shape)
    t = np.arange(0,n)
    _, ax = plt.subplots(figsize=(12,8))
    ax.plot(t,np.real(data), label='Real Component')
    ax.plot(t,np.imag(data), label='Imaginary Component')
    ax.legend(fontsize=16)
    
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=18)
    
    plt.show()
    
    
import numpy as np
from scipy.signal import unit_impulse
import matplotlib.pyplot as plt

def fftc(a):
    """
    performs an FFT with the origin of the function
    in both domains in the center of the vector.
    """
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(a)))


def fft2c(a):
    """
    performs an n-dimensional FFT with the origin of the function
    in both domains in the center.
    """

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))

def ifftc(a):
    """
    performs an inverse FFT with the origin of the function
    in both domains in the center of the vector.
    """
    
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(a)))

def ifft2c(a):
    """
    performs an n-dimenstional inverse FFT with the origin of the function
    in both domains in the center of the vector.
    """
    
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(a))) 

def plot_filt(dfilt: np.ndarray, fs: np.number):
    """
    dfilt - ndarray - output  of scipy.filters.firwin
    fs - number - sample rate

    Visualize a filter designed with the scipy.filter.firwin
    """
    
    dShow = np.convolve(dfilt, unit_impulse(len(dfilt)))
    freqs = np.linspace(0, fs/2, int(np.floor(len(dShow)/2)))
    H = np.fft.fft(dShow)
    H = H[:len(freqs)]
    
    _, ax = plt.subplots(figsize=(6,4))
    ax.plot(freqs, 10*np.log(np.abs(H)), 'b-', linewidth=1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Magnitude Response (dB)")
    
    plt.show()   

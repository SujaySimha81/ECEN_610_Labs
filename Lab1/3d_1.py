import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import blackman
from scipy.fft import fft, fftfreq
F = 2e6  # frequency of x(t)
Fs = 5e6  # Sampling frequency
Ts = 1 / Fs 
N = 64 #DFT points

t = np.linspace(0, 10/F, 10000)  
x1_t = np.cos(2*np.pi*F*t)  
x_t = x1_t * blackman(len(t))
n = np.arange(0, 10/F, Ts)  
x_n = np.cos(2*np.pi*F*n) *  blackman(len(n))
N_range = np.arange(0, N)
X = fft(x_n, N)
freq = fftfreq(N, d=1.0/Fs)

# Plots
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(t*1e9, x_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("x(t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n, x_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid()

plt.figure(figsize=(10, 5))
plt.stem(freq, abs(X), linefmt='r', markerfmt='ro')
plt.xlabel("F (MHz)")
plt.ylabel("| X[f] |")
plt.grid()
plt.tight_layout()
plt.show()


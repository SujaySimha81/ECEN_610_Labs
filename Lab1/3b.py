import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
F1 = 200e6  
F2 = 400e6  
Fs = 500e6  # Sampling frequency
Ts = 1 / Fs 
N = 64 #DFT points

t1 = np.linspace(0, 10/F1, 10000)  
t2 = np.linspace(0, 10/F2, 10000) 

y_t = np.cos(2*np.pi*F1*t2) + np.cos(2*np.pi*F2*t2)
n = np.arange(0, 10/F1, Ts)  
y_n = np.cos(2*np.pi*F1*n) + np.cos(2*np.pi*F2*n)  # Sampled signal
N_range = np.arange(0, N)
Y = fft(y_n, N)
freq = fftfreq(N, d=1.0/Fs)

# Plots
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(t2*1e9, y_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("y(t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n, y_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid()

plt.figure(figsize=(10, 5))
plt.stem(freq, abs(Y), linefmt='r', markerfmt='ro')
plt.xlabel("F (MHz)")
plt.ylabel("| Y[f] |")
plt.grid()
plt.tight_layout()
plt.show()


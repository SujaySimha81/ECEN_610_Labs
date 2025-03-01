import numpy as np
import matplotlib.pyplot as plt

F  = 2e6  # frequency of x(t)
Fs = 5e6  # Sampling frequency
Ts = 1 / Fs 

t = np.linspace(0, 10/F, 100000)  
x_t = 1*np.sin(2*np.pi*F*t)  
n = np.arange(0, 10/F, Ts)  
x_n = 1*np.sin(2*np.pi*F*n)  # Sampled signal

# Plots
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t * 1e9, x_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("x(t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n, x_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid()
plt.tight_layout()
plt.show()


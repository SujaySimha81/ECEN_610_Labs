import numpy as np
import matplotlib.pyplot as plt

F1 = 300e6  # frequency of x1(t)
F2 = 800e6  # frequency of x2(t)
Fs = 10e9  # Sampling frequency
Ts = 1 / Fs 

t1 = np.linspace(0, 10/F1, 10000)  
t2 = np.linspace(0, 10/F2, 10000)  
x1_t = np.cos(2*np.pi*F1*t1)  
x2_t = np.cos(2*np.pi*F2*t2)  
n1 = np.arange(0, 10/F1, Ts)  
n2 = np.arange(0, 10/F2, Ts)  
x1_n = np.cos(2*np.pi*F1*n1)  # Sampled signal
x2_n = np.cos(2*np.pi*F2*n2)  # Sampled signal

# Plots
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t1 * 1e9, x1_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("x1(t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n1, x1_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("x1[n]")
plt.grid()
plt.tight_layout()
plt.show()

# Plots
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t2 * 1e9, x2_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("x2(t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n2, x2_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("x2[n]")
plt.grid()
plt.tight_layout()
plt.show()


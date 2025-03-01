import numpy as np
import matplotlib.pyplot as plt

F1 = 300e6  
Fs = 500e6  
Ts = 1/Fs

t = np.linspace(0, 10/F1, 1000)
t1 = np.arange(0, 10/F1 - Ts, Ts)     
t2 = np.arange(Ts/2, 10/F1 - Ts, Ts)  

x = np.cos(2*np.pi*F1*t)  
x1_n = np.cos(2*np.pi*F1*t1)  
x2_n = np.cos(2*np.pi*F1*t2)  

#function to reconstrct the signal
def reconstruct_x(t, x, nTs, Ts):
    xr = np.zeros_like(t)
    for n in range(0, len(nTs)):
        xr += x[n] * np.sinc((t - nTs[n])/Ts)
    return xr

# Reconstruct signals from samples
x_r1 = reconstruct_x(t, x1_n, t1, Ts)
x_r2 = reconstruct_x(t, x2_n, t2, Ts)

mse1 = np.mean((x_r1 - x)**2)
mse2 = np.mean((x_r2 - x)**2)

print("MSE for case 1 (unshifted sampling) = ", mse1)
print("MSE for case 2 (shifted sampling)   = ", mse2)

# Plots
plt.figure(figsize=(10, 5))

plt.subplot(3, 1, 1)
plt.plot(t*1e9, x, color='b')
plt.xlabel("Time (ns)")
plt.ylabel("x(t)")
plt.subplot(3, 1, 2)
plt.stem(t1, x1_n, linefmt='r', markerfmt='ro')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.subplot(3, 1, 3)
plt.plot(t*1e9, x_r1, color='r')
plt.xlabel("Time (ns)")
plt.ylabel("Reconstructed x(t)")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(t*1e9, x, color='b')
plt.xlabel("Time (ns)")
plt.ylabel("x(t)")
plt.subplot(3, 1, 2)
plt.stem(t2, x2_n, linefmt='r', markerfmt='ro')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.subplot(3, 1, 3)
plt.plot(t*1e9, x_r2, color='r')
plt.xlabel("Time (ns)")
plt.ylabel("Reconstructed x(t)")
plt.grid()
plt.tight_layout()
plt.show()




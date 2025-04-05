import numpy as np
import matplotlib.pyplot as plt
import math

f  = 1e9  # frequency of Vin(t)
T  = 1/f
Fs = 10e9  # Sampling frequency
Ts = 1 / Fs 
time_constant = 10e-12 # RC Time constant
C = 2 # number of cycles for plot

Tph = Ts/2 #tracking phase time

t = np.linspace(0, C/f, 10000)  
Vin_t = np.sin(2*np.pi*f*t)

n =  np.arange(0, C/f, Ts)
Vin_n = np.sin(2*np.pi*f*n)

Vsw_t = np.zeros(len(t))
Vout_t = np.zeros(len(t))

hold_val = 0
sw_count = 1
num_sample_points_per_cycle = np.ceil(len(t)/(C*T/Ts))

for i in range(len(t)):
    t1 = t[i]
    if(i <= sw_count*num_sample_points_per_cycle/2): #tracking
        Vsw_t[i] = 1
        Vout_t[i] = Vin_t[i] + (hold_val - Vin_t[i])*(math.exp(-(t1-((sw_count-1)/2)*Ts)/time_constant))
        if(i == sw_count*num_sample_points_per_cycle/2):
            hold_val = Vout_t[i]
    elif (i == (sw_count+1)*num_sample_points_per_cycle/2): 
        Vsw_t[i] = 0
        sw_count = sw_count + 2
        Vout_t[i] = hold_val
    else: #hold
        Vsw_t[i] = 0
        Vout_t[i] = hold_val

# Plots
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t * 1e9, Vin_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("Vin(t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n, Vin_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Sampled Vin : Vin[n]")
plt.grid()
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t * 1e9, Vsw_t,  color='r')
plt.xlabel("Time (ns)")
plt.ylabel("NMOS switch : Vsw(t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t * 1e9, Vout_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("Vout(t)")
plt.grid()
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import math

def quantize(signal, num_bits):
    num_levels = 2**num_bits
    quant_step = 1/num_levels # if the range is [-0.5,0.5]
    signal_q = np.round(signal/quant_step)*quant_step
    return signal_q

f1  = 0.2e9
f2  = 0.58e9
f3  = 1e9
f4  = 1.7e9
f5  = 2.4e9

f = np.gcd(np.gcd(int(f1),int(f2)),np.gcd(int(f3),int(f4)))
T  = 1/f
Fs = 10e9  # Sampling frequency
Ts = 1 / Fs 
time_constant = 12e-12 # RC Time constant
C = 0.1 # number of cycles for plot
#C = 2 # number of cycles for plot

N = 7 #ADC quantizer bits
M_range = np.arange(2, 11) # FIR filter taps

Tph = Ts/2 #tracking phase time

t = np.linspace(0, C/f, 10000)  

V1_t = 0.125*np.sin(2*np.pi*f1*t)
V2_t = 0.125*np.sin(2*np.pi*f2*t)
V3_t = 0.125*np.sin(2*np.pi*f3*t)
V4_t = 0.125*np.sin(2*np.pi*f4*t)
V5_t = 0.125*np.sin(2*np.pi*f5*t)

Vin_t = V1_t + V2_t + V3_t + V4_t + V5_t

n =  np.arange(Ts/2, C/f, Ts)

V1_n = 0.125*np.sin(2*np.pi*f1*n)
V2_n = 0.125*np.sin(2*np.pi*f2*n)
V3_n = 0.125*np.sin(2*np.pi*f3*n)
V4_n = 0.125*np.sin(2*np.pi*f4*n)
V5_n = 0.125*np.sin(2*np.pi*f5*n)

Vin_n = V1_n + V2_n + V3_n + V4_n + V5_n # ideally sampled Vin signal 



Vsw_t = np.zeros(len(t))
Vout_t = np.zeros(len(t))
Vout_n = np.zeros(len(n))

hold_val = 0
sw_count = 1
num_sample_points_per_cycle = np.ceil(len(t)/(C*T/Ts))
idx = 0

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
        Vout_n[idx] = hold_val
        idx = idx + 1
    else: #hold
        Vsw_t[i] = 0
        Vout_t[i] = hold_val
Vout_n[idx] = hold_val

Vout_q_n = quantize(Vout_n, N) # ADC with sampling from the circuit
Vin_q_n  = quantize(Vin_n, N)  # ADC with ideal sampling

Var_E = np.zeros(len(M_range))

k=0
#estimate error = Vout_q_n - Vin_n
for M in M_range:
    Vout_previous_values = np.zeros((len(n) - (M), M))
    for i in range(M):
        Vout_previous_values[:, i] = Vout_q_n[M - i -1 : len(n)-i-1]

    #target_signal = Vin_q_n[M-1:]
    d = Vout_q_n[M : len(n)] - Vout_n[M : len(n)]
    #to get FIR filter coefficients using Least squares estimnation
    h_filter = np.linalg.pinv(Vout_previous_values.T @ Vout_previous_values) @ Vout_previous_values.T @ d
    
    E_estimate = Vout_previous_values @ h_filter
    Vout_q_new_n = Vout_q_n[M:] + E_estimate
    
    E = Vout_q_new_n - Vout_n[M:]
    Var_E[k] = np.var(E)
    k = k+1

#calculated previously
Var_uniform_q_noise = 5.086e-6

Var_ratio = Var_E[::-1]/Var_uniform_q_noise

# Plots
plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(t * 1e9, Vin_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("Vin(t)")
plt.grid()
plt.subplot(3, 1, 2)
plt.stem(n, Vin_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Sampled Vin : Vin[n]")
plt.grid()
plt.subplot(3, 1, 3)
plt.stem(n, Vin_q_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Quantized from ADC Vin[n]")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(t * 1e9, Vsw_t,  color='r')
plt.xlabel("Time (ns)")
plt.ylabel("NMOS switch : Vsw(t)")
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t * 1e9, Vout_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("Vout(t)")
plt.grid()
plt.subplot(3, 1, 3)
plt.stem(n, Vout_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Sampled from circuit Vout[n]")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.stem(n, Vout_q_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Quantized from ADC Vout[n]")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(M_range, Var_ratio,  color='r', marker='o')
plt.xlabel("M - number of FIR Taps")
plt.ylabel("Ratio of Variances of Error")
plt.grid()
plt.tight_layout()
plt.show()

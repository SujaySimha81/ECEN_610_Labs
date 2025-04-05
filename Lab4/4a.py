import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram as psd

num_bits =10
bitstream = np.random.randint(0, 2, num_bits)
nrz_signal = 0.5*bitstream

f = 1e9
T = 1/f
Fs = 10e9  # Sampling frequency
Ts = 1 / Fs 
time_constant = 12e-12 # RC Time constant
C = 10 # number of cycles for plot

Tph = Ts/2 #tracking phase time

n =  np.arange(0, C/f, Ts)

t = np.linspace(0, C/f, 10000)  


Vin_t = np.random.uniform(-0.5, 0.5, size=len(t))
Vin_n = np.zeros(len(n))
for i in range(len(n)):
    Vin_n[i] = Vin_t[int(i*Ts)]

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

time_mismatch   = 1e-12 #1ps
offset_mismatch = 0.05 
bw_mismatch     = 0.9

Fs_2 = bw_mismatch*Fs
Ts = 1/Fs

Vin_2_t = np.roll(Vin_t, 1) + offset_mismatch
Vin_2_t[:1] = 0
for i in range(len(n)):
    Vin_2_n = Vin_2_t[int(i*Ts)]


n_2 = np.arange(0, C/f, Ts)
Vout_2_t = np.zeros(len(t))
Vout_2_n = np.zeros(len(n_2))
hold_val = 0
sw_count = 1
idx = 0
for i in range(len(t)):
    t1 = t[i]
    if(i <= sw_count*num_sample_points_per_cycle/2): #tracking
        Vsw_t[i] = 1
        Vout_2_t[i] = Vin_2_t[i] + (hold_val - Vin_2_t[i])*(math.exp(-(t1-((sw_count-1)/2)*Ts)/time_constant))
        if(i == sw_count*num_sample_points_per_cycle/2):
            hold_val = Vout_2_t[i]
    elif (i == (sw_count+1)*num_sample_points_per_cycle/2): 
        Vsw_t[i] = 0
        sw_count = sw_count + 2
        Vout_2_t[i] = hold_val
        Vout_2_n[idx] = hold_val
        idx = idx + 1
    else: #hold
        Vsw_t[i] = 0
        Vout_2_t[i] = hold_val
Vout_2_n[idx] = hold_val

Vout_2way = np.zeros(len(n))
Vout_2way[0::2] = Vout_n[0::2]  
Vout_2way[1::2] = Vout_2_n[1::2]

#to find and plot PSD
freq, pxx = psd(Vout_2way, Fs, nfft=2048)
signal_power = np.mean(pow(Vout_2way, 2))
print("Signal power =", signal_power)
signal_power_psd = pxx[np.argmax(pxx)]
noise_power = np.sum(pxx) - signal_power_psd
print("Noise power =", noise_power)
SNR_dB = 10*np.log10(signal_power/noise_power)
print("SNR in dB =", SNR_dB)

plt.subplot(2, 1, 1)
plt.plot(t * 1e9, Vin_t,  color='b')
plt.xlabel("Time (ns)")
plt.ylabel("Vin(t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n, Vout_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Sampled Vout : Vout[n]")
plt.grid()
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.stem(n, Vout_2_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Sampled Vout with channel mismatches: Vout[n]")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n, Vout_2way, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Resultant Vout: Vout[n]")
plt.grid()
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(freq, pxx, color='r')
plt.xlabel("F (MHz)")
plt.ylabel("psd")
plt.grid()
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import psd
from scipy.fft import fft, fftfreq

def quantize(signal, num_bits):
    num_levels = 2**num_bits
    quant_step = 2/num_levels # if teh range is [-1,1]
    signal_q = np.round(signal/quant_step)*quant_step
    return signal_q

F  = 200e6  # frequency of x(t)
Fs = 400e6  # Sampling frequency
Ts = 1 / Fs 

t = np.linspace(0, 30/F, 100000)  
x_t = 1*np.sin(2*np.pi*F*t)  
n = np.arange(Ts/3, 30/F+Ts/3, Ts)  
N = len(n)
#n = np.arange(0, 30/F, Ts)  
x_n = 1*np.sin(2*np.pi*F*n)  # Sampled signal

#Quantization :
x_q_n = quantize(x_n, 12)

noise = np.abs(x_n - x_q_n)
#to find and plot PSD of signal
pxx, f = psd(x_q_n, N, Fs, pad_to=N, c='r')
signal_power_psd = np.sum(pxx[len(pxx)-12:])
noise_power_psd = np.sum(pxx) - signal_power_psd
print("Signal power =", signal_power_psd)
print("Noise power  =", noise_power_psd)
SNR_dB_psd = 10*np.log10(signal_power_psd/noise_power_psd)
print("SNR in dB from PSD =", SNR_dB_psd)

#
# Plots
plt.figure(figsize=(9, 5))
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


plt.figure(figsize=(9, 5))
plt.subplot(2, 1, 1)
plt.stem(n, x_q_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Quantized x[n]")
plt.grid()


plt.subplot(2, 1, 2)
plt.stem(n, noise, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("noise[n]")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.plot(f, pxx, color='r')
plt.xlabel("F (MHz)")
plt.ylabel("psd")
plt.grid()
plt.tight_layout()
plt.show()


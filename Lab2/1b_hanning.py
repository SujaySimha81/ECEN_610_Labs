import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram as psd
from scipy.signal import windows as window

F = 2e6  # frequency of x(t)
Fs = 5e6  # Sampling frequency
Ts = 1 / Fs 
N = 256 #DFT points
A = 1 # amplitude of x
SNR_dB = 50 # snr of resultant signal in dB 

t = np.linspace(0, 10/F, 1000)  
x_t = A*np.sin(2*np.pi*F*t)  
n = np.arange(0, 500/F, Ts)  
n1 = np.arange(0, len(n))
x_n = A*np.sin(2*np.pi*F*n)  # Sampled signal
#N_range = np.arange(0, N)
#X = fft(x_n, N)
#freq = fftfreq(N, d=1.0/Fs)

signal_power = pow(A,2)/2
noise_power = signal_power/pow(10, SNR_dB/10)
noise_n = np.random.normal(0, np.sqrt(noise_power), len(n))# mean = 0, variance = power
print("Required Noise power calculated = ", noise_power)
x_noise_n = x_n + noise_n # noisy signal

hanning_w_n = window.hann(len(n))
x_noise_w_n = x_noise_n*hanning_w_n

#to find and plot PSD
f, pxx = psd(x_noise_w_n, Fs)

#signal_power : max psd at f~F
Fmargin = 8e3
signal_power_psd = np.sum(pxx[(f>=F-Fmargin) & (f<=F+Fmargin)])
print("Signal power =", signal_power_psd)
#noise_power : everywhere esle
noise_power_psd = np.sum(pxx[(f<F-Fmargin) | (f>F+Fmargin)])
print("Noise power =", noise_power_psd)
SNR_dB_psd = 10*np.log10(signal_power_psd/noise_power_psd)
print("SNR in dB from PSD =", SNR_dB_psd)


# Plots
plt.figure(figsize=(7, 5))
plt.stem(n1, x_noise_w_n, linefmt='r', markerfmt='ro')
plt.xlabel("n")
plt.ylabel("Windowed Signal with Noise x_noise_w_n[n]")
plt.grid()

plt.figure(figsize=(7, 5))
plt.plot(f, pxx, color='r')
plt.xlabel("F (MHz)")
plt.ylabel("psd")
plt.grid()
plt.tight_layout()
plt.show()
#

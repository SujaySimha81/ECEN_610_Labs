import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram as psd


fs      = 500e6         # Sampling frequency
Ts      = 1/fs
N       = 8192          # Number of samples = FFT points)
nfft    = N           
C       = 3271          # num cycles
fin     = fs * C / N    # fin : incommensurate ~200MHz
Vfs     = 1.0           # Full-scale
b_stage = 2.5           # bits per  stage
k       = 6             # num pipeline stages
gain    = 2**b_stage    # Residue amplifier gain per stage

n   = np.arange(0,N)
vin = 0.5 * np.sin(2 * np.pi * fin * n/fs)
t = np.linspace(0, 10/fin, N*10) #timedomain
vin_t = 0.5 * np.sin(2 * np.pi * fin * t)
print("*******************************************")
print("fin = ", fin)
print("fs = ", fs)
print("*******************************************")

def mdac_stage(Vin):
    step      = Vfs / (2**b_stage)
    code     = np.round(Vin / step)
    max_code = np.floor((2**b_stage - 1) / 2)
    min_code = -max_code
    code     = np.clip(code, min_code, max_code)
    Vq       = code * step
    Vres     = (Vin - Vq) * gain
    return code.astype(int), Vres

def pipeline_adc(v_signal):
    codes = []
    v     = v_signal.copy()
    for _ in range(k):
        c, v = mdac_stage(v)
        codes.append(c)
    codes      = np.vstack(codes)
    weights    = [2**((k - i - 1) * b_stage) for i in range(k)]
    digital15  = np.sum(codes * np.array(weights)[:, None], axis=0)
    return digital15.astype(int)

def plot_sndr(voltage, fs, nfft):
    # Remove DC offset
    v = voltage - np.mean(voltage)
    w = np.hanning(len(v))
    #v = v*w
    # Compute PSD via periodogram 
    f, Pxx = psd(v, fs=fs, nfft=nfft, scaling='spectrum')
    # Zero out DC bin
    Pxx[0] = 0
    psd_db = 10 * np.log10(Pxx + 1e-20)

    # Compute SNDR
    sig_bin  = np.argmax(Pxx)
    Psignal  = Pxx[sig_bin]
    print("*******************************************")
    print("Signal Power = ",Psignal) 
    Pnoise   = np.sum(Pxx) - Psignal
    print("Noise Power = ",Psignal)
    SNDR     = 10 * np.log10(Psignal / Pnoise)
    print(f"SNDR in dB = {SNDR:.2f} dB")
    print("*******************************************")

    plt.figure()
    plt.plot(f/1e6, psd_db)
    plt.title("PSD of Pipeline ADC Output")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB)")
    plt.grid(True)

code15      = pipeline_adc(vin) # get teh codes from pipleined ADC(6*2.5=15)
code13      = code15 >> 2          # truncate 15bits to 13
lsb_tot     = Vfs/(2**13)          
voltage_out = code13 * lsb_tot

n10 = int(10 * fs / fin)
plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(t*1e9 , vin_t)
plt.xlabel("time (ns)")
plt.ylabel("Vin(t) (V)")
plt.grid()
plt.subplot(3, 1, 2)
plt.stem(n[:n10], vin[:n10])
plt.xlabel("n")
plt.ylabel("Ideal Samples Vin[n] (V)")
plt.grid()
plt.subplot(3, 1, 3)
plt.stem(n[:n10], voltage_out[:n10]) 
plt.xlabel("n")
plt.ylabel("Vout[n] (V)")
plt.grid(True)


plot_sndr(voltage_out, fs, nfft)

plt.show()

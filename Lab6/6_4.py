import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram as psd
import random

fs      = 500e6         # Sampling frequency
Ts      = 1/fs
N       = 8192          # Number of samples = FFT points
nfft    = N
C       = 3271          # Number of input cycles
fin     = fs * C / N    # ≃200 MHz
Vfs     = 1.0           # Full-scale (±0.5 V)
Vref    = Vfs / 2       # Reference for cap mismatch term
b_stage = 2.5           # Bits per MDAC stage
k       = 6             # Pipeline stages
gain    = 2**b_stage    # Ideal residue-amplifier gain per stage

ENABLE_OTA_GAIN_ERROR    = 0
ENABLE_OTA_OFFSET_ERROR  = 0
ENABLE_CAP_MISMATCH      = 0
ENABLE_COMP_OFFSET       = 0
ENABLE_OPAMP_NONLIN      = 1
ENABLE_OPAMP_BW_LIMIT    = 0

OTA_OPEN_LOOP_GAIN  = 1e-4    # Finite OTA open-loop gain (Aol)
OTA_OFFSET_V        = 1.1    # OTA offset voltage (V)
CAP_MISMATCH        = [0.45, -0.3, 0.7, -0.1, -0.9, 0.99]   # capacitor mismatch 
COMP_OFFSET         = [-0.075, 0.034, -0.5, 0.12, 0.45, 0.6] # comparator offset 
OPAMP_NONLIN_VTH    = 0.0007 # Voltage threshold for non-linear Aol roll-off
OPAMP_GBW           = 50.75e6

n     = np.arange(N)
vin   = 0.5 * np.sin(2 * np.pi * fin * n / fs)
t     = np.linspace(0, 10/fin, N*10)
vin_t = 0.5 * np.sin(2 * np.pi * fin * t)

print("*******************************************")
print("fin = ", fin)
print("fs = ", fs)
print("*******************************************")

print("*******************************************")
print("Errors added : ")
if ENABLE_OTA_GAIN_ERROR:
    print("Finite OTA Gain Aol =", OTA_OPEN_LOOP_GAIN)
if ENABLE_OTA_OFFSET_ERROR:
    print("OTA Offset =", OTA_OFFSET_V)
if ENABLE_CAP_MISMATCH:
    print("Capacitor Mismatch =", CAP_MISMATCH)
if ENABLE_COMP_OFFSET:
    print("Comparator Offset =", COMP_OFFSET)
if ENABLE_OPAMP_NONLIN:
    print("Non-linear OTA Aol threshold Vth =", OPAMP_NONLIN_VTH)
if ENABLE_OPAMP_BW_LIMIT:
    print("OPAMP_GBW =", OPAMP_GBW)
print("*******************************************")

def mdac_stage(Vin):
    base_step = Vfs / (2**b_stage)
    step = base_step + (COMP_OFFSET[random.randint(0, k-1)] if ENABLE_COMP_OFFSET else 0)

    code     = np.round(Vin / step)
    max_code = np.floor((2**b_stage - 1) / 2)
    min_code = -max_code
    code     = np.clip(code, min_code, max_code).astype(int)

    Vq = code * step
    if ENABLE_CAP_MISMATCH:
        Vq += CAP_MISMATCH[random.randint(0, k-1)] * Vref

    diff = Vin - Vq
    Gcl = gain
    if ENABLE_OTA_GAIN_ERROR:
        Gcl = gain * (OTA_OPEN_LOOP_GAIN / (OTA_OPEN_LOOP_GAIN + gain))
    if ENABLE_OPAMP_NONLIN:
        Aol_eff = OTA_OPEN_LOOP_GAIN / (1 + (diff/OPAMP_NONLIN_VTH)**2)
        Gcl = gain * (Aol_eff / (Aol_eff + gain))

    Vres = diff * Gcl
    if ENABLE_OTA_OFFSET_ERROR:
        Vres += OTA_OFFSET_V

    if ENABLE_OPAMP_BW_LIMIT:
        f3 = OPAMP_GBW / Gcl
        tau = 1/(2*np.pi*f3)
        Vres = Vres * (1 - np.exp(-Ts/tau))

    return code, Vres

def pipeline_adc(v_signal):
    v = v_signal.copy()
    codes = []
    for _ in range(k):
        c, v = mdac_stage(v)
        codes.append(c)
    codes      = np.vstack(codes)
    weights    = [2**((k - i - 1) * b_stage) for i in range(k)]
    digital15  = np.sum(codes * np.array(weights)[:, None], axis=0)
    return digital15.astype(int)

def plot_sndr(voltage, fs, nfft):
    v = voltage - np.mean(voltage)
    w = np.hanning(len(v))
    f, Pxx = psd(v, fs=fs, nfft=nfft, scaling='spectrum')
    Pxx[0] = 0
    psd_db = 10 * np.log10(Pxx + 1e-20)

    sig_bin = np.argmax(Pxx)
    Psignal = Pxx[sig_bin]
    Pnoise  = np.sum(Pxx) - Psignal
    SNDR    = 10 * np.log10(Psignal /  0.0030872302211804037)
    print("*******************************************")
    print("Signal Power =", Psignal)
    print("Noise Power =", Pnoise)
    print(f"SNDR in dB = {SNDR:.5f} dB")
    print("*******************************************")

    plt.figure()
    plt.plot(f/1e6, psd_db)
    plt.title("PSD of Pipeline ADC Output")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB)")
    plt.grid(True)

code15      = pipeline_adc(vin)  # 15-bit codes from 6×2.5b
code13      = code15 >> 2        # truncate to 13 bits
lsb_tot     = Vfs / (2**13)
voltage_out = code13 * lsb_tot   # map to ±0.5 V

n10 = int(10 * fs / fin)
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t*1e9, vin_t);           plt.ylabel("Vin(t)"); plt.grid(True)
plt.subplot(3,1,2)
plt.stem(n[:n10], vin[:n10], basefmt=" "); plt.ylabel("Vin[n]"); plt.grid(True)
plt.subplot(3,1,3)
plt.stem(n[:n10], voltage_out[:n10], basefmt=" "); plt.ylabel("Vout[n]"); plt.xlabel("n"); plt.grid(True)

plot_sndr(voltage_out, fs, nfft)


def pipeline_stage_codes(v_signal):
    v = v_signal.copy()
    codes = np.zeros((k, len(v_signal)), dtype=int)
    for stage in range(k):
        next_v = np.zeros_like(v)
        for i in range(len(v)):
            c, r = mdac_stage(v[i])
            codes[stage, i] = c
            next_v[i]      = r
        v = next_v
    return codes

stage_codes = pipeline_stage_codes(vin)  
X = stage_codes[:4].T                    
ideal13 = np.round((vin + 0.5)*(2**13 - 1)/Vfs - 2**12).astype(float)
# Ideal 16-bit reference code, zero-centered:
ideal_ref = np.round((vin + 0.5)*(2**16 - 1)/Vfs - 2**(16-1)).astype(float)
mu     = 10e-4
w      = np.zeros(4)
errors = np.zeros(N)
W_hist = np.zeros((N,4))

for n in range(N):
    x_n = X[n]
    y_n = w.dot(x_n)
    e_n = ideal_ref[n] - y_n
    w   += mu * e_n * x_n
    
    errors[n] = e_n
    W_hist[n] = w/N


mu_nlms = 0.2
eps = 1e-6
w_nlms = np.zeros(4)
err_nlms = np.zeros(N)
W_nlms = np.zeros((N,4))
for i in range(N):
    x = X[i]
    e = ideal_ref[i] - w_nlms.dot(x)
    norm2 = x.dot(x) + eps
    w_nlms += (mu_nlms / norm2) * e * x
    err_nlms[i] = e
    W_nlms[i] = w_nlms/N

y_cal = code13+X.dot(w)
v_cal = (y_cal + 2**12)*(Vfs/(2**13 - 1)) - 0.5
v_cal = (y_cal + 2**15)*(Vfs/(2**16 - 1)) - 0.5

y_cal_1 = code13+X.dot(w_nlms)
v_cal_1 = (y_cal_1 + 2**15)*(Vfs/(2**16 - 1)) - 0.5

print("*******************************************")
print("With LMS Calibration : ")
plot_sndr(v_cal_1, fs, nfft)
print("*******************************************")
print("With Normalized LMS Calibration : ")
plot_sndr(v_cal, fs, nfft)
print("*******************************************")


thr = np.std(errors)
def conv_iter(err):
    cnt=0
    for idx, v in enumerate(np.abs(err)):
        cnt = cnt+1 if v<thr else 0
        if cnt>=N//5:
            return idx-99
    return None

print(f"LMS  MSE converging in {conv_iter(errors)} iterations")
print(f"NLMS MSE converging in {conv_iter(err_nlms)} iterations")


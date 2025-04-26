import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram as psd

# === ADC & simulation parameters ===
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

# === Error switches (0=off, 1=on) ===
ENABLE_OTA_GAIN_ERROR    = 0
ENABLE_OTA_OFFSET_ERROR  = 0
ENABLE_CAP_MISMATCH      = 0
ENABLE_COMP_OFFSET       = 0
ENABLE_OPAMP_NONLIN      = 0
ENABLE_OPAMP_BW_LIMIT    = 1

OTA_OPEN_LOOP_GAIN  = 1e4    # Finite OTA open-loop gain (Aol)
OTA_OFFSET_V        = 1.1    # OTA offset voltage (V)
CAP_MISMATCH        = 0.45   # capacitor mismatch 
COMP_OFFSET         = -0.075  # comparator offset 
OPAMP_NONLIN_VTH    = 0.0007    # Voltage threshold for non-linear Aol roll-off
OPAMP_GBW           = 50.75e6

#vin
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
    step = base_step + (COMP_OFFSET if ENABLE_COMP_OFFSET else 0)
    code     = np.round(Vin / step)
    max_code = np.floor((2**b_stage - 1) / 2)
    min_code = -max_code
    code     = np.clip(code, min_code, max_code).astype(int)

    Vq = code * step
    if ENABLE_CAP_MISMATCH:
        Vq += CAP_MISMATCH * Vref

    diff = Vin - Vq
    # default ideal
    Gcl = gain
    # finite linear gain
    if ENABLE_OTA_GAIN_ERROR:
        Gcl = gain * (OTA_OPEN_LOOP_GAIN / (OTA_OPEN_LOOP_GAIN + gain))

    # non-linear open-loop gain?
    if ENABLE_OPAMP_NONLIN:
        Aol_eff = OTA_OPEN_LOOP_GAIN / (1 + (diff/OPAMP_NONLIN_VTH)**2)
        Gcl     = gain * (Aol_eff / (Aol_eff + gain))
        
    # OTA offset
    Vres = diff * Gcl
    if ENABLE_OTA_OFFSET_ERROR:
        Vres += OTA_OFFSET_V
    
    if ENABLE_OPAMP_BW_LIMIT:
        f3dB = OPAMP_GBW/Gcl
        tau  = 1/(2 * np.pi * f3dB)
        Vres = Vres*(1 - np.exp(-Ts/tau))

    return code, Vres

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
code13      = code15 >> 2   # get teh codes from pipleined ADC(6*2.5=15)
lsb_tot     = Vfs / (2**13)
voltage_out = code13 * lsb_tot

n10 = int(10 * fs / fin)
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t*1e9, vin_t); plt.ylabel("Vin(t)"); plt.grid(True)
plt.subplot(3,1,2)
plt.stem(n[:n10], vin[:n10], basefmt=" "); plt.ylabel("Vin[n]"); plt.grid(True)
plt.subplot(3,1,3)
plt.stem(n[:n10], voltage_out[:n10], basefmt=" "); plt.ylabel("Vout[n]"); plt.xlabel("n"); plt.grid(True)

plot_sndr(voltage_out, fs, nfft)
plt.tight_layout()
plt.show()
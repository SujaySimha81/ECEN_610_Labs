import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram as psd
import random

fs      = 500e6         # Sampling frequency
Ts      = 1/fs
N       = 8192          # (unused for DFT; kept for compatibility)
nfft    = N
C       = 3271          # for single‐tone, now unused
fin     = fs * C / N    # ≃200 MHz, unused
Vfs     = 1.0           # Full-scale (±0.5 V)
Vref    = Vfs / 2
b_stage = 2.5           # Bits per MDAC stage
k       = 6             # Pipeline stages
gain    = 2**b_stage

ENABLE_OTA_GAIN_ERROR   = 0
ENABLE_OTA_OFFSET_ERROR = 0
ENABLE_CAP_MISMATCH     = 1
ENABLE_COMP_OFFSET      = 0
ENABLE_OPAMP_NONLIN     = 0
ENABLE_OPAMP_BW_LIMIT   = 0

OTA_OPEN_LOOP_GAIN  = 1e-4    # Finite OTA open-loop gain (Aol)
OTA_OFFSET_V        = 1.1    # OTA offset voltage (V)
CAP_MISMATCH        = [0.45, -0.3, 0.7, -0.1, -0.9, 0.99]   # capacitor mismatch 
COMP_OFFSET         = [-0.075, 0.034, -0.5, 0.12, 0.45, 0.6] # comparator offset 
OPAMP_NONLIN_VTH    = 0.0007 # Voltage threshold for non-linear Aol roll-off
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
Ntones = 128
BW     = 200e6
tone_sp = BW / Ntones
Nfft   = int(fs / tone_sp)      # ensures tone bins align to DFT bin
n      = np.arange(Nfft)

#BPSK Modulation

bits = np.random.choice([-0.5, 0.5], size=Ntones)
vin = np.zeros(Nfft)
for i in range(Ntones):
    f_i = tone_sp * (i + 0.5)
    vin += bits[i] * np.cos(2*np.pi*f_i*n/fs)

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
        #Open loopgain order gain
        a2, a3, a4, a5 = 0.10, 0.20, 0.15, 0.10
        factor = (1 + a2*(diff/Vfs)**2 + a3*(diff/Vfs)**3 + a4*(diff/Vfs)**4 + a5*(diff/Vfs)**5)
        Aol_eff = OTA_OPEN_LOOP_GAIN * factor
        Gcl = gain * (Aol_eff / (Aol_eff + gain))

    Vres = diff * Gcl
    if ENABLE_OTA_OFFSET_ERROR:
        Vres += OTA_OFFSET_V


    return code, Vres

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
weights15   = 2**((np.arange(k)[::-1])*b_stage)
digital15   = weights15.dot(stage_codes)
#truncate to 13 bits
code13      = np.floor(digital15/4).astype(int)
lsb_tot     = Vfs/(2**13)
voltage_out = code13*lsb_tot

# DFT & detect before calibration
X_before  = np.fft.fft(voltage_out, Nfft)
est_before = np.sign(X_before[:Ntones].real)
MSE_before = np.mean((est_before - bits)**2)
BER_before = np.mean(est_before != bits)

print("*******************************************")
print("MSE before Calibration = ", MSE_before)
print("BER before Calibration = ", BER_before)
print("*******************************************")

X = stage_codes[:4].T.astype(float)   
# Ideal reference codes from perfect ADC (no mismatch)
ideal_ref = np.round((vin+0.5)*(2**13-1)/Vfs - 2**12).astype(float)

mu     = 10e-4
w      = np.zeros(4)
errors = np.zeros(Nfft)
for i in range(Nfft):
    x_i = X[i]
    e   = ideal_ref[i] - w.dot(x_i)
    w  += mu * e * x_i
    errors[i] = e

code13_cal = code13 + X.dot(w)
vol_cal    = code13_cal * lsb_tot

# DFT  after calibration
X_after   = np.fft.fft(vol_cal, Nfft)
est_after = np.sign(X_after[:Ntones].real)
MSE_after = np.mean((est_after - bits)**2)
BER_after = np.mean(est_after != bits)

print("*******************************************")
print("MSE after Calibration = ", MSE_after)
print("BER after Calibration = ", BER_after)
print("*******************************************")
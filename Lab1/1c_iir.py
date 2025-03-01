import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# IIR filter with transfer function H(z) = 1/(1 - z^-1)
b = [1]  # Numerator 
a = [1, -2]     # Denominator 

# use freqz from SciPy for frequency response
w, h = signal.freqz(b, a)

# calculate zeros and poles
zeros, poles, k = signal.tf2zpk(b, a)

# Plot 1. Magnitude Response (Transfer function vs Angular Frequency)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.title('Magnitude Response')
plt.xlabel('Angular frequency W [radians/sample]')
plt.ylabel('Transfer Function magnitude |H| [dB]')
plt.grid()
plt.show()

# Plot 2. Phase Response (Phase vs Angular Frequency)
plt.plot(w, np.angle(h), 'r')
plt.title('Phase Response')
plt.xlabel('Angular frequency [radians/sample]')
plt.ylabel('Phase [radians]')
plt.grid()
plt.show()

# Plot 3. Pole Zero Plot
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label="Zeros")
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label="Poles")
plt.title('Pole-Zero Plot')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.legend()
plt.grid()
plt.show()


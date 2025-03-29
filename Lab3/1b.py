import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

gm = 1
fs = 2.4e9
Cs = 15.925e-12
N  = 8

s = 1/(2*fs*Cs) 
# s is the scaling factor

# filter with transfer function H(z)
numerator   = s*np.ones(N)   
denominator = np.concatenate(([1], np.zeros(2*N - 1), [-1]))      

print(numerator)
print(denominator)

# use freqz from SciPy for frequency response
w, h = signal.freqz(numerator, denominator, worN = 8001)
 
# Plot 1. Magnitude Response 
plt.plot((w*fs/(2*np.pi)), 20 * np.log10(abs(h)), 'b')
plt.title('Magnitude Response')
plt.xlabel('Frequency f [Hz]')
plt.ylabel('Transfer Function magnitude |H| [dB]')
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

gm = 1
fs = 2.4e9
Ch = 15.425e-12
Cr = 0.5e-12
N  = 8

a1 = Ch/(Ch + Cr)
# a1 is the weighing factor

s = 1/(2*fs*(Ch+Cr)) 
# s is the scaling factor

# filter with transfer function H(z)
fir1_numerator   = s*np.ones(N) 
fir1_denominator = np.ones(1)

iir1_numerator =  np.ones(1)
iir1_denominator = np.concatenate(([1], np.zeros(N - 1), [-a1]))

fir2_numerator   = np.concatenate(([1], np.zeros(N - 1), [a1], np.zeros(N - 1), [np.power(a1,2)], np.zeros(N - 1), [np.power(a1,3)] )) 
fir2_denominator = np.ones(1)   

iir2_numerator =  np.ones(1)
iir2_denominator = np.concatenate(([1], np.zeros(8*N - 1), [-a1]))

numerator = np.convolve(fir1_numerator, iir1_numerator)
numerator = np.convolve(numerator, fir2_numerator)
numerator = np.convolve(numerator, iir2_numerator)

denominator = np.convolve(fir1_denominator, iir1_denominator)
denominator = np.convolve(denominator, fir2_denominator)
denominator = np.convolve(denominator, iir2_denominator)

print(numerator)
print(denominator)

# use freqz from SciPy for frequency response
w, h  = signal.freqz(numerator, denominator)
 
# Plot 1. Magnitude Response 
plt.plot((w*fs/(2*np.pi)), 20 * np.log10(abs(h)), 'b')
plt.title('Magnitude Response')
plt.xlabel('Frequency f [Hz]')
plt.ylabel('Transfer Function magnitude |H| [dB]')
plt.grid()
plt.show()


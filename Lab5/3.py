import numpy as np
import matplotlib.pyplot as plt

ideal_v = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
actual_v = np.array([-0.01, 0.105, 0.195, 0.28, 0.37, 0.48, 0.6, 0.75])
print("Actual Voltage levels  = ", actual_v)
print("Ideal Voltage levels   = ", ideal_v)

LSB = 0.1 # 100mV = 0.1V
n = 3 # 2^3 = 8 codes
codes =np.arange(0, 2**n)

# Offset Error = (Actual Output at Code = 000 - Ideal Output at Code = 000)
offset_error_lsb = (actual_v[0] - ideal_v[0]) / LSB
print("Offset Error in LSB = ", (actual_v[0] - ideal_v[0]))

# Full scale Error = (Actual Output at Code = 111 - Ideal Output at Code = 111)
full_scale_error_lsb = (actual_v[2**n-1] - ideal_v[2**n-1]) / LSB
print("Full scale Error in LSB = ", full_scale_error_lsb)

offset_corrected_v = actual_v - offset_error_lsb*LSB #offset corrected
print("Voltage levels after offset correction = ", offset_corrected_v)

end_point_gain_error_lsb = (offset_corrected_v[2**n-1] - ideal_v[2**n-1]) / LSB #end point at code = 7
print("End point Gain Error in LSB = ", end_point_gain_error_lsb)

gain_corrected_v = offset_corrected_v *  (ideal_v[2**n-1] / offset_corrected_v[2**n-1])
print("Voltage levels after gain correction = ", gain_corrected_v)

step = np.diff(gain_corrected_v) #actual step height
print("Actual step = ", step)

dnl = (step - LSB)/ LSB
print("DNL for codes levels = ", dnl)
max_dnl = np.max(np.abs(dnl))
print("Max DNL = ", max_dnl)


inl = np.cumsum(dnl)# cumulative sum to convert DNL to INL
print("INL for codes levels = ", inl)
max_inl = np.max(np.abs(inl))
print("Max INL = ", max_inl)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(codes[1:], dnl,  color='b', marker='o')
plt.xlabel("Digital Code Voltage Level")
plt.ylabel("dnl")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(codes[1:], inl,  color='b', marker='o')
plt.xlabel("Digital Code Voltage Level")
plt.ylabel("inl")
plt.grid()
plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
n =4 #4 bits
ramp_hs = np.array([43, 115, 85, 101, 122, 170, 75, 146, 125, 60, 95, 95, 115, 40, 120, 242])
codes =np.arange(0, 2**n)

total_sum = np.sum(ramp_hs)
ideal_LSB = total_sum/(2**n)

dnl = (ramp_hs - ideal_LSB)/ideal_LSB
print("DNL for codes levels = ", dnl)
max_dnl = np.max(np.abs(dnl))
print("Max DNL = ", max_dnl)

inl = np.cumsum(dnl)# cumulative sum to convert DNL to INL
print("INL for codes levels = ", inl)
max_inl = np.max(np.abs(inl))
print("Max INL = ", max_inl)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(codes, dnl,  color='b', marker='o')
plt.xlabel("Digital Code Voltage Level")
plt.ylabel("dnl")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(codes, inl,  color='b', marker='o')
plt.xlabel("Digital Code Voltage Level")
plt.ylabel("inl")
plt.grid()
plt.tight_layout()
plt.show()
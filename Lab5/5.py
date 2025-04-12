import numpy as np
import matplotlib.pyplot as plt
n =3 #3 bits
codes =np.arange(0, 2**n)

dnl = np.array([0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0])

offset_error = 0.5 #0.5LSB
fullscale_error = 0.5#0.5LSB

inl = np.cumsum(dnl)# cumulative sum to convert DNL to INL
print("INL for codes levels = ", inl)

ideal_output = codes #in LSB
actual_output = ideal_output + offset_error + inl
actual_output[2**n - 1] = actual_output[2**n - 1] + fullscale_error
print("Ideal Output (in LSB) = ", ideal_output)
print("Actual Output (in LSB) = ", actual_output)

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

plt.figure(figsize=(10, 5))
plt.step(codes, actual_output, where='post', color='b', marker='o')
plt.plot(codes, ideal_output, color='r', linestyle=':')
plt.xlabel("Digital Code Voltage Level")
plt.ylabel("ACtual Output (in LSB)")
plt.grid()
plt.tight_layout()
plt.show()

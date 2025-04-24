import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import time
from scipy.fft import fft, fftfreq
from scipy.signal import welch, periodogram as psd

N_coarse = 5
N_fine = 9
Vref = 1

#2 stage split pipeline SAR ADC quantizer
def pipeline_SAR_Quantize(x, channel_num):
    comparator_mismatch_std = 0.25
    comparator_mismatch_coarse = np.random.normal(0, comparator_mismatch_std, N_coarse)
    comparator_mismatch_fine   = np.random.normal(0, comparator_mismatch_std, N_fine)

    #clipping input to avoid -ve
    x = np.clip((x + Vref)/2, 0, Vref)
    #stage 1 coarse ADC
    bits_c = np.zeros(N_coarse, int)
    est = 0.0
    for i in range(N_coarse):
        step = Vref / (2**(i+1))
        thr  = est + step + comparator_mismatch_coarse[i] * Vref
        if x >= thr:
            bits_c[i] = 1
            est += step
    
    residual = x - est

    if(channel_num==1):
        residual = residual*1.5 #Gain mismatch b/w channels

   #stage 2 fine ADC 
    bits_f = np.zeros(N_fine, int)
    est_f = 0.0
    for i in range(N_fine):
        step = Vref / (2**(N_fine - i))
        thr  = est_f + step + comparator_mismatch_fine[i] * (Vref / (2**N_coarse))
        if residual >= thr:
            bits_f[i] = 1
            est_f += step
    
    code_c = bits_c.dot(1 << np.arange(N_coarse - 1, -1, -1))
    code_f = bits_f.dot(1 << np.arange(N_fine - 1, -1, -1))
    code   = (code_c << N_fine) + code_f

    return code, np.concatenate([bits_c, bits_f])

#MLP model - for calibration
def mlp_regress_calibrate(X_train, Y_train):
    mlp = MLPRegressor(
        hidden_layer_sizes=(128,),
        activation='relu', #relu activation function
        solver='adam', #optimization function
        max_iter=450,
        random_state=0
    )
    mlp.fit(X_train, Y_train)
    return mlp

def sndr_sfdr_calc(y, fs, nfft=8192):
    y = y - np.mean(y) # offset normalization
    N = len(y)
    #hanning window for clear representation
    w = np.hanning(N)
    y_w = y*w
    Y = np.fft.fft(y_w, nfft)[:nfft//2] #consider positive frequencies
    f = np.fft.fftfreq(nfft, d=1/fs)[:nfft//2] #frequencies
    psd = np.abs(Y)**2
    psd_db = 20*np.log10(psd)

    idx = np.argmax(psd)

    fbins_win  = 25
    fbins_low  = max(idx-fbins_win , 0)
    fbins_high = min(idx+fbins_win , nfft-1)

    signal_power = np.sum(psd[fbins_low:fbins_high])
    noise_power = np.sum(psd) - signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    print("*******************************************")
    print("Signal Power = ",signal_power)
    print("Noise Power = ",noise_power)
    print("SNDR in dB = ",SNDR)
    print("*******************************************")

    psd_1 = psd
    psd_1[fbins_low:fbins_high] = 0
    spur = np.max(psd_1)
    SFDR = 10*np.log10(signal_power/spur)
    print("*******************************************")
    print("Signal Power = ",signal_power)
    print("Spur = ",spur)
    print("SFDR in dB = ",SFDR)
    print("*******************************************")

    return SNDR, SFDR, f, psd_db

def run_ADC():
    start_time = time.time()

    N_train = 20000
    N_test  = 8192
    
    #inputs ->

    fs = 1e9   #input freequency
    num_cycles = 1397
    #for incommensurate sampling
    fin = fs * num_cycles / N_test
    print("*******************************************")
    print("fin = ", fin)
    print("fs = ", fs)
    print("*******************************************")
    #Training data:
    t_train = np.linspace(0, num_cycles/fin, N_train)
    #x_tr = (2 * np.random.rand(N_train) - 1)
    x_tr = np.sin(2 * np.pi * fin * t_train)

    #Testing data:
    t = np.linspace(0, num_cycles/fin, N_test*5) #timedomain
    x = np.sin(2 * np.pi * fin *t)
    t_test = np.linspace(0, num_cycles/fin, N_test)
    x_te   = np.sin(2 * np.pi * fin * t_test)

    def channels_quantize(x, channel_num):
        codes = []
        bits  = []
        for xi in x:
            c, b = pipeline_SAR_Quantize(xi, channel_num)
            codes.append(c)
            bits.append(b)
        return np.array(codes), np.array(bits)

    #channel_num is 0 for channel A and 1 for channel B
    codes_A_tr , bits_A_tr = channels_quantize(x_tr, 0)
    codes_B_tr , bits_B_tr = channels_quantize(x_tr, 1)
    codes_A_te , bits_A_te = channels_quantize(x_te, 0)
    codes_B_te , bits_B_te = channels_quantize(x_te, 1)

    def code_to_v(code):
        v = ( (code / (2**14 - 1)) * 2 - 1 )
        return v
    
    #using bits_A_tr to train such that Loss = codes_B_tr - codes_A_tr converges
    X_tr = bits_A_tr
    Y_tr = ((codes_B_tr) - (codes_A_tr)).astype(float)
    mlp  = mlp_regress_calibrate(X_tr, Y_tr)

    # Plot training loss
    plt.plot(mlp.loss_curve_)
    plt.title('ML based Calibration Loss function')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.figure(figsize=(10, 5))
    plt.title('Vin')
    plt.subplot(2, 1, 1)
    plt.plot(t[:N_test//100] * 1e9, x[:N_test//100],  color='r')
    plt.xlabel("Time (ns)")
    plt.ylabel("Vin(t) (V)")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.stem(t_test[:N_test//100], x_te[:N_test//100],  linefmt='r', markerfmt='ro')
    plt.xlabel("n")
    plt.ylabel("Ideal Samples Vin[n] (V)")
    plt.grid()
    plt.tight_layout()


    #use mlp to predict with bits_B_te
    predict    = mlp.predict(bits_A_te)
    print("*******************************************")
    mse_before = np.mean((code_to_v(codes_A_te) - code_to_v(codes_B_te))**2)
    print(f"MSE between two channels before calibration = {mse_before:.5f}")    
    mse_after  = np.mean(((code_to_v(codes_A_te + predict)) - code_to_v(codes_B_te))**2)
    print(f"MSE between two channels after calibration  = {mse_after:.5f}")
    print("*******************************************")
    
    y_before = (code_to_v(codes_A_te) + code_to_v(codes_B_te))/2
    y_after  = (code_to_v(codes_A_te + predict) + code_to_v(codes_B_te))/2

    SNDR_before, SFDR_before, f, psd_before = sndr_sfdr_calc(y_before, fs)
    SNDR_after, SFDR_after, f, psd_after = sndr_sfdr_calc(y_after, fs)

    plt.figure(figsize=(10, 5))
    plt.suptitle("Before Calibration")
    plt.subplot(3, 1, 1)
    plt.stem(t_test[:N_test//100], code_to_v(codes_A_te)[:N_test//100], linefmt='b', markerfmt='bo')
    plt.xlabel("n")
    plt.ylabel("Channel A output (V)")
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.stem(t_test[:N_test//100], code_to_v(codes_B_te)[:N_test//100], linefmt='b', markerfmt='bo')
    plt.xlabel("n")
    plt.ylabel("Channel A output (V)")
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.stem(t_test[:N_test//100], y_before[:N_test//100], linefmt='r', markerfmt='ro')
    plt.xlabel("n")
    plt.ylabel("FInal Output Vout (V)")
    plt.grid()
    plt.tight_layout()


    plt.figure(figsize=(10, 5))
    plt.suptitle("After Calibration")
    plt.subplot(3, 1, 1)
    plt.stem(t_test[:N_test//100], code_to_v(codes_A_te + predict)[:N_test//100], linefmt='b', markerfmt='bo')
    plt.xlabel("n")
    plt.ylabel("Channel A output (V)")
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.stem(t_test[:N_test//100], code_to_v(codes_B_te)[:N_test//100], linefmt='b', markerfmt='bo')
    plt.xlabel("n")
    plt.ylabel("Channel A output (V)")
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.stem(t_test[:N_test//100], y_after[:N_test//100], linefmt='r', markerfmt='ro')
    plt.xlabel("n")
    plt.ylabel("FInal Output Vout (V)")
    plt.grid()
    plt.tight_layout()


    print("*******************************************")
    print(f"Before ML Calibration: SNDR = {SNDR_before:.4f} dB, SFDR = {SFDR_before:.4f} dB")
    print(f"After  ML Calibration: SNDR = {SNDR_after:.4f} dB,  SFDR = {SFDR_after:.4f} dB")
    print("*******************************************")

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f}s")

    plt.figure()
    plt.plot(f, psd_before, label='Before ML Calibration')
    plt.plot(f, psd_after, label='After ML Calibration')
    plt.title("Power Spectral Density Before vs After Calibration")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()


    
if __name__ == '__main__':
    run_ADC()




    

from scipy.linalg import toeplitz
from pyAudioKits.audio.audio import AudioFrames,Audio,srJudge
import numpy as np
from pyAudioKits.analyse.modelBasedAnalyse import LPC,LPC1time,nextpow2
import scipy.signal

def specSubstract(input, noise, beta=0.002, frameDuration = 0.03, overlapRate = 0.5, window = None):
    """Using spectral subtraction to reduce noise. 

    input: An Audio object of signal + noise. 
    noise: An Audio object of estimate noise. 
    beta: The beta parameter. 
    frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each fram (sample points). 
    overlapRate: A float object in [0,1) for the overlapping rate of the frame. 
    window: 
            If string, it's the name of the window function (e.g., "hann")
            If tuple, it's the name of the window function and any parameters (e.g., ("kaiser", 4.0))
            If numeric, it is treated as the beta parameter of the "kaiser" window, as in scipy.signal.get_window.
            If callable, it's a function that accepts one integer argument (the window length)
            If list-like, it's a pre-computed window of the correct length Nx

    return: An Audio object of filtered signal. 
    """
    srJudge(input, noise)
    input = input.framing(frameDuration, overlapRate, window)
    noise = noise.framing(frameDuration, overlapRate, window)
    x=input.samples.T
    len_ = input.samples.shape[1]
    nFFT = 2 * 2 ** (nextpow2(len_))
    Expnt=2
    noise_mu = np.abs(np.fft.fft(noise.samples,axis=-1,n=nFFT))
    if len(noise_mu.shape) > 1:
        noise_mu=np.mean(noise_mu,axis=0)
    noise_mu = noise_mu[:,None]

    img = 1j
    spec = np.fft.fft(x, n=nFFT,axis=0)
    sig=np.abs(spec)
    theta=np.angle(spec)
    SNRseg = 10 * np.log10(np.linalg.norm(sig, 2,axis=0) ** 2 / np.linalg.norm(noise_mu, 2,axis=0) ** 2)
    def berouti(SNR):
        SNR=np.where((SNR>=-5)&(SNR<=20),4 - SNR * 3 / 20,SNR)
        SNR=np.where(SNR<-5,5,SNR)
        SNR=np.where(SNR>20,1,SNR)
        return SNR
    alpha = berouti(SNRseg)
    sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt
    diffw = sub_speech - beta * noise_mu ** Expnt
    sub_speech=np.where(diffw<0,beta*noise_mu**Expnt,sub_speech)
    x_phase = (sub_speech ** (1 / Expnt)) * (np.cos(theta) + img * np.sin(theta))
    xi = np.fft.ifft(x_phase,axis=0).real
    xi = xi[:input.samples.shape[1]]
    return AudioFrames(xi.T,input.sr,input.step,input.window).retrieve()

def wienerFilter(observed_signal,desired_signal,h_length=200, frameDuration = 0.03, overlapRate = 0.5, window = None):
    """Using Wiener filtering to reduce noise. 

    observed_signal: An Audio object of signal + noise. 
    desired_signal: An Audio object or estimated signal. 
    h_length: Orders. 
    frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each fram (sample points). 
    overlapRate: A float object in [0,1) for the overlapping rate of the frame. 
    window: 
            If string, it's the name of the window function (e.g., "hann")
            If tuple, it's the name of the window function and any parameters (e.g., ("kaiser", 4.0))
            If numeric, it is treated as the beta parameter of the "kaiser" window, as in scipy.signal.get_window.
            If callable, it's a function that accepts one integer argument (the window length)
            If list-like, it's a pre-computed window of the correct length Nx

    return: An Audio object of filtered signal. 
    """
    srJudge(observed_signal, desired_signal)
    observed_signal = observed_signal.framing(frameDuration, overlapRate, window)
    desired_signal = desired_signal.framing(frameDuration, overlapRate, window)
    row_number = h_length
    col_number = row_number
    M = col_number
    results = []
    for i in range(observed_signal.shape[0]):
        noise_=desired_signal.samples[i]
        observed_signal_=observed_signal.samples[i]
        Rx_c_full = scipy.signal.correlate(observed_signal_,observed_signal_,"full")
        k = np.argmax(Rx_c_full)
        Rx_c = Rx_c_full[k:k+M]
        Rdx_c_full = scipy.signal.correlate(noise_,observed_signal_,"full")
        Rdx_c = Rdx_c_full[k:k+M]
        A = toeplitz(Rx_c,Rx_c)
        b = np.mat(Rdx_c[:,None])
        h = np.mat(A).I*b
        h=np.squeeze(np.array(h))
        #results.append(Audio(observed_signal_ - scipy.signal.convolve(h, observed_signal_,"full")[:len(observed_signal_)],observed_signal.sr))
        results.append(scipy.signal.convolve(h, observed_signal_,"full")[:len(observed_signal_)])
    return AudioFrames(np.array(results), observed_signal.sr, observed_signal.step, observed_signal.window).retrieve()


def kalmanFilter(input,noise,numIter=7,p=20, frameDuration = 0.05):
    """Using Kalman filtering to reduce noise. 

    input: An Audio object of signal + noise.
    noise: An Audio object of estimate noise.   
    numIter: Iterating times. 
    p: Orders. 
    frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each fram (sample points). 

    return: An Audio object of filtered signal. 
    """    
    srJudge(input, noise)
    f = input.framing(frameDuration,0)
    framedSignal=f.samples
    H=np.zeros([1,p])
    H[0,-1]=1
    H=np.mat(H)
    R = np.var(noise.samples)
    Q,lpcs=LPC(f,p)
    errCov = np.mat(R * np.eye(p))
    output = np.zeros(input.samples.shape[0])
    output[0:p] = input.samples[0 : p]
    estOutput = np.mat(input.samples[:p][:,None])
    part1=np.eye(p-1)
    part0=np.zeros([p-1,1])
    part=np.concatenate([part0,part1],axis=1)
    w=framedSignal.shape[1]
    for k in range(framedSignal.shape[0]):
        oldOutput = estOutput
        if k == 0:
            iiStart = p
        else:
            iiStart = 0
        for iter in range(numIter):
            A=np.mat(np.concatenate([part,np.flip(-lpcs[k])[None]],axis=0))
            for ii in range(iiStart,w):
                aheadEstOutput = A * estOutput
                aheadErrCov  = A * errCov * A.T + H.T * Q[k] * H
                K = (aheadErrCov * H.T) / (H * aheadErrCov * H.T + R)
                estOutput = aheadEstOutput + K * (framedSignal[k, ii] - H * aheadEstOutput)
                index = ii - iiStart + p  + k * w
                output[index - p  : index] = np.squeeze(estOutput)
                errCov  = (np.mat(np.eye(p)) - K * H) * aheadErrCov
            if iter < numIter-1:
                estOutput = oldOutput
            Q[k],lpcs[k] = LPC1time(output[k * w  : (k+1) * w], p)
    f1=Audio(output,input.sr)
    return f1

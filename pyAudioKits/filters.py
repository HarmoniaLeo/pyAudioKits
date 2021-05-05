from scipy.signal import butter,sosfiltfilt,buttord
from scipy.fftpack import fft,ifft
from scipy.linalg import toeplitz
from pyAudioKits.audio import concatenate,typeJudge,AudioFrames,Audio
from pyAudioKits.freqDomainAnalyse import PSD
from pyAudioKits.timeDomainAnalyse import power
import numpy as np
from pykalman import KalmanFilter
from pyAudioKits.modelBasedAnalyse import LPC,LPC1time,nextpow2
import librosa
import math

def lowPassFilterN(input,n,f,analog=False):
    """N order low pass Butterworth filter. 

    input: An Audio object or an Audio frame object. 
    n: The order. 
    f: The cut-off frequency. 
    analog: Whether the f is the analog frenquency [0,∞) or digital frenquency [0,π)
    return: An Audio object or an Audio frame object. 
    """
    sos=butter(n,2*f/input.sr,btype='lowpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,input.samples)
    return typeJudge(input,y)

def highPassFilterN(input,n,f,analog=False):
    """N order high pass Butterworth filter. 

    input: An Audio object or an Audio frame object. 
    n: The order. 
    f: The cut-off frequency. 
    analog: Whether the f is the analog frenquency [0,∞) or digital frenquency [0,π)
    return: An Audio object or an Audio frame object. 
    """
    sos=butter(n,2*f/input.sr,btype='highpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,input.samples)
    return typeJudge(input,y)

def bandPassFilterN(input,n,fLow,fHigh,analog=False):
    """N order band pass Butterworth filter. 

    input: An Audio object or an Audio frame object. 
    n: The order. 
    fLow: The low cut-off frequency. 
    fHigh: The high cut-off frequency. 
    analog: Whether the f is the analog frenquency [0,∞) or digital frenquency [0,π)
    return: An Audio object or an Audio frame object. 
    """
    sos=butter(n,[2*fLow/input.sr,2*fHigh/input.sr],btype='bandpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,input.samples)
    return typeJudge(input,y)

def bandStopFilterN(input,n,fLow,fHigh,analog=False):
    """N order band stop Butterworth filter. 

    input: An Audio object or an Audio frame object. 
    n: The order. 
    fLow: The low cut-off frequency. 
    fHigh: The high cut-off frequency. 
    analog: Whether the f is the analog frenquency [0,∞) or digital frenquency [0,π)
    return: An Audio object or an Audio frame object. 
    """
    sos=butter(n,[2*fLow/input.sr,2*fHigh/input.sr],btype='bandstop',output='sos',analog=analog)
    y=sosfiltfilt(sos,input.samples)
    return typeJudge(input,y)

def lowPassFilter(input,fPass,fStop,ripplePass,rippleStop,analog=False):
    """Low pass Butterworth filter with specified ripple. 

    input: An Audio object or an Audio frame object. 
    fPass: The passband frequency. 
    fStop: The stopband frequency. 
    ripplePass: The passband ripple. 
    rippleStop: The stopband ripple. 
    analog: Whether the f is the analog frenquency [0,∞) or digital frenquency [0,π)
    return: An Audio object or an Audio frame object. 
    """
    n,f=buttord(fPass,fStop,ripplePass,rippleStop,analog=analog)
    sos=butter(n,2*f/input.sr,btype='lowpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,input.samples)
    return typeJudge(input,y)

def highPassFilter(input,fPass,fStop,ripplePass,rippleStop,analog=False):
    """High pass Butterworth filter with specified ripple. 

    input: An Audio object or an Audio frame object. 
    fPass: The passband frequency. 
    fStop: The stopband frequency. 
    ripplePass: The passband ripple. 
    rippleStop: The stopband ripple. 
    analog: Whether the f is the analog frenquency [0,∞) or digital frenquency [0,π)
    return: An Audio object or an Audio frame object. 
    """
    n,f=buttord(fPass,fStop,ripplePass,rippleStop,analog=analog)
    sos=butter(n,2*f/input.sr,btype='highpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,input.samples)
    return typeJudge(input,y)

def bandPassFilter(input,fLowPass,fLowStop,fHighPass,fHighStop,ripplePass,rippleStop,analog=False):
    """Band pass Butterworth filter with specified ripple. 

    input: An Audio object or an Audio frame object. 
    fLowPass: The low passband frequency. 
    fLowStop: The low stopband frequency.
    fHighPass: The high passband frequency. 
    fHighStop: The high stopband frequency. 
    ripplePass: The passband ripple. 
    rippleStop: The stopband ripple. 
    analog: Whether the f is the analog frenquency [0,∞) or digital frenquency [0,π)
    return: An Audio object or an Audio frame object. 
    """
    n,[fLow,fHigh]=buttord([fLowPass,fHighPass],[fLowStop,fHighStop],ripplePass,rippleStop,analog=analog)
    sos=butter(n,[2*fLow/input.sr,2*fHigh/input.sr],btype='bandpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,input.samples)
    return typeJudge(input,y)

def bandStopFilter(input,fLowPass,fLowStop,fHighPass,fHighStop,ripplePass,rippleStop,analog=False):
    """Band stop Butterworth filter with specified ripple. 

    input: An Audio object or an Audio frame object. 
    fLowPass: The low passband frequency. 
    fLowStop: The low stopband frequency.
    fHighPass: The high passband frequency. 
    fHighStop: The high stopband frequency. 
    ripplePass: The passband ripple. 
    rippleStop: The stopband ripple. 
    analog: Whether the f is the analog frenquency [0,∞) or digital frenquency [0,π)
    return: An Audio object or an Audio frame object. 
    """
    n,[fLow,fHigh]=buttord([fLowPass,fHighPass],[fLowStop,fHighStop],ripplePass,rippleStop,analog=analog)
    sos=butter(n,[2*fLow/input.sr,2*fHigh/input.sr],btype='bandstop',output='sos',analog=analog)
    y=sosfiltfilt(sos,input.samples)
    return typeJudge(input,y)

def specSubstract(input,noise,Thres=3,beta=0.002,G=0.9):
    """Using spectral subtraction to reduce noise. 

    input: An AudioFrames object of signal + noise. 
    noise: An AudioFrames object or an Audio frame object of estimate noise. 
    return: An AudioFrames object of signal. 
    """
    x=input.samples
    len_ = input.samples.shape[0]
    nFFT = 2 * 2 ** (nextpow2(len_))
    Expnt=2
    noise_mu=np.mean(np.abs(np.fft.fft(noise.samples,axis=0,n=nFFT)),axis=1)
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
    xi = xi[:input.samples.shape[0]]
    return AudioFrames(xi,input.sr,input.step,input.window)


class wiener:
    def __init__(self,h_length,desired_signal,observed_signal):
        """Train a wiener filter. 

        h_length: The order of wiener filter. 
        desired_signal: An Audio object. Signal without noise. 
        observed_signal: An Audio object. Signal with noise. 
        """
        row_number = h_length
        col_number = row_number
        M = col_number
        desired_signal=desired_signal.samples
        observed_signal=observed_signal.samples
        Rx_c_full = np.correlate(observed_signal,observed_signal,"full")
        k = np.argmax(Rx_c_full)
        Rx_c = Rx_c_full[k:k+M]
        Rdx_c_full = np.correlate(desired_signal,observed_signal,"full")
        Rdx_c = Rdx_c_full[k:k+M]
        A = toeplitz(Rx_c,Rx_c)
        b = np.mat(Rdx_c[:,None])
        h = np.mat(A).I*b
        self.h=np.squeeze(np.array(h))
    
    def apply(self,signal):
        """Using pre-trained Wiener filter to reduce noise. 

        signal: An Audio object of signal + noise. 
        return: An Audio object of signal. 
        """
        return Audio(np.convolve(self.h, signal.samples,"full")[:len(signal.samples)],signal.sr)


def kalmanFilter(input,noise,numIter=7,p=20):
    """Using Kalman filter to reduce noise. 

    input: An Audio object of signal + noise.
    noise: An Audio object of estimate noise.   
    numIter: Iterating times. 
    p: Orders. 
    return: An Audio object of signal. 
    """
    f = input.framing(frameDuration=0.05,overlapRate=0)
    framedSignal=f.samples.T
    p = 20
    numIter = 7
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

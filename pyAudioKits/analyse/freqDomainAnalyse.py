from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pyAudioKits.datastructures.ds import tResult, fResult
import librosa

def calculate_H(frame_w,N):
    freq = FFT(frame_w,N).samples
    Y = freq * np.conj(freq)
    sumY=np.sum(Y,axis=-1)
    if type(sumY) == np.ndarray:
        P = Y/sumY[:,None]
    else:
        P = Y/sumY
    H=-np.sum(P*(np.log2(P+1e-6)),axis=-1)
    return H

def specEntropy(input,N=None):
    """Calculate the spectral entropy of the audio. 

    input: An Audio object or an AudioFrames object. 
    N: Num of Fourier Transform points. None if use all samples. 
    return: 
        If input is an Audio object: A float object for the spectral entropy of the whole audio. 
        If input is an AudioFrames object: A frame_axis_ds object containing spectral entropy of each frame. 
    """
    if N is None:
        N=input.samples.shape[-1]
    return tResult(calculate_H(input,N), input, "bit")

def PSD(input,N=None,dB=False):
    """Calculate the power spectral density of the audio. 

    input: An Audio object or an AudioFrames object. 
    N: Num of Fourier Transform points. None if use all samples. 
    dB: Whether to express the output in gain (dB). 
    return: 
        If input is an Audio object: A freq_axis_ds_real object containing power spectral density of the whole audio. 
        If input is an AudioFrames object: A freq_frame_axis_ds_real object containing power spectral density of each frame. 
    """
    if N is None:
        N=input.samples.shape[-1]
    freq=fft(input.samples,N,axis=-1)
    result = np.power(np.abs(freq),2)/N
    if dB:
        result = 10 * np.log10(result)
        unit = "dB"
    else:
        unit = "amplitude"
    return fResult(result, input, unit)

def FFT(input,N=None):
    """Calculate the FFT of the audio. 

    input: An Audio object or an AudioFrames object (STFT). 
    N: Num of Fourier Transform points. None if use all samples. 
    return: 
        If input is an Audio object: A freq_axis_ds object containing spectrum. 
        If input is an AudioFrames object: A freq_frame_axis_ds object containing short-time spectrum. 
    """
    if N is None:
        N=input.samples.shape[-1]
    result = fft(input.samples,n=N,axis=-1)
    return fResult(result,input,"complex")

def getMaxFrequency(input,N=None,dB=False):
    """Get the frequency and the amplitude of spectrum peak. 

    input: An Audio object or an AudioFrames object. 
    N: Num of Fourier Transform points. None if use all samples. 
    dB: Whether to express the output amplitude in gain(dB). 
    return: 
        If input is an Audio object: The frequency and the amplitude of the spectrum peak of the whole audio. 
        If input is an AudioFrames object: A frame_axis_ds object containing the frequency of the spectrum peak and a frame_axis_ds object containing the amplitude of the spectrum peak. 
    """
    if N is None:
        N=input.samples.shape[-1]
    freq=FFT(input, N).samples
    indices = np.argmax(2*np.abs(freq), axis = -1)
    max_freq = freq[indices] if len(freq.shape) == 1 else freq[(np.matrix([np.arange(len(indices))]), np.matrix([indices]))][0]
    result = 2*np.abs(max_freq)/freq.shape[-1]
    if dB:
        result = 10 * np.log10(result)
        unit = "dB"
    else:
        unit = "amplitude"
    return tResult(indices*(input.sr/freq.shape[-1]), input,"Hz"), tResult(result, input,unit)
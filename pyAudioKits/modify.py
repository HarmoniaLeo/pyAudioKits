import numpy as np
from scipy.fftpack import fft,ifft
import librosa.effects
from scipy.signal import butter,sosfiltfilt,buttord
from audio import Audio

def amplify(audio,dB):
    amp=np.power(10,(dB/10))
    return Audio(audio.samples*amp,audio.sr)

def pitch_shift(audio,halfSteps):
    return Audio(librosa.effects.pitch_shift(audio.samples, audio.sr, n_steps=halfSteps),audio.sr)

def addWgn(audio,snr):
    x0=audio.samples
    P_signal = np.sum(abs(x0)**2)/len(x0)
    P_noise = P_signal/10**(snr/10.0)
    x0=np.random.randn(len(x0)) * np.sqrt(P_noise)+x0
    return Audio(x0,audio.sr)

def lowPassFilterN(audio,n,f,analog=False):
    sos=butter(n,2*f/audio.sr,btype='lowpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,audio.samples)
    return Audio(y,audio.sr)

def highPassFilterN(audio,n,f,analog=False):
    sos=butter(n,2*f/audio.sr,btype='highpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,audio.samples)
    return Audio(y,audio.sr)

def bandPassFilterN(audio,n,fLow,fHigh,analog=False):
    sos=butter(n,[2*fLow/audio.sr,2*fHigh/audio.sr],btype='bandpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,audio.samples)
    return Audio(y,audio.sr)

def bandStopFilterN(audio,n,fLow,fHigh,analog=False):
    sos=butter(n,[2*fLow/audio.sr,2*fHigh/audio.sr],btype='bandstop',output='sos',analog=analog)
    y=sosfiltfilt(sos,audio.samples)
    return Audio(y,audio.sr)

def lowPassFilter(audio,fPass,fStop,ripplePass,rippleStop,analog=False):
    n,f=buttord(fPass,fStop,ripplePass,rippleStop,analog=analog)
    sos=butter(n,2*f/audio.sr,btype='lowpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,audio.samples)
    return Audio(y,audio.sr)

def highPassFilter(audio,fPass,fStop,fLowStart,fLowEnd,fHighStart,fHighEnd,ripplePass,rippleStop,analog=False):
    n,f=buttord(fPass,fStop,ripplePass,rippleStop,analog=analog)
    sos=butter(n,2*f/audio.sr,btype='highpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,audio.samples)
    return Audio(y,audio.sr)

def bandPassFilter(audio,fLowPass,fLowStop,fHighPass,fHighStop,fLowStart,fLowEnd,fHighStart,fHighEnd,ripplePass,rippleStop,analog=False):
    n,[fLow,fHigh]=buttord([fLowPass,fHighPass],[fLowStop,fHighStop],ripplePass,rippleStop,analog=analog)
    sos=butter(n,[2*fLow/audio.sr,2*fHigh/audio.sr],btype='bandpass',output='sos',analog=analog)
    y=sosfiltfilt(sos,audio.samples)
    return Audio(y,audio.sr)

def bandStopFilter(audio,fLowPass,fLowStop,fHighPass,fHighStop,fLowStart,fLowEnd,fHighStart,fHighEnd,ripplePass,rippleStop,analog=False):
    n,[fLow,fHigh]=buttord([fLowPass,fHighPass],[fLowStop,fHighStop],ripplePass,rippleStop,analog=analog)
    sos=butter(n,[2*fLow/audio.sr,2*fHigh/audio.sr],btype='bandstop',output='sos',analog=analog)
    y=sosfiltfilt(sos,audio.samples)
    return Audio(y,audio.sr)
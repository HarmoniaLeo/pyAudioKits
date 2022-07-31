import numpy as np
import scipy
import scipy.signal
from pyAudioKits.datastructures.ds import time_delta_frame_axis_ds, time_delta_axis_ds, tResult
import librosa
        
def energyCal(frame_w):
    return np.sum(np.power(frame_w,2),axis=-1)

def overzeroCal(frame_w,fs):
    sgs=np.sign(frame_w.T)
    overzeros=np.sum(np.abs(sgs[1:]-sgs[:-1]),axis=0)
    return overzeros

def power(input, dB = False):
    """Calculate the power of the audio. 

    input: An Audio object or an AudioFrames object. 
    dB: Whether to express the result in the form of gain (dB). 
    return: 
        If input is an Audio object: Power of the whole audio. 
        If input is an AudioFrames object: A frame_axis_ds object containing power of each frame. 
    """
    result = np.sum(input.samples**2,axis=-1)/input.samples.shape[-1]
    if dB:
        result = 10 * np.log10(result)
        unit = "dB"
    else:
        unit = "amplitude"
    return tResult(result, input, unit)

def energy(input, dB = False):
    """Calculate the energy of the audio. 

    input: An Audio object or an AudioFrames object. 
    dB: Whether to express the result in the form of gain (dB). 
    return: 
        If input is an Audio object: Energy of the whole audio. 
        If input is an AudioFrames object: A frame_axis_ds object containing energy of each frame. 
    """
    result = energyCal(input.samples)
    if dB:
        result = 10 * np.log10(result)
        unit = "dB"
    else:
        unit = "amplitude"
    return tResult(result, input, unit)

def zerocrossing(input):
    """Calculate the zero crossing rate of the audio. 

    input: An AudioFrames object. 
    return: A frame_axis_ds object containing zero crossing rate of each frame. 
    """
    return tResult(overzeroCal(input.samples,input.sr), input, "Hz")
    
def autocorr(input):
    """Calculate the auto correlation function of the audio. 

    input: An Audio object or an AudioFrames object. 
    return: 
        If input is an Audio object: A time_delta_axis_ds object containing auto correlation function.
        If input is an AudioFrames object: A time_delta_frame_axis_ds object containing short-time auto correlation result for each frames.
    """
    if len(input.shape) == 2:
        corr = np.array([scipy.signal.correlate(input.samples[i],input.samples[i],"full") for i in range(input.shape[0])])
    else:
        corr = scipy.signal.correlate(input.samples,input.samples,"full")
    #corr = np.correlate(input.samples,input.samples,"full")
    if len(corr.shape) == 2:
        return time_delta_frame_axis_ds(corr, input.sr, input.step, "value")
    else:
        return time_delta_axis_ds(corr, input.sr, "value")

def snr(signal,signalAndNoise):
    """Calculate the SNR(dB). 
    
    signal: An Audio object. The ground truth of the signal. 
    signalAndNoise: An Audio object. Signal mixed with noise. 
    return: A float object of the SNR(dB). 
    """
    noise=signalAndNoise-signal
    signalPower=power(signal)
    noisePower=power(noise)
    return 10*np.log10(signalPower/noisePower)
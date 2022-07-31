import numpy as np
from .audio import Audio
from pyAudioKits.analyse import power
from pyAudioKits.audio.audio import srJudge

def create_Single_Freq_Audio(amp,freq,sr,time,phase = 0):
    """Generate a sine wave signal. 

    amp: A float object for amplitude. 
    freq: A float object for frequency (Hz). 
    sr: An int object for sample rate.
    time: An int object for duration (seconds).  
    phase: The phase of the sine wave. Specifying None will generate independently and uniformly distributed random phase between [-π,π] each moment.

    return: An Audio object. 
    """
    ts=np.arange(0,time,1/sr)
    if phase is None:
        phase = np.random.uniform(np.zeros_like(ts)-np.pi,np.zeros_like(ts)+np.pi)
    y=amp*np.sin(2*np.pi*freq*ts+phase)
    return Audio(y,sr)

def mixWithSNR(signal,noise,snr,maintain="signal"):
    """Mixing signal and noise with specific SNR(dB). The signal and noise should have the same sample rate. 
    
    signal: An Audio object. The signal without noise. 
    noise: An Audio object. Noise to mix with signal. 
    snr: A float object. The SNR(dB) between signal and noise. 
    maintain: maintain="signal" to maintain the intensity of signal while maintain="noise" to maintain the intensity of noise. 
    return: An Audio object. Signal mixed with noise. 
    """
    srJudge(signal, noise)
    signalPower=power(signal)
    noisePower=power(noise)
    snrNow=10*np.log10(signalPower/noisePower)
    r=snr/snrNow
    x=np.sqrt(signalPower/noisePower*((noisePower/signalPower)**r))
    if maintain=="signal":
        return signal+noise*x
    if maintain=="noise":
        return signal/x+noise
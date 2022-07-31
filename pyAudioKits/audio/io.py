import librosa
from .audio import Audio
import numpy as np

def read_Audio(direction):
    """Get audio file from specified path. Dual channel audio returns a tuple. 
    
    direction: The path.
    return:
        An Audio object when reading single channel audio.
        A tuple with two Audio objects when reading double channels audio. 
    """
    y,r=librosa.load(direction,sr=None)
    if len(y.shape)==2:
        return (Audio(y[0],r),Audio(y[1],r))
    elif len(y.shape)==1:
        return Audio(y,r)
    else:
        raise Exception("Wrong number of tracks. ")


def save_Audio_DoubleTracks(audio1,audio2,direction):
    """Combine two audio with the same length and sampling rate into a dual channel audio and save it to the specified path.
    
    audio1: An Audio object for the first channel. 
    audio2: An Audio object for the second channel. 
    direction: The saving path. 
    """
    if audio1.sr!=audio2.sr:
        raise Exception("Two audios should have the same sample rate. ")
    librosa.output.write_wav(direction,np.asfortranarray(np.array([np.real(audio1.samples),np.real(audio2.samples)])),audio1.sr)
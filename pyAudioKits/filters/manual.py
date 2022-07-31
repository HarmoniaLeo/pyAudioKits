from pyAudioKits.audio.audio import typeJudge
from scipy.signal import lfilter, filtfilt, lfilter_zi

def ltiFilter(input,numerators,denominators,zero_phase=False):
    """LTI filter design by specifying the denominator and numerator coefficients of the system function. 

    input: An Audio object or an Audio frame object. 
    numerators: An Numpy array of the numerator coefficients of the system function. 
    denominators: An Numpy array of the denominator coefficients of the system function. 
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """
    if zero_phase:
        y=filtfilt(numerators,denominators,input.samples, axis=-1)
    else:
        y,_=lfilter(numerators,denominators,input.samples, axis=-1,zi=input.samples[-1]*lfilter_zi(numerators,denominators))
    return typeJudge(input,y)
from pyAudioKits.audio.audio import typeJudge
from scipy.signal import butter,sosfiltfilt,buttord, sosfilt, sosfilt_zi
import numpy as np

transform_mapping = {"frequency/Hz": lambda x, y: x * 2 / y.sr, "frequency/(rad/s)": lambda x, y: x / y.sr / np.pi, "normalized frequency/Hz": lambda x, y: x, "normalized frequency/(rad/s)": lambda x, y: x / np.pi}
analog_mapping = {"frequency/Hz": False, "frequency/(rad/s)": False, "normalized frequency/Hz": False, "normalized frequency/(rad/s)": False}

def lowPassButterN(input,n,f,freq_type = "frequency/Hz", zero_phase = True):
    """N order low pass Butterworth filter. 

    input: An Audio object or an AudioFrames object. 
    n: The order. 
    f: The cut-off frequency. 
    freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """

    sos=butter(n,transform_mapping[freq_type](f,input),btype='lowpass',output='sos',analog=analog_mapping[freq_type])
    if zero_phase:
        y=sosfiltfilt(sos,input.samples,axis=-1)
    else:
        y,_=sosfilt(sos,input.samples,axis=-1,zi=input.samples[-1]*sosfilt_zi(sos))
    return typeJudge(input,y)

def highPassButterN(input,n,f,freq_type = "frequency/Hz", zero_phase = True):
    """N order high pass Butterworth filter. 

    input: An Audio object or an AudioFrames object. 
    n: The order. 
    f: The cut-off frequency. 
    freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """
    sos=butter(n,transform_mapping[freq_type](f,input),btype='highpass',output='sos',analog=analog_mapping[freq_type])
    if zero_phase:
        y=sosfiltfilt(sos,input.samples,axis=-1)
    else:
        y,_=sosfilt(sos,input.samples,axis=-1,zi=input.samples[-1]*sosfilt_zi(sos))
    return typeJudge(input,y)

def bandPassButterN(input,n,fLow,fHigh,freq_type = "frequency/Hz", zero_phase = True):
    """N order band pass Butterworth filter. 

    input: An Audio object or an AudioFrames object. 
    n: The order. 
    fLow: The low cut-off frequency. 
    fHigh: The high cut-off frequency. 
    freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """
    sos=butter(n,[transform_mapping[freq_type](fLow,input),transform_mapping[freq_type](fHigh,input)],btype='bandpass',output='sos',analog=analog_mapping[freq_type])
    if zero_phase:
        y=sosfiltfilt(sos,input.samples,axis=-1)
    else:
        y,_=sosfilt(sos,input.samples,axis=-1,zi=input.samples[-1]*sosfilt_zi(sos))
    return typeJudge(input,y)

def bandStopButterN(input,n,fLow,fHigh,freq_type = "frequency/Hz", zero_phase = True):
    """N order band stop Butterworth filter. 

    input: An Audio object or an AudioFrames object. 
    n: The order. 
    fLow: The low cut-off frequency. 
    fHigh: The high cut-off frequency. 
    freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """
    sos=butter(n,[transform_mapping[freq_type](fLow,input),transform_mapping[freq_type](fHigh,input)],btype='bandstop',output='sos',analog=analog_mapping[freq_type])
    if zero_phase:
        y=sosfiltfilt(sos,input.samples,axis=-1)
    else:
        y,_=sosfilt(sos,input.samples,axis=-1,zi=input.samples[-1]*sosfilt_zi(sos))
    return typeJudge(input,y)

def lowPassButter(input,fPass,fStop,ripplePass,rippleStop,freq_type = "frequency/Hz", zero_phase = True):
    """Low pass Butterworth filter with specified ripple. 

    input: An Audio object or an AudioFrames object. 
    fPass: The passband frequency. 
    fStop: The stopband frequency. 
    ripplePass: The passband ripple. The signal will loses no more than ripplePass dB in the passband.
    rippleStop: The stopband ripple. The signal will have at least rippleStop dB attenuation in the stopband. 
    freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """
    n,f=buttord(transform_mapping[freq_type](fPass,input),transform_mapping[freq_type](fStop,input),ripplePass,rippleStop,analog=analog_mapping[freq_type])
    sos=butter(n,f,btype='lowpass',output='sos',analog=analog_mapping[freq_type])
    if zero_phase:
        y=sosfiltfilt(sos,input.samples,axis=-1)
    else:
        y,_=sosfilt(sos,input.samples,axis=-1,zi=input.samples[-1]*sosfilt_zi(sos))
    return typeJudge(input,y)

def highPassButter(input,fPass,fStop,ripplePass,rippleStop,freq_type = "frequency/Hz", zero_phase = True):
    """High pass Butterworth filter with specified ripple. 

    input: An Audio object or an AudioFrames object. 
    fPass: The passband frequency. 
    fStop: The stopband frequency. 
    ripplePass: The passband ripple. The signal will loses no more than ripplePass dB in the passband.
    rippleStop: The stopband ripple. The signal will have at least rippleStop dB attenuation in the stopband
    freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """
    n,f=buttord(transform_mapping[freq_type](fPass,input),transform_mapping[freq_type](fStop,input),ripplePass,rippleStop,analog=analog_mapping[freq_type])
    sos=butter(n,f,btype='highpass',output='sos',analog=analog_mapping[freq_type])
    if zero_phase:
        y=sosfiltfilt(sos,input.samples,axis=-1)
    else:
        y,_=sosfilt(sos,input.samples,axis=-1,zi=input.samples[-1]*sosfilt_zi(sos))
    return typeJudge(input,y)

def bandPassButter(input,fLowPass,fLowStop,fHighPass,fHighStop,ripplePass,rippleStop,freq_type = "frequency/Hz", zero_phase = True):
    """Band pass Butterworth filter with specified ripple. 

    input: An Audio object or an AudioFrames object. 
    fLowPass: The low passband frequency. 
    fLowStop: The low stopband frequency.
    fHighPass: The high passband frequency. 
    fHighStop: The high stopband frequency. 
    ripplePass: The passband ripple. The signal will loses no more than ripplePass dB in the passband.
    rippleStop: The stopband ripple. The signal will have at least rippleStop dB attenuation in the stopband
    freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """
    n,[fLow,fHigh]=buttord([transform_mapping[freq_type](fLowPass,input),transform_mapping[freq_type](fHighPass,input)],[transform_mapping[freq_type](fLowStop,input),transform_mapping[freq_type](fHighStop,input)],ripplePass,rippleStop,analog=analog_mapping[freq_type])
    sos=butter(n,[fLow,fHigh],btype='bandpass',output='sos',analog=analog_mapping[freq_type])
    if zero_phase:
        y=sosfiltfilt(sos,input.samples,axis=-1)
    else:
        y,_=sosfilt(sos,input.samples,axis=-1,zi=input.samples[-1]*sosfilt_zi(sos))
    return typeJudge(input,y)

def bandStopButter(input,fLowPass,fLowStop,fHighPass,fHighStop,ripplePass,rippleStop,freq_type = "frequency/Hz", zero_phase = True):
    """Band stop Butterworth filter with specified ripple. 

    input: An Audio object or an AudioFrames object. 
    fLowPass: The low passband frequency. 
    fLowStop: The low stopband frequency.
    fHighPass: The high passband frequency. 
    fHighStop: The high stopband frequency. 
    ripplePass: The passband ripple. The signal will loses no more than ripplePass dB in the passband.
    rippleStop: The stopband ripple. The signal will have at least rippleStop dB attenuation in the stopband
    freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
    zero_phase: Use bi-directional filtering to maintain a phase response of 0. 
    return: 
        An Audio object if the input is an Audio object. 
        An AudioFrames object if the input is an AudioFrames object. 
    """
    n,[fLow,fHigh]=buttord([transform_mapping[freq_type](fLowPass,input),transform_mapping[freq_type](fHighPass,input)],[transform_mapping[freq_type](fLowStop,input),transform_mapping[freq_type](fHighStop,input)],ripplePass,rippleStop,analog=analog_mapping[freq_type])
    sos=butter(n,[fLow,fHigh],btype='bandstop',output='sos',analog=analog_mapping[freq_type])
    if zero_phase:
        y=sosfiltfilt(sos,input.samples,axis=-1)
    else:
        y,_=sosfilt(sos,input.samples,axis=-1,zi=input.samples[-1]*sosfilt_zi(sos))
    return typeJudge(input,y)
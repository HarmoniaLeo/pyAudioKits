from python_speech_features import mfcc,delta
from sklearn.linear_model import LinearRegression
import numpy as np
from pyAudioKits.audio import Audio,typeJudge,isType
import librosa
from scipy.fftpack import fft,ifft
import ctypes as ct

class FloatBits(ct.Structure):
    _fields_ = [
        ('M', ct.c_uint, 23),
        ('E', ct.c_uint, 8),
        ('S', ct.c_uint, 1)
    ]


class Float(ct.Union):
    _anonymous_ = ('bits',)
    _fields_ = [
        ('value', ct.c_float),
        ('bits', FloatBits)
    ]


def nextpow2(x):
    if x < 0:
        x = -x
    if x == 0:
        return 0
    d = Float()
    d.value = x
    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1

def MFCC(input,p=13,frameDuration=0.03,overlapRate=0.5):
    """Calculate the MFCC features of the Audio. The function itself does the framing. 

    input: An Audio object. 
    p: MFCC order. The number of features of each frame will be 3*p
    frameDuration: A float object for the duration of each frame (seconds). 
    overlapRate: A float object in [0,1) for the overlapping rate of the frame. 

    return: A 2-D NumPy array of MFCC features. Each row will be MFCC features of one frame. 
    """
    isType(input,"Audio")
    windowLength=int(frameDuration*input.sr)
    step=int(windowLength*(1-overlapRate))
    wav_feature=mfcc(input.samples,input.sr,numcep=p,winlen=windowLength/input.sr,winstep=step/input.sr)
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    return np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))

def levinson(r, lpcOrder):
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]
    for k in range(1, lpcOrder):
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]
        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)
        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)
        a = np.array(U) + lam * np.array(V)
        e[k + 1] = e[k] * (1.0 - lam * lam)
    return a, e[-1]

def myLPC(x,p):
    X = fft(x,2**nextpow2(len(x)))
    R = ifft(abs(X)**2)
    R = R/len(x)
    R = np.real(R)
    a, e  = levinson(R, p)
    a=np.real(a)
    return a,e

def LPC1time(input,p):
    x=input
    lpc,e=myLPC(np.asfortranarray(x), p)
    lpc=lpc[1:]
    return e,lpc

def LPC(input,p=10):
    """Matlab style LPC for each frame. 

    input: An AudioFrames object. 
    p: LPC order. 

    return: 
        List of LPC error of each frame. 
        List of LPC coefficient of each frame. 
    """
    isType(input,"AudioFrames")
    y1=input.samples
    es=[]
    ws=[]
    for i in range(0,y1.shape[1]):
        x=np.squeeze(y1[:,i])
        e,lpc=LPC1time(np.asfortranarray(x), p)
        es.append(e)
        ws.append(lpc)
    return es,ws
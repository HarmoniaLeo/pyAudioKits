from scipy.fftpack import fft,ifft
from pyAudioKits.audio import Audio,typeJudge,isType
import matplotlib.pyplot as plt
import librosa
import numpy as np


def calculate_H(frame_w,fs):
    N=1024
    freq = fft(frame_w,N,axis=0)
    spect=np.real(10*np.log10(freq+1e-6))
    Y = spect*spect
    sumY=np.sum(Y[:,0:int(N/2)],axis=0)
    P = Y/sumY
    H=-np.sum(P[:,0:int(N/2)]*(np.log2(P[:,0:int(N/2)]+1e-6)),axis=0)
    return H

def specEntropy(input):
    """Calculate the spectral entropy of the audio. 

    input: An Audio object or an AudioFrames object. 
    return: 
        If input is an Audio object: Spectral entropy of the whole audio. 
        If input is an AudioFrame object: A numpy array for spectral entropy of each frame. 
    """
    isType(input)
    return calculate_H(input.samples,input.sr)

def PSD(input,N=None):
    """Calculate the power spectral density of the audio. 

    input: An Audio object or an AudioFrames object. 
    N: Num of Fourier Transform points. None if use all points. 
    return: 
        If input is an Audio object: Power spectral density of the whole audio. 
        If input is an AudioFrame object: A numpy array for power spectral density of each frame. 
    """
    isType(input)
    if N is None:
        N=input.samples.shape[0]
    freq=fft(input.samples,N,axis=0)
    return np.power(freq,2)/N


class AudioFrequency:
    def __init__(self,x,y,input):
        self.freqPoints=x
        self.freq=y
        self.input=input
        self.sr=input.sr

    def plot(self,minf=0,maxf=None,axes=None,imgPath=None):
        """If the AudioFrequency object was derived from an Audio object: 
        To draw the spectrum on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 
        

        minf: The starting frequency. Default = 0. 
        maxf: The ending frequency. Default = Half of the sample rate. 
        axes: A matplotlib.pyplot.axes object. 
        imgPath: The path to save the graph. 

        If the AudioFrequency object was derived from an AudioFrame object: Display the spectrogram
        """
        if len(self.freq.shape)==1:
            if maxf==None:
                maxf=int(self.sr/2)
            minPoint=int(np.ceil(minf/self.sr*(self.freqPoints.shape[0])))
            maxPoint=int(np.ceil(maxf/self.sr*(self.freqPoints.shape[0])))
            if axes==None:
                plt.xlabel("frequency/Hz")
                plt.ylabel("amplitude")
                plt.plot(self.freqPoints[minPoint:maxPoint],2*np.abs(self.freq)[minPoint:maxPoint])
                if imgPath==None:
                    plt.show()
                else:
                    plt.savefig(imgPath,dpi=500, bbox_inches = 'tight')
                plt.clf()
            else:
                axes.plot(self.freqPoints[minPoint:maxPoint],2*np.abs(self.freq)[minPoint:maxPoint])
        else:
            self.input.spec()

    def IFFT(self):
        """Transform from frequency domain to time domain. 

        return: An Audio object or an AudioFrames object, the transform result. 
        """
        return typeJudge(self.input,ifft(self.freq,axis=0))

    def getMaxFrequency(self):
        """Get the frequency and the amplitude of spectrum peak. 

        If the AudioFrequency object was derived from an Audio object: 
            return:
                Frequency of spectrum peak.
                Spectrum peak. 
        
        If the AudioFrequency object was derived from an AudioFrame object: 
            return:
                A numpy array for frequency of each frame's spectrum peak.
                A numpy array of each frame's spectrum peak.
        """
        return np.argmax(2*np.abs(self.freq))*(self.sr/self.freq.shape[0]),2*np.abs(self.freq[np.argmax(2*np.abs(self.freq))])/self.freq.shape[0]


def FFT(input,N=None):
    """Calculate the FFT of the audio. 

    audio: An Audio object or an AudioFrames object (STFT). 
    N: Num of Fourier Transform points. None if use all points. 
    return: An AudioFrequency object. 
    """
    isType(input)
    sr=input.sr
    if N is None:
        N=input.samples.shape[0]
    return AudioFrequency(np.arange(0,sr,sr/N),fft(input.samples,axis=0),input)
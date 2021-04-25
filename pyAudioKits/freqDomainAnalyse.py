from scipy.fftpack import fft,ifft
from audio import Audio
import matplotlib.pyplot as plt
import numpy as np

class AudioFrequency:
    def __init__(self,x,y,sr):
        self.__x=x
        self.__f=y
        self.__sr=sr

    def plot(self,maxf=None,axes=None,imgPath=None):
        if maxf==None:
            maxf=int(self.__sr/2)
        maxPoint=int(np.ceil(maxf/self.__sr*(self.__x.shape[0])))
        if axes==None:
            plt.xlabel("frequency/Hz")
            plt.ylabel("amplitude")
            plt.plot(self.__x[:maxPoint],2*np.abs(self.__f)[:maxPoint])
            if imgPath==None:
                plt.show()
            else:
                plt.savefig(imgPath,dpi=500, bbox_inches = 'tight')
            plt.clf()
        else:
            axes.plot(self.__x[:maxPoint],2*np.abs(self.__f)[:maxPoint])
        
    def IFFT(self):
        return Audio(ifft(self.__f),self.__sr)

    def getMaxFrequency(self):
        return np.argmax(2*np.abs(self.__f))*(self.__sr/self.__f.shape[0]),2*np.abs(self.__f[np.argmax(2*np.abs(self.__f))])/self.__f.shape[0]
    

def FFT(audio):
    sr=audio.sr
    tfLength=audio.samples.shape[0]
    return AudioFrequency(np.arange(0,sr,sr/tfLength),fft(audio.samples),audio.sr)

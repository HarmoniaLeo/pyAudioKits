import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import librosa

def energyCal(frame_w):
    return 10*np.log(np.sum(np.power(frame_w,2),axis=1)+1e-6)

def calculate_H(frame_w,fs):
    N=1024
    freq = fft(frame_w,N,axis = 1)
    spect=np.real(10*np.log10(freq+1e-6))
    Y = spect*spect
    sumY=np.sum(Y[:,0:int(N/2)],axis=1)
    sumY=np.array([sumY]).T
    P = Y/sumY
    H=-np.sum(P[:,0:int(N/2)]*(np.log2(P[:,0:int(N/2)]+1e-6)),axis=1)
    return H

def overzeroCal(frame_w,fs):
    sgs=np.sign(frame_w)
    overzeros=np.sum(np.abs(sgs[:,1:]-sgs[:,:-1]),axis=1)
    return overzeros

def logEnergy(audio,frameDuration=None,overlapRate=0.5):
    if frameDuration==None:
        return 10*np.log10(np.sum(np.power(audio.samples,2)))
    else:
        windowLength=int(frameDuration*audio.sr)
        step=int(windowLength*(1-overlapRate))
        y=audio.samples
        l=y.shape[0]
        indexer = np.arange(windowLength)[None, :] + step*np.arange(int((l-windowLength)/step))[:, None]
        flatten=y[indexer]
        filters=librosa.filters.get_window(window = "hamming", Nx = windowLength, fftbins = False)
        flatten=flatten*filters
        energys=energyCal(flatten)
        return energys

def VAD(audio,energyPerLow=12.5,energyPerHigh=40,overzeroPer=45,minVoice=0.15,maxSilence=0.08,frameDuration=0.03,overlapRate=0.5):
    windowLength=int(frameDuration*audio.sr)
    step=int(windowLength*(1-overlapRate))
    l=audio.samples.shape[0]
    indexer = np.arange(windowLength)[None, :] + step*np.arange(int((l-windowLength)/step))[:, None]
    flatten=audio.samples[indexer]
    filters=librosa.filters.get_window(window = "hamming", Nx = windowLength, fftbins = False)
    flatten=flatten*filters
    energys=energyCal(flatten)
    overzeros=overzeroCal(flatten,audio.sr)
    clip1=np.percentile(energys,energyPerLow)
    clip2=np.percentile(energys,energyPerHigh)
    clip3=np.percentile(overzeros,overzeroPer)
    label=np.where(energys>clip2,1,0)
    label=np.where((energys>clip1)&(overzeros>clip3),1,label)
    count=0
    startPoint=0
    for i in range(1,len(label)):
        if label[i]!=label[i-1]:
            if label[i-1]==0:
                length=i-startPoint
                if length*frameDuration>maxSilence:
                    label[startPoint:i-1]=1
            else:
                startPoint=i
            count=0
        count+=1
    count=0
    startPoint=0
    for i in range(1,len(label)):
        if label[i]!=label[i-1]:
            if label[i-1]==1:
                length=i-startPoint
                if length*frameDuration<minVoice:
                    label[startPoint:i-1]=0
            else:
                startPoint=i
            count=0
        count+=1
    return label

def VADPlot(audio,label,frameDuration=0.03,overlapRate=0.5,imgPath=None):
    windowLength=int(frameDuration*audio.sr)
    step=int(windowLength*(1-overlapRate))
    label2=np.array([label[i] for i in range(0,label.shape[0]) for n in range(0,step)])
    x=np.arange(0,label2.shape[0]/audio.sr,1/audio.sr)
    y=audio.samples[:label2.shape[0]]
    plt.xlabel("t/s")
    plt.ylabel("amplitude")
    plt.plot(x,np.where(label2==1,y,None),color="m")
    plt.plot(x,np.where(label2==0,y,None))
    if imgPath!=None:
        plt.savefig(imgPath,dpi=500, bbox_inches = 'tight')
    else:
        plt.show()
    plt.clf()

def VADSlice(audio,label,frameDuration=0.03,overlapRate=0.5):
    windowLength=int(frameDuration*audio.sr)
    step=int(windowLength*(1-overlapRate))
    label=label.tolist()
    slices=[]
    label2=[]
    start=0
    for i in range(0,len(label)-1):
        if (label[i+1]==0 and label[i]!=0) or (label[i+1]!=0 and label[i]==0):
            slices.append(audio.select(start*step/audio.sr,(i+1)*step/audio.sr))
            if label[i]==0:
                label2.append(0)
            else:
                label2.append(1)
            start=i+1
    return np.array(slices),np.array(label2)
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt

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
        step=int(windowLength*overlapRate)
        y=audio.samples
        l=y.shape[0]
        indexer = np.arange(windowLength)[None, :] + step*np.arange(int((l-windowLength)/step))[:, None]
        flatten=y[indexer]
        energys=energyCal(flatten)
        return energys

def VAD(audio,energyPer,overzeroPer,frameDuration=0.03,overlapRate=0.5):
    windowLength=int(frameDuration*audio.sr)
    step=int(windowLength*overlapRate)
    l=audio.samples.shape[0]
    indexer = np.arange(windowLength)[None, :] + step*np.arange(int((l-windowLength)/step))[:, None]
    flatten=audio.samples[indexer]
    energys=energyCal(flatten)
    overzeros=overzeroCal(flatten,audio.sr)
    clip1=np.percentile(energys,energyPer)
    clip2=np.percentile(overzeros,overzeroPer)
    label1=np.where(energys>=clip1,1,0)
    label2=np.where(overzeros>=clip2,1,0)
    difflabel1=label1[1:]-label1[:-1]
    difflabel2=label2[1:]-label2[:-1]
    if label1[0]==1:
        difflabel1[0]=1
    if label2[0]==1:
        difflabel2[0]=1
    label=np.zeros(difflabel1.shape[0])
    voice=False
    veriVoice=False
    startPoint=0
    for i in range(0,difflabel1.shape[0]):
        if (not voice) and (difflabel2[i]==1):
            voice=True
            startPoint=i
        if (not voice) and (difflabel1[i]==1):
            voice=True
            veriVoice=True
            startPoint=i
        if (voice) and (difflabel1[i]==1):
            veriVoice=True
        if ((veriVoice) and (difflabel1[i]==-1) and (label2[i]==0)) or \
            ((veriVoice) and (difflabel2[i]==-1) and (label1[i]==0)):
            label[startPoint:i]=1
            veriVoice=False
            voice=False
        if (not veriVoice) and (difflabel2[i]==-1):
            voice=False
    return label

def VADPlot(audio,label,frameDuration=0.03,overlapRate=0.5,imgPath=None):
    windowLength=int(frameDuration*audio.sr)
    step=int(windowLength*overlapRate)
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
    step=int(windowLength*overlapRate)
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
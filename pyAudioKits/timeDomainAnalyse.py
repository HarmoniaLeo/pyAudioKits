import numpy as np
import matplotlib.pyplot as plt
import librosa
from pyAudioKits.audio import isType

def energyCal(frame_w):
    return 10*np.log(np.sum(np.power(frame_w,2),axis=0)+1e-6)

def overzeroCal(frame_w,fs):
    sgs=np.sign(frame_w)
    overzeros=np.sum(np.abs(sgs[1:,:]-sgs[:-1,:]),axis=0)
    return overzeros

def power(input):
    """Calculate the power of the audio. 

    input: An Audio object or an AudioFrames object. 
    return: 
        If input is an Audio object: Power of the whole audio. 
        If input is an AudioFrame object: A numpy array for power of each frame. 
    """
    isType(input)
    return np.sum(input.samples**2,axis=0)/input.samples.shape[0]

def logEnergy(input):
    """Calculate the log energy of the audio. 

    input: An Audio object or an AudioFrames object. 
    return: 
        If input is an Audio object: Log energy of the whole audio. 
        If input is an AudioFrame object: A numpy array for log energy of each frame. 
    """
    isType(input)
    return energyCal(input.samples)

def overzero(input):
    """Calculate the over-zero rate of the audio. 

    input: An AudioFrames object. 
    return: A numpy array for over-zero rate of each frame. 
    """
    isType(input,t="AudioFrames")
    return overzeroCal(input.samples,input.sr)

def correlate(input1,input2):
    """Calculate the correlation function between audio1 and audio2. 

    input1: The first audio or AudioFrames. 
    input2: The second audio or AudioFrames. 
    return: 
        A numpy array for correlation result with length of (len(audio1.samples)+len(audio2.samples)-1).
        A 2-D numpy array for short-time correlation result for each frames.
    """
    if type(input1)!=type(input2):
        raise Exception("The two inputs should be the same type. ")
    isType(input1)
    isType(input2)
    return np.correlate(input1.samples,input2.samples,"full")

def snr(signal,signalAndNoise):
    """Calculate the SNR(dB). 
    
    signal: An Audio object. The ground truth of the signal. 
    signalAndNoise: An Audio object. Signal mixed with noise. 
    return: A float object of the SNR(dB). 
    """
    if type(signal)!=type(signalAndNoise):
        raise Exception("The signal and signalAndNoise should both be Audio. ")
    isType(signal,"Audio")
    isType(signalAndNoise,"Audio")
    noise=signalAndNoise-signal
    signalPower=power(signal)
    noisePower=power(noise)
    return 10*np.log10(signalPower/noisePower)

def mixWithSNR(signal,noise,snr,maintain="signal"):
    """Mixing signal and noise with specific SNR(dB). 
    
    signal: An Audio object. The signal without noise. 
    noise: An Audio object. Noise to mix with signal. 
    snr: A float object. The SNR(dB) between signal and noise. 
    maintain: maintain="signal" to maintain the intensity of signal while maintain="noise" to maintain the intensity of noise. 
    return: An Audio object. Signal mixed with noise. 
    """
    if type(signal)!=type(noise):
        raise Exception("The signal and noise should both be Audio. ")
    isType(signal,"Audio")
    isType(noise,"Audio")
    signalPower=power(signal)
    noisePower=power(noise)
    snrNow=10*np.log10(signalPower/noisePower)
    r=snr/snrNow
    x=np.sqrt(signalPower/noisePower*((noisePower/signalPower)**r))
    if maintain=="signal":
        return signal+noise*x
    if maintain=="noise":
        return signal/x+noise

class VAD:
    def __init__(self,input,energyPerLow=12.5,energyPerHigh=40,overzeroPer=45,minVoice=0.15,minSilence=0.08):
        """Speech endpoint detection based on double threshold. 

        input: An AudioFrame object. 
        energyPerLow: Low threshold of energy. [0,100]
        energyPerHigh: High threshold of energy. [0,100]
        overzeroPer: Over-zero rate threshold. [0,100]
        minVoice: A float object. The voice shorter than minVoice should be silence. 
        minSilence: A float object. The silence shorter than minSilence should be voice. 
        """
        flatten=input.samples
        energys=energyCal(flatten)
        overzeros=overzeroCal(flatten,input.sr)
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
                    if length*input.step/input.sr<minSilence:
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
                    if length*input.step/input.sr<minVoice:
                        label[startPoint:i-1]=0
                else:
                    startPoint=i
                count=0
            count+=1
        self.label=label
        self.input=input

    def plot(self,imgPath=None):
        """Visualize the result of VAD. Save the figure if imgPath is given, otherwise display directly. 

        imgPath: The direction to save the figure. 
        """
        label=self.label
        audio=self.input.retrieve()
        step=self.input.step
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

    def slices(self):
        """Return the slices of audio with labels, 0 for silence and 1 for voice. 

        return:
            A numpy array for slices of Audio. 
            A numpy array for label of each slice. 0 for silence and 1 for voice. 
        """
        label=self.label
        audio=self.input.retrieve()
        step=self.input.step
        label=label.tolist()
        slices=[]
        label2=[]
        start=0
        for i in range(0,len(label)-1):
            if (label[i+1]==0 and label[i]!=0) or (label[i+1]!=0 and label[i]==0):
                slices.append(audio.selectByTime(start*step/audio.sr,(i+1)*step/audio.sr))
                if label[i]==0:
                    label2.append(0)
                else:
                    label2.append(1)
                start=i+1
        return np.array(slices),np.array(label2)
    
    def longestVoice(self):
        """Return the longest slice of voice. 

        return: An Audio object. The longest slice of voice. 
        """
        slices,labels=self.slices()
        maxLength=0
        maxId=0
        for i in range(0,len(slices)):
            if labels[i]==1:
                if slices[i].getDuration()>maxLength:
                    maxLength=slices[i].getDuration()
                    maxId=i
        return slices[maxId]
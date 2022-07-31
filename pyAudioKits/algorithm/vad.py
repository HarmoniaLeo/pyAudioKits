from pyAudioKits.analyse.timeDomainAnalyse import energyCal, overzeroCal
import numpy as np
import matplotlib.pyplot as plt

class VAD:

    def __init__(self,input,energyThresLow,energyThresHigh,zerocrossingThres,frameDuration = 0.03,overlapRate=0.5):
        """Speech endpoint detection based on double threshold. 

        input: An Audio object. 
        energyThresLow: A lower threshold of energy for distinguish between silence and voice.
        energyThresHigh: A higher threshold of energy for distinguish between unvoiced and voiced.
        zerocrossingThres: Zero crossing rate threshold.
        frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each fram (sample points). 
        overlapRate: A float object in [0,1) for the overlapping rate of the frame.

        return: A VAD object.  
        """
        input = input.framing(frameDuration, overlapRate)
        flatten=input.samples
        energys=energyCal(flatten)
        overzeros=overzeroCal(flatten,input.sr)

        voice_begins, voice_ends = self.__distinguish(0, len(energys), energys, overzeros, True, energyThresLow, zerocrossingThres)

        labels = np.zeros_like(energys)
        for i in range(len(voice_begins)):
            labels[voice_begins[i]:voice_ends[i]] = 1

        for b, e in zip(voice_begins, voice_ends):
            voice_begins2, voice_ends2 = self.__distinguish(b, e, energys, overzeros, False, energyThresHigh, zerocrossingThres)
            for i in range(len(voice_begins2)):
                labels[voice_begins2[i]:voice_ends2[i]] = 2

        self.label=labels
        self.input=input
    
    def __distinguish(self, b, e, energys, overzeros, overzeroOver, energyThres, overzeroThres):
        clip1=energyThres
        voice_begins = []
        voice_ends = []
        if energys[b] >= clip1:
            voice_begins.append(b)
        for i in range(b+1,e):
            if (energys[i] >= clip1) and (energys[i-1] < clip1):
                voice_begins.append(i)
            elif (energys[i] <= clip1) and (energys[i-1] > clip1):
                voice_ends.append(i)
        if len(voice_begins) - 1 == len(voice_ends):
            voice_ends.append(e)
        assert len(voice_begins) == len(voice_ends)
        
        clip2=overzeroThres
        voice_begins2 = []
        voice_ends2 = []
        for voice_begin in voice_begins[::-1]:
            appended = False
            for i in range(voice_begin, b-1, -1):
                if (overzeroOver and (overzeros[i] < clip2)) or ((not overzeroOver) and (overzeros[i] >= clip2)):
                    voice_begins2.append(i+1)
                    appended = True
                    break
            if not appended:
                voice_begins2.append(b)
        voice_begins2 = voice_begins2[::-1]
        for voice_end in voice_ends:
            appended = False
            for i in range(voice_end, e):
                if (overzeroOver and (overzeros[i] < clip2)) or ((not overzeroOver) and (overzeros[i] >= clip2)):
                    voice_ends2.append(i)
                    appended = True
                    break
            if not appended:
                voice_ends2.append(e)
        assert len(voice_begins2) == len(voice_ends2)

        return voice_begins2, voice_ends2


    def plot(self,imgPath=None):
        """Visualize the result of VAD. Save the figure if imgPath is given, otherwise display directly. The blue parts are silence and the magenta parts are voice. 

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
        plt.plot(x,np.where(label2==0,y,None))
        plt.plot(x,np.where(label2==1,y,None),color="m")
        plt.plot(x,np.where(label2==2,y,None),color="orange")
        if imgPath!=None:
            plt.savefig(imgPath,dpi=500, bbox_inches = 'tight')
        else:
            plt.show()
        plt.clf()

    def slices(self):
        """Return the slices of voices in audio. 

        return: A list for Audio objects. 
        """
        label=self.label
        audio=self.input.retrieve()
        step=self.input.step
        slices=[]
        label2=[]
        start=0
        for i in range(0,len(label)-1):
            if (label[i+1]==0 and label[i]!=0) or (label[i+1]!=0 and label[i]==0):
                slices.append(audio[start*step/audio.sr:(i+1)*step/audio.sr])
                if label[i]==0:
                    label2.append(0)
                else:
                    label2.append(1)
                start=i+1
        return [s for s, l in zip(slices, label2) if l==1]
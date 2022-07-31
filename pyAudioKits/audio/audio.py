import librosa
import librosa.display
from playsound import playsound
import numpy as np
from pyAudioKits.analyse import power
import soundfile as sf
from pyAudioKits.datastructures.ds import time_axis_ds, time_frame_axis_ds

class Audio(time_axis_ds):
    def __init__(self,samples,sr):
        """Initialize an Audio object with a numpy array and a specific sample rate. 

        samples: A numpy array object. 
        sr: An int object. 

        Return: An Audio object. 
        """
        super().__init__(np.real(samples), sr, "amplitude")
    
    def save(self,direction):
        """Save the audio to specified path. 

        direction: The saving path. 
        """
        sf.write(direction,np.ascontiguousarray(np.real(self.samples)),self.sr)

    def sound(self):
        """Play the audio. 
        """
        print("The power of the audio: ",power(self))
        if (power(self)>0.003):
            print("In order to protect your hearing, the amplitude is limited. ")
            rate=power(self)/0.003
            rate=np.sqrt(rate)
            sf.write("tmp.wav",np.ascontiguousarray(np.real(self.samples)/rate),self.sr)
        else:
            sf.write("tmp.wav",np.ascontiguousarray(np.real(self.samples)),self.sr)
        playsound("tmp.wav")
    
    def __getitem__(self, index):
        return Audio(super().__getitem__(index), self.sr)
    
    def __setitem__(self, index, values):
        if type(values) == Audio:
            if self.sr != values.sr:
                raise Exception("Sample rate error.")
            values = values.samples
        if type(index) == tuple:
            raise Exception("Only support 1-D indices. ")
        else:
            self.samples[self.axis[index]] = values
    
    def getDuration(self):
        """Get the duration of the audio (seconds). 

        return: A float object. 
        """
        return self.samples.shape[0]/self.sr
    
    def __add__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples+other,self.sr)
        if other.sr!=self.sr:
            raise Exception("Sample rate error.")
        return Audio(self.samples+other.samples,self.sr)
        
    def __sub__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples-other,self.sr)
        srJudge(self, other)
        return Audio(self.samples-other.samples,self.sr)

    def __mul__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples*other,self.sr)
        srJudge(self, other)
        return Audio(self.samples*other.samples,self.sr)
        
    def __truediv__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples/other,self.sr)
        srJudge(self, other)
        return Audio(self.samples/other.samples,self.sr)

    def amplify(self,dB):
        """Amplify the audio with dB. 

        dB: A float object for the gain in dB. 
        return: An Audio object. 
        """
        amp=np.power(10,(dB/10))
        return Audio(self.samples*amp,self.sr)

    def pitch_shift(self,halfSteps):
        """Shift the pitch of the audio. A possive value of halfSteps increase the frenquency of the audio.

        halfSteps: An int object for how much halfSteps to shift. 
        return: An Audio object. 
        """
        return Audio(librosa.effects.pitch_shift(self.samples, self.sr, n_steps=halfSteps),self.sr)

    def resample(self,newRate):
        """Resample the audio with a new sample rate.

        newRate: An int object for the new sample rate.
        return: An Audio object. 
        """
        return Audio(librosa.resample(self.samples,self.sr,newRate),newRate)

    def addWgn(self,snr):
        """Add white noise of specific intensity. 

        snr: An float object for the white noise's intensity (snr).
        return: An Audio object. 
        """
        x0=self.samples
        P_signal = np.sum(abs(x0)**2)/len(x0)
        P_noise = P_signal/(10**(snr/10.0))
        x0=np.random.randn(len(x0)) * np.sqrt(P_noise)+x0
        return Audio(x0,self.sr)
    
    def framing(self,frameDuration=0.03,overlapRate=0.5,window=None):
        """Add frame to the audio. 

        frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each fram (sample points). 
        overlapRate: A float object in [0,1) for the overlapping rate of the frame. 
        window: 
            If string, it's the name of the window function (e.g., "hann")
            If tuple, it's the name of the window function and any parameters (e.g., ("kaiser", 4.0))
            If numeric, it is treated as the beta parameter of the "kaiser" window, as in scipy.signal.get_window.
            If callable, it's a function that accepts one integer argument (the window length)
            If list-like, it's a pre-computed window of the correct length Nx
        return: An AudioFrames object. 
        """
        if type(frameDuration) == float:
            windowLength = int(frameDuration*self.sr)
        else:
            windowLength = frameDuration
        step=int(windowLength*(1-overlapRate))
        y=self.samples
        l=y.shape[0]
        indexer = np.arange(windowLength)[None, :] + step*np.arange((l-windowLength)//step + 1)[:, None]
        flatten=y[indexer]
        if not (window is None):
            filters=librosa.filters.get_window(window = window, Nx = windowLength, fftbins = False)
            flatten=flatten*filters
        else:
            filters=None
        return AudioFrames(flatten,self.sr,step,filters)
    
    def timeSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting part of the audio. 
        
        start: 
            If index_type is "t/s" or "t/ms", then it will be the starting time stamp of slicing. Default = 0. 
            If index_type is "n", then it will be the starting sample count of slicing. Default = 0.
        end: 
            If index_type is "t/s" or "t/ms", then it will be the ending time stamp of slicing. Default = The last time stamp of the audio. 
            If index_type is "n", then it will be the ending sample count of slicing. Default = The total count of samples. 
        step: 
            If index_type is "t/s" or "t/ms", then it will be the time step of slicing. 
            If index_type is "n", then it will be the samples step of slicing.
            default = 1 sample. 

        index_type: "t/s", "t/ms" or "n". 

        return:
            An Audio object of part of the audio. 
        """
        return Audio(self.samples[self.axis.slicing(start, end, step, index_type)], self.sr)

    def padding(self, audioDuration, start = 0):
        """Zero-padding the audio to the given length. 
        
        audioDuration: If float, the duration of the padded audio (in seconds); if int, the number of samples of the padded audio. 
        start: The start position of the original audio in the padded audio. If float, it is the start time (in seconds); if int, it is the number of start samples.

        return:
            The zero-padded audio. 
        """
        if type(audioDuration) == float:
            audioDuration = int(audioDuration * self.sr)
        if type(start) == float:
            start = int(audioDuration * self.sr)
        zero_padding = np.zeros(audioDuration)
        zero_padding[start:start+len(self.samples)] = self.samples
        return Audio(zero_padding, self.sr)
    


class AudioFrames(time_frame_axis_ds):
    def __init__(self,flatten,sr,step,window):
        super().__init__(flatten, sr, step, "amplitude")
        self.step=step
        self.window=window

    def retrieve(self,method="first_half"):
        """Retrieve the AudioFrames object back to an Audio object. 

        method: 
            "last_half": When overlapping, preserve the last half of each frame. 
            "first_half": When overlapping, preserve the first half of each frame. 
        return: An Audio object. 
        """
        frames=self.samples
        if not (self.window is None):
            frames=frames/self.window
        if method=="first_half":
            frames1=frames[:-1,:self.step]
            samples=np.concatenate(frames1)
            return Audio(np.concatenate([samples,frames[-1]]),self.sr)
        if method=="last_half":
            frames1=frames[1:,self.step:]
            samples=np.concatenate(frames1)
            return Audio(np.concatenate([frames[0],samples]),self.sr)
    
    def frameSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting some of the frames. 
        
        start: 
            If index_type is "t/s" or "t/ms", then it will be the starting time stamp of slicing. Default = 0. 
            If index_type is "n", then it will be the starting sample count of slicing. Default = 0.
            If index_type is "frame", then it will be the starting frame count. Default = 0.
        end: 
            If index_type is "t/s" or "t/ms", then it will be the ending time stamp of slicing. Default = The last time stamp of the audio. 
            If index_type is "n", then it will be the ending sample count of slicing. Default = The total count of samples. 
            If index_type is "frame", then it will be the ending frame count of slicing. Default = The total count of frames. 
        step: 
            If index_type is "t/s" or "t/ms", then it will be the time step of slicing. 
            If index_type is "n", then it will be the samples step of slicing.
             If index_type is "frame", then it will be the frames count step of slicing. 
            Default = 1 frame. 

        index_type: "t/s", "t/ms", "n" or "frame". 

        return:
            If step > 1, then return row priority 2-D NumPy array of samples. 
            If only one frame is selected, then return an Audio object of that frame. 
            Otherwise, return an AudioFrames object. 
        """
        samples = super().frameSelect(start, end, step,index_type)
        if (not (step is None)) and (step > 1):
            return samples
        elif len(samples.shape) == 1:
            return Audio(samples, self.sr)
        elif samples.shape[0] == 1:
            return Audio(samples[0], self.sr)
        else:
            return AudioFrames(samples, self.sr, self.step, self.window)

    def __getitem__(self, index):
        samples = super().__getitem__(index)
        if type(index) == tuple:
            if type(index[1]) == slice:
                if len(samples.shape) == 1:
                    return Audio(samples, self.sr)
                elif samples.shape[0] == 1:
                    return Audio(samples[0], self.sr)
                elif (not (index[1].start is None)) or (not (index[1].stop is None)) or (not (index[1].step is None)) or ((not (index[0].step is None)) and (index[0].step > 1)):
                    return samples
                else:
                    return AudioFrames(samples, self.sr, self.step, self.window)
            else:
                return samples
        else:
            if type(index) == slice:
                if (not (index.start is None)) or (not (index.stop is None)) or (not (index.step is None)):
                    return samples
                else:
                    return AudioFrames(samples, self.sr, self.step, self.window)
            else:
                return samples
            
    def __len__(self):
        return self.step * (self.getFrameCount() - 1) + self.getFrameLength() 
    
    def getFrameCount(self):
        return self.samples.shape[0]
    
    def getFrameLength(self):
        return self.samples.shape[1]
    
    def getFrameDuration(self):
        return self.samples.shape[1]/self.sr
    
    def getStepLength(self):
        return self.step
    
    def getStepDuration(self):
        return self.step/self.sr
    
    def getDuration(self):
        return self.getStepDuration() * (self.getFrameCount() - 1) + self.getFrameDuration() 
        
def typeJudge(input,samples):
    if type(input)==Audio:
        return Audio(samples,input.sr)
    if type(input)==AudioFrames:
        return AudioFrames(samples,input.sr,input.step,input.window)

def srJudge(input1, input2):
    if input1.sr!=input2.sr:
        raise Exception("Sample rate error.")

def concatenate(lis):
    """Concatenating the Audio objects in a tuple or a list.

    lis: A tuple or a list for Audio objects. 
    return: An Audio object. 
    """
    ys=[]
    for i in lis:
        srJudge(i, lis[0])
        ys.append(i.samples)
    y=np.concatenate(ys)
    return Audio(y,lis[0].sr)

def synthesis(lis):
    """Synthesis the Audio objects in a tuple or a list.

    lis: A tuple or a list for Audio objects. 
    return: An Audio object. 
    """
    for i in lis:
        srJudge(i, lis[0])
    audio = lis[0]
    for aud in lis[1:]:
        audio = audio + aud
    return audio
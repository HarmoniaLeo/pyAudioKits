import librosa
import librosa.display
from playsound import playsound
import matplotlib.pyplot as plt
import wave
import numpy as np
import threading
from pyAudioKits.timeDomainAnalyse import power

class Audio:
    def __init__(self,samples,sr):
        """Initialize with a numpy array and specific sample rate. 

        samples: A numpy array object. 
        sr: An int object. 
        """
        self.samples=np.real(samples)
        self.sr=sr
    
    def save(self,direction):
        """Save the audio to specified path. 

        direction: The path. 
        """
        librosa.output.write_wav(direction,np.ascontiguousarray(np.real(self.samples)),self.sr)

    def sound(self):
        """Play the audio. 
        """
        print("The power of the audio: ",power(self))
        if (power(self)>0.003):
            print("In order to protect your hearing, the amplitude is limited. ")
            rate=power(self)/0.003
            rate=np.sqrt(rate)
            librosa.output.write_wav("tmp.wav",np.ascontiguousarray(np.real(self.samples)/rate),self.sr)
        else:
            librosa.output.write_wav("tmp.wav",np.ascontiguousarray(np.real(self.samples)),self.sr)
        playsound("tmp.wav")

    def selectByTime(self,startTime=None,endTime=None):
        """Capture a short piece of audio. 

        startTime: The starting time stamp(in seconds). Default = 0. 
        endTime: The ending time stamp(in seconds). Default = The last time stamp of the audio. 
        return: An Audio object. 
        """
        if startTime==None:
            start=0
        else:
            start=int(startTime*self.sr)
        if endTime==None:
            end=self.samples.shape[0]
        else:
            end=int(endTime*self.sr)
        return Audio(self.samples[start:end],self.sr)
    
    def getDuration(self):
        """Get the duration of the audio (seconds). 

        return: A float object. 
        """
        return self.samples.shape[0]/self.sr
    
    def plot(self,startTime=0,endTime=None,axes=None,imgPath=None):
        """To draw the audio waveform on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        startTime: The starting time stamp. Default = 0. 
        endTime: The ending time stamp. Default = The last time stamp of the audio. 
        axes: A matplotlib.pyplot.axes object. 
        imgPath: The path to save the graph. 
        """
        end=self.samples.shape[0]
        x=np.arange(0,end/self.sr,1/self.sr)
        y=self.samples
        if endTime is None:
            endTime=self.getDuration()
        if axes==None:
            plt.xlabel("t/s")
            plt.ylabel("amplitude")
            plt.plot(x[int(startTime*self.sr):int(endTime*self.sr)],y[int(startTime*self.sr):int(endTime*self.sr)])
            if imgPath==None:
                plt.show()
            else:
                plt.savefig(imgPath,dpi=500, bbox_inches = 'tight')
            plt.clf()
        else:
            axes.plot(x,y)
    
    def spec(self):
        """Display the spectrogram
        """
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.samples)),sr=self.sr)
    
    def __add__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples+other,self.sr)
        if other.sr!=self.sr:
            raise Exception("Sample rate error.")
        return Audio(self.samples+other.samples,self.sr)
        
    def __sub__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples-other,self.sr)
        if self.sr!=other.sr:
            raise Exception("Sample rate error.")
        return Audio(self.samples-other.samples,self.sr)

    def __mul__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples*other,self.sr)
        if other.sr!=self.sr:
            raise Exception("Sample rate error.")
        return Audio(self.samples*other.samples,self.sr)
        
    def __truediv__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples/other,self.sr)
        if other.sr!=self.sr:
            raise Exception("Sample rate error.")
        return Audio(self.samples/other.samples,self.sr)
    
    def amplify(self,dB):
        """Amplify the audio with dB. 

        dB: A float object for the gain. 
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
        """
        x0=self.samples
        P_signal = np.sum(abs(x0)**2)/len(x0)
        P_noise = P_signal/(10**(snr/10.0))
        x0=np.random.randn(len(x0)) * np.sqrt(P_noise)+x0
        return Audio(x0,self.sr)
    
    def framing(self,frameDuration=0.03,overlapRate=0.5,window=None):
        """Add frame to the audio. 

        frameDuration: A float object for the duration of each frame (seconds). 
        overlapRate: A float object in [0,1) for the overlapping rate of the frame. 
        window: The window shape. 
        return: An AudioFrames object. 
        """
        windowLength=int(frameDuration*self.sr)
        step=int(windowLength*(1-overlapRate))
        y=self.samples
        l=y.shape[0]
        indexer = np.arange(windowLength)[None, :] + step*np.arange(int((l-windowLength)/step))[:, None]
        flatten=y[indexer]
        if not (window is None):
            filters=librosa.filters.get_window(window = window, Nx = windowLength, fftbins = False)
            flatten=flatten*filters
        else:
            filters=None
        return AudioFrames(flatten.T,self.sr,step,filters)

class AudioFrames:
    def __init__(self,flatten,sr,step,window):
        self.samples=flatten
        self.sr=sr
        self.step=step
        self.window=window
    
    def retrieve(self,method="first_half"):
        """Retrieve the AudioFrames back to audio. 

        method: 
            "last_half": When overlapping, preserve the last half of each frame. 
            "first_half": When overlapping, preserve the first half of each frame. 
        return: An Audio object. 
        """
        frames=self.samples.T
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
        
def typeJudge(input,samples):
    if type(input)==Audio:
        return Audio(samples,input.sr)
    if type(input)==AudioFrames:
        return AudioFrames(samples,input.sr,input.step,input.window)

def isType(input,t="all"):
    if t=="all":
        if (type(input)!=Audio) and (type(input)!=AudioFrames):
            raise Exception("The input should be Audio or AudioFrames. ")
    elif t=="Audio":
        if type(input)!=Audio:
            raise Exception("The input should be Audio. ")
    elif t=="AudioFrames":
        if type(input)!=AudioFrames:
            raise Exception("The input should be AudioFrames. ")

def read_Audio(direction):
    """Get audio file from specified path. Dual channel audio returns a tuple. 
    
    direction: The path.
    return:
        An Audio object when reading single track sound.
        A tuple with two Audio objects when reading double tracks sound. 
    """
    y,r=librosa.load(direction)
    if len(y.shape)==2:
        return (Audio(y[0],r),Audio(y[1],r))
    elif len(y.shape)==1:
        return Audio(y,r)
    else:
        raise Exception("Wrong number of tracks. ")


def save_Audio_DoubleTracks(audio1,audio2,direction):
    """Combine two audio with the same length and sampling rate into a dual channel audio and save it to the specified path.
    
    audio1: An Audio object for the first track. 
    audio2: An Audio object for the second track. 
    direction: The path. 
    """
    if audio1.sr!=audio2.sr:
        raise Exception("Two audios should have the same sample rate. ")
    librosa.output.write_wav(direction,[np.ascontiguousarray(np.real(audio1.samples)),np.ascontiguousarray(np.real(audio2.samples))],audio1.sr)

def record(sr,RECORD_SECONDS):
    """Record audio with duration of recordSeconds from the microphone at the sample rate of sr

    sr: An int object for the sample rate. 
    RECORD_SECONDS: An int object (seconds).
    return: An Audio object. 
    """
    filepath="tmp.wav"
    def save_wave_file(pa, filename, data,sr):
        '''save the date to the wavfile'''
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        # wf.setsampwidth(sampwidth)
        wf.setsampwidth(pa.get_sample_size(paInt16))
        wf.setframerate(sr)
        wf.writeframes(b"".join(data))
        wf.close()
    isstart = str(input("Enter 1 to start recording. Enter 0 to exit. "))
    if isstart == str("1"):
        pa = PyAudio()
        stream = pa.open(format=paInt16,
                         channels=1,
                         rate=sr,
                         input=True,
                         frames_per_buffer=1024)
        print("*" * 10, "Please speak in "+str(RECORD_SECONDS)+" seconds. ")
        frames = []
        for i in range(0, int(sr / 1024 * RECORD_SECONDS)):
            data = stream.read(1024)
            frames.append(data)
        print("*" * 10, "End of recording.\n")
 
        stream.stop_stream()
        stream.close()
        pa.terminate()
 
        save_wave_file(pa, filepath, frames,sr)
    elif isstart == str("0"):
        exit()
    else:
        print("Invalid input, please select again. ")
        record(sr,RECORD_SECONDS)
    return read_Audio(filepath)

recordEnd=False

class recordThread(threading.Thread):
    def __init__(self,sr):
        threading.Thread.__init__(self)
        self.__sr=sr

    def run(self):
        global recordEnd
        filepath="tmp.wav"
        def save_wave_file(pa, filename, data,sr):
            '''save the date to the wavfile'''
            wf = wave.open(filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(paInt16))
            wf.setframerate(sr)
            wf.writeframes(b"".join(data))
            wf.close()
        pa = PyAudio()
        stream = pa.open(format=paInt16,
                         channels=1,
                         rate=self.__sr,
                         input=True,
                         frames_per_buffer=1024)
        frames = []
        while not recordEnd: 
            data = stream.read(1024)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        save_wave_file(pa, filepath, frames,self.__sr)


def start_record(sr):
    """Begin a recording thread. 

    sr: An int object for the sample rate. 
    """
    global recordEnd
    global recorder
    recordEnd=False
    recorder=recordThread(sr)
    recorder.start()

def end_record(sr):
    """End a recording thread and return an audio with the sample rate of sr. 

    sr: An int object for the sample rate. 
    return: An Audio object. 
    """
    global recordEnd
    global recorder
    recordEnd=True
    recorder.join()
    filepath="tmp.wav"
    return read_Audio(filepath)

def create_Single_Freq_Audio(amp,freq,sr,time):
    """Generate a sine wave with amplitude of amp, frequency of freq, sampling rate of sr and duration of time

    amp: A float object for amplitude. 
    freq: A float object for frequency (Hz). 
    sr: An int object for sample rate.
    time: An int object for duration (seconds).  
    return: An Audio object. 
    """
    ts=np.linsstep(0,time,num=int(sr*time))
    y=amp*np.sin(2*np.pi*freq*ts)
    return Audio(y,sr)

def concatenate(lis):
    """Generate a sine wave with amplitude of amp, frequency of freq, sampling rate of sr and duration of time

    lis: A tuple or a list for Audio objects. 
    return: An Audio object. 
    """
    sr=lis[0].getSR()
    ys=[]
    for i in lis:
        if sr!=i.getSR():
            raise Exception("Sample rate error.")
        ys.append(i.toArr())
    y=np.concatenate(ys)
    return Audio(y,sr)
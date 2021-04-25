import librosa
import librosa.display
import librosa.effects
from pyaudio import PyAudio, paInt16
from playsound import playsound
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import wave
import numpy as np
import threading


class Audio:
    def __init__(self,samples,sr):
        self.samples=np.real(samples)
        self.sr=sr
    
    def save(self,direction):
        librosa.output.write_wav(direction,np.ascontiguousarray(np.real(self.samples)),self.sr)

    def sound(self):
        librosa.output.write_wav("tmp.wav",np.ascontiguousarray(np.real(self.samples)),self.sr)
        playsound("tmp.wav")
    
    def resample(self,newRate):
        return Audio(librosa.resample(self.samples,self.sr,newRate),newRate)

    def selectByTime(self,startTime=None,endTime=None):
        if startTime==None:
            start=0
        else:
            start=int(startTime*self.sr)
        if endTime==None:
            end=self.samples.shape[0]
        else:
            end=int(endTime*self.sr)
        return Audio(self.samples[start:end],self.sr)
    
    def getLength(self):
        return self.samples.shape[0]/self.sr
    
    def plot(self,axes=None,imgPath=None):
        end=self.samples.shape[0]
        x=np.arange(0,end/self.sr,1/self.sr)
        y=self.samples
        if axes==None:
            plt.xlabel("t/s")
            plt.ylabel("amplitude")
            plt.plot(x,y)
            if imgPath==None:
                plt.show()
            else:
                plt.savefig(imgPath,dpi=500, bbox_inches = 'tight')
            plt.clf()
        else:
            axes.plot(x,y)
    
    def spec(self):
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
        if other.__sr!=other.__sr:
            raise Exception("Sample rate error.")
        return Audio(self.samples-other.samples,self.sr)

    def __mul__(self,other):
        if type(other)!=Audio:
            return Audio(self.samples*other,self.sr)
        if other.sr!=self.sr:
            raise Exception("Sample rate error.")
        return Audio(self.samples*other.samples,self.sr)
        



def read_Audio(direction):
    y,r=librosa.load(direction)
    if len(y.shape)==2:
        return (Audio(y[0],r),Audio(y[1],r))
    elif len(y.shape)==1:
        return Audio(y,r)
    else:
        raise Exception("Wrong number of tracks. ")

def save_Audio_DoubleTracks(audio1,audio2,direction):
    if audio1.sr!=audio2.sr:
        raise Exception("Two audios should have the same sample rate. ")
    librosa.output.write_wav(direction,[np.ascontiguousarray(np.real(audio1.samples)),np.ascontiguousarray(np.real(audio2.samples))],audio1.sr)

def get_audio(sr,RECORD_SECONDS):
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
    isstart = str(input("是否开始录音？ （1/0）")) #输出提示文本，input接收一个值,转为str，赋值给aa
    if isstart == str("1"):
        pa = PyAudio()
        stream = pa.open(format=paInt16,
                         channels=1,
                         rate=sr,
                         input=True,
                         frames_per_buffer=1024)
        print("*" * 10, "开始录音：请在"+str(RECORD_SECONDS)+"秒内输入语音")
        frames = []  # 定义一个列表
        for i in range(0, int(sr / 1024 * RECORD_SECONDS)):  # 循环，采样率 44100 / 1024 * 5
            data = stream.read(1024)  # 读取chunk个字节 保存到data中
            frames.append(data)  # 向列表frames中添加数据data
        print("*" * 10, "录音结束\n")
 
        stream.stop_stream()
        stream.close()  # 关闭
        pa.terminate()  # 终结
 
        save_wave_file(pa, filepath, frames,sr)
    elif isstart == str("0"):
        exit()
    else:
        print("无效输入，请重新选择")
        get_audio(sr,RECORD_SECONDS)
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
            # wf.setsampwidth(sampwidth)
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
        while not recordEnd:  # 循环，采样率 44100 / 1024 * 5
            data = stream.read(1024)  # 读取chunk个字节 保存到data中
            frames.append(data)  # 向列表frames中添加数据data
        stream.stop_stream()
        stream.close()  # 关闭
        pa.terminate()  # 终结
        save_wave_file(pa, filepath, frames,self.__sr)


def start_record(sr):
    global recordEnd
    global recorder
    recordEnd=False
    recorder=recordThread(sr)
    recorder.start()

def end_record(sr):
    global recordEnd
    global recorder
    recordEnd=True
    recorder.join()
    filepath="tmp.wav"
    return read_Audio(filepath)

def create_Single_Freq_Audio(amp,freq,sr,time):
    ts=np.linsstep(0,time,num=int(sr*time))
    y=amp*np.sin(2*np.pi*freq*ts)
    return Audio(y,sr)

def concatenate(lis):
    sr=lis[0].getSR()
    ys=[]
    for i in lis:
        if sr!=i.getSR():
            raise Exception("Sample rate error.")
        ys.append(i.toArr())
    y=np.concatenate(ys)
    return Audio(y,sr)
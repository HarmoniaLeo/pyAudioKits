from pyaudio import PyAudio, paInt16
import wave
import numpy as np
from pyAudioKits.audio.io import read_Audio
import os
import threading

def record(sr,RECORD_SECONDS,channels=1):
    """Record audio with duration of recordSeconds from the microphone at the sample rate of sr

    sr: An int object for the sample rate. 
    RECORD_SECONDS: An int object (seconds).
    return: An Audio object. 
    """
    filepath="tmp.wav"
    isstart = str(input("Enter 1 to start recording. Enter 0 to exit. "))
    if isstart == str("1"):
        pa = PyAudio()
        FORMAT = paInt16
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(sr)
        stream = pa.open(format=paInt16,
                         channels=channels,
                         rate=sr,
                         input=True,
                         frames_per_buffer=1024)
        print("*" * 10, "Please speak in "+str(RECORD_SECONDS)+" seconds. ")
        for i in range(0, int(sr / 1024 * RECORD_SECONDS)):
            data = stream.read(1024)
            wf.writeframes(data)
        print("*" * 10, "End of recording.\n")
 
        stream.stop_stream()
        stream.close()
        pa.terminate()
        wf.close()
    elif isstart == str("0"):
        exit()
    else:
        print("Invalid input, please select again. ")
        record(sr,RECORD_SECONDS)
    audio = read_Audio(filepath)
    total_len = int(RECORD_SECONDS*sr)
    if type(audio)==tuple:
        audio[0].samples = np.concatenate([audio[0].samples,np.zeros(total_len-len(audio[0].samples))])
        audio[1].samples = np.concatenate([audio[1].samples,np.zeros(total_len-len(audio[1].samples))])
    else:
        audio.samples = np.concatenate([audio.samples,np.zeros(total_len-len(audio.samples))])
    os.remove(filepath)
    return audio

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
    """Beginning a recording thread. 

    sr: An int object for the sample rate. 
    """
    global recordEnd
    global recorder
    recordEnd=False
    recorder=recordThread(sr)
    recorder.start()

def end_record(sr):
    """Ending a recording thread and return an audio with the sample rate of sr. 

    sr: An int object for the sample rate. 
    return: An Audio object. 
    """
    global recordEnd
    global recorder
    recordEnd=True
    recorder.join()
    filepath="tmp.wav"
    audio = read_Audio(filepath)
    os.remove(filepath)
    return audio
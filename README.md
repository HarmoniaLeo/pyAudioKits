# pyAudioKits
Powerful Python audio workflow support based on librosa and other libraries

```
!python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps pyAudioKits
```

## Basic usage

```python
import pyAudioKits.audio as ak
```

### Create or load an audio

```python
audio=ak.Audio(array,sr)	#Initialize with a numpy array and specific sample rate

audio=ak.read_Audio("audio.wav")	#Get audio file from specified path

(track1,track2)=ak.read_Audio("audio.wav")	#Dual channel audio returns a tuple

audio=ak.record(sr,recordSeconds)	#Record audio with duration of recordSeconds from the microphone at the sample rate of sr

ak.audio.start_record(sr)	#Begin a recording thread. 

audio=ak.end_record(sr)	#End a recording thread and return an audio with the sample rate of sr. The sr should be equal to the value when you began a recording thread. 

audio=ak.create_Single_Freq_Audio(amp,freq,sr,time)	#Generate a sine wave with amplitude of amp, frequency of freq, sampling rate of sr and duration of time (seconds).
```

### Manipulate

```python
audio.sound()	#Play the audio

audio.save("audio.wav")	#Save the audio to specified path

ak.save_Audio_DoubleTracks(track1,track2,"audio.wav")	#Combine two audio with the same length and sampling rate into a dual channel audio and save it to the specified path

audio=ak.concatenate([audio1,audio2,...])	#Connect multiple audio with equal sampling rate

audioSelected=audio.selectByTime(startTime,endTime)	#Capture a short piece of audio. Defaultly, startTime = 0 and endTime = the end. 
```

### Interact

```python
duration=audio.getDuration()	#Get the duration of the audio (seconds)

sample_rate=audio.sr	#Get the sample rate

samples=audio.samples	#Get the samples as numpy array
```

### Display

```python
audio.plot(axes,imgPath=None)	#To draw the audio waveform on the sub graph. If no subgraph is passed in, it will be displayed directly. If imgpath is passed in, the graph will be saved

audio.plot(startTime,endTime)	#The start time stamp and the end time stamp can also be specified. Defaultly, startTime = 0 and endTime = the end. 

audio.spec()	#Display the spectrogram
```

### Modify

```python
audio1=audio1+audio2	#Audio overlay. The two audio should have the same length and sample rate. 
audio1=audio1+arr	#Audio can also overlay a numpy array, whose shape is the same as audio1.samples.shape
audio1=audio1-audio2	#Audio substract. The two audio should have the same length and sample rate. 
audio1=audio1-arr	#Audio can also substract a numpy array, whose shape is the same as audio1.samples.shape
audio1=audio1*audio2	#Audio modulate. The two audio should have the same length and sample rate. 
audio1=audio1*arr	#Audio can also be modulated by a numpy array, whose shape is the same as audio1.samples.shape
audio1=audio1/audio2	#Audio demodulate. The two audio should have the same length and sample rate. 
audio1=audio1/arr	#Audio can also be demodulated by a numpy array, whose shape is the same as audio1.samples.shape

audio1=audio1*value	#Using multiplication to amplify an audio

audio1=audio1/value	#Using divide to attenuate an audio

audio1=audio1.amplify(dB)	#Amplify the audio with dB

audio1=audio1.pitch_shift(halfSteps)	#Shift the pitch of the audio. A possive value of halfSteps increases the frenquency of the audio, while a possive value decreases. 

audio1=audio1.resample(newRate)	#Resample the audio with a new sample rate

audio1=audio1.addWgn(dB)	#Add white noise of specific intensity
```

### Framing and retrieving

```python
audioFrames=audio.framing(frameDuration,overlapRate,"hamming")	#Add frame to the audio and return an AudioFrames object. Duration (seconds), overlapping rate and window shape of each frame can be specified. 

audio=audioFrames.retrieve("last_half")	#Retrieving the audio frames back to audio. "last_half" or "first_half" to preserve when overlapping. 
```

## Analyse

Unless otherwise specified, the input of the analyse functions can accept both the Audio object and the AudioFrames object.

When accepting an Audio object, returns the statistics of the Audio object or an another Audio object. 

When an AudioFrames object is accepted, it returns the statistical results of each frame in the AudioFrames object (1-D NumPy array or column priority 2-D NumPy array unless otherwise specified) or an another AudioFrames object. 

### Time domain analyse

```python
import pyAudioKits.timeDomainAnalyse as tda

power=tda.power(input)	#Calculate the power

log_energy=tda.logEnergy(input)	#Calculate the log energy

overzeros=tda.overzero(input)	#Calculate the over-zero rate. The input can only be an AudioFrames object

corr=tda.correlate(input1,input2)
#When input1 and input2 are both Audio objects, calculating the correlation between input1 and input2, get a numpy array for correlation result with length of (len(audio1.samples)+len(audio2.samples)-1)
#When input1 and input2 are both AudioFrames objects, calculating the short time correlation of each frame between input1 and input2

snr=tda.snr(signal,signalAndNoise)	#Calculate the SNR(dB). Signal is an Audio object for the ground truth of the signal. SignalAndNoise is an Audio object for the signal mixed with noise. 

signalAndNoise=tda.mixWithSNR(signal,noise,snr,maintain="signal")	#Mixing signal and noise with specific SNR(dB). maintain="signal" to maintain the intensity of signal while maintain="noise" to maintain the intensity of noise. 

vad=tda.VAD(input,energyPerLow,energyPerHigh,overzeroPer,minVoice,minSilence)	#Voice activity detection based on double threshold. Return a VAD object

label=vad.label	#Get the label of each frame. 0 is silence and 1 is voice

vad.plot(imgPath)	#Visualize the result of VAD. Save the figure if imgPath is given, otherwise display directly

slices,labels=vad.slices()	#Return the slices of audio with labels, 0 for silence and 1 for voice.)

voice=vad.longestVoice()	#Return the longest slice of voice
```

### Frequency domain analyse

```python
import pyAudioKits.freqDomainAnalyse as fda

spectral_entropy=fda.specEntropy(input)	#Calculate the spectral entropy

spectral_power_density=fda.PSD(input,N=1024)	#Calculate the power spectral density. N is the num of Fourier Transform points. None if use all points. 

freq=fda.FFT(input,N)	#Doing N-points FFT (for Audio) or STFT (for AudioFrames) to the input. Return an AudioFrequency object. N = None when using all points of the audio or each frame. 

freq.plot(axes,imgPath=None)	#To draw the spectrum on the sub graph. If no subgraph is passed in, it will be displayed directly. If imgpath is passed in, the graph will be saved

freq.plot(startTime,endTime)	#The min frequency point and the max frequency point can also be specified. Defaultly, minf = 0 and maxf = Half of the sample rate

freq.plot()	#If the AudioFrequency object was derived from an AudioFrame object, then display the spectrogram

output=freq.IFFT()	#Transform from frequency domain to time domain, return an Audio object or an AudioFrames object

freq,peak=freq.getMaxFrequency()	#Get the frequency and the amplitude of spectrum peak
```

### Model based analyse

```python
import pyAudioKits.modelBasedAnalyse as mba

es,ws=mba.LPC(audio,p=10)	#Matlab style LPC for each frame. Return a list of LPC error of each frame and a list of LPC coefficient of each frame. 

features=mba.MFCC(audio,p=13,frameDuration=0.03,overlapRate=0.5)	#Calculate the MFCC features of the Audio. The function itself does the framing. Return a 2-D NumPy array of MFCC features, each row will be MFCC features of one frame. The number of features of each frame will be 3*p. 
```

### Filtering

```python
import pyAudioKits.filters as flt

output=flt.lowPassFilterN(input,n,f,analog=False)	#N order low pass Butterworth filter. 

output=flt.highPassFilterN(input,n,f,analog=False)	#N order high pass Butterworth filter.

output=flt.bandPassFilterN(input,n,fLow,fHigh,analog=False)	#N order band pass Butterworth filter.

output=flt.bandStopFilterN(input,n,fLow,fHigh,analog=False)	#N order band stop Butterworth filter.

output=flt.lowPassFilter(input,fPass,fStop,ripplePass,rippleStop,analog=False)	#Low pass Butterworth filter with specified ripple. 

output=flt.highPassFilter(input,fPass,fStop,ripplePass,rippleStop,analog=False)	#High pass Butterworth filter with specified ripple. 

output=flt.bandPassFilter(input,fLowPass,fLowStop,fHighPass,fHighStop,ripplePass,rippleStop,analog=False)	#Band pass Butterworth filter with specified ripple. 

output=flt.bandStopFilter(input,fLowPass,fLowStop,fHighPass,fHighStop,ripplePass,rippleStop,analog=False)	#Band stop Butterworth filter with specified ripple. 

output=flt.specSubstract(input,noise,Thres=3,beta=0.002,G=0.9)	#Using spectral subtraction to reduce noise. 

wiener=flt.wiener(h_length=200,desired_signal,observed_signal)	#Using a signal without noise and the same signal with noise to train a wiener filter. 

output=wiener.apply(input)	#Using pre-trained Wiener filter to reduce noise. 

output=flt.kalmanFilter(input,noise,numIter=7,p=20)	#Using Kalman filter to reduce noise. 
```




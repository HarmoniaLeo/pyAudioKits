- [pyAudioKits](#pyaudiokits)
  - [Basic Usage](#basic-usage)
    - [Create or load an audio](#create-or-load-an-audio)
      - [From NumPy array](#from-numpy-array)
      - [From File](#from-file)
      - [Recording](#recording)
      - [Simulating](#simulating)
    - [The Audio object](#the-audio-object)
      - [Play](#play)
      - [Plot](#plot)
      - [Get Properties](#get-properties)
      - [Save](#save)
      - [Indexing and Slicing](#indexing-and-slicing)
      - [Concatenate](#concatenate)
      - [Synthesis](#synthesis)
      - [Arithmetic operations](#arithmetic-operations)
      - [Amplify](#amplify)
      - [Pitch shift](#pitch-shift)
      - [Resample](#resample)
      - [Add Gaussian White Noise](#add-gaussian-white-noise)
      - [Padding](#padding)
      - [Framing and Windowing](#framing-and-windowing)
    - [The AudioFrames object](#the-audioframes-object)
      - [Get Properties](#get-properties-1)
      - [Indexing and Slicing](#indexing-and-slicing-1)
      - [Retrieve](#retrieve)
  - [Time domain analyse](#time-domain-analyse)
    - [Methods](#methods)
      - [Power and Energy](#power-and-energy)
      - [Zero Crossing Rate](#zero-crossing-rate)
      - [Auto-correlation](#auto-correlation)
    - [Statistic result data structures](#statistic-result-data-structures)
      - [frame_axis_ds object](#frame_axis_ds-object)
      - [time_delta_axis_ds object](#time_delta_axis_ds-object)
      - [time_delta_frame_axis_ds object](#time_delta_frame_axis_ds-object)
  - [Frequency domain analyse](#frequency-domain-analyse)
    - [Methods](#methods-1)
      - [FFT](#fft)
      - [Power Spectral Density](#power-spectral-density)
      - [Spectral Entropy](#spectral-entropy)
      - [Spectrum Peak](#spectrum-peak)
    - [Statistic result data structures](#statistic-result-data-structures-1)
      - [freq_axis_ds object](#freq_axis_ds-object)
      - [freq_frame_axis_ds object](#freq_frame_axis_ds-object)
  - [Model based analyse](#model-based-analyse)
    - [MFCC](#mfcc)
    - [FBank](#fbank)
    - [Mel Spectrogram](#mel-spectrogram)
    - [LPC](#lpc)
  - [Filter](#filter)
    - [General Design](#general-design)
    - [Butterworth Filter Design](#butterworth-filter-design)
  - [Algorithm](#algorithm)
    - [Speech Endpoint Detection](#speech-endpoint-detection)
    - [Speech Enhancement](#speech-enhancement)
      - [Spectrum subtraction](#spectrum-subtraction)
      - [Wiener Filtering](#wiener-filtering)
      - [Kalman Filtering](#kalman-filtering)
    - [Speech Recognition](#speech-recognition)
      - [DTW](#dtw)
      - [GMM+HMM](#gmmhmm)

# pyAudioKits

Powerful Python audio workflow support based on librosa and other libraries

```
pip install pyAudioKits
```

This gives you access to the vast majority of pyAudioKits' features. If you want to use pyAudioKits' recording API, please refer to the section [Recording](#Recording). 

## Basic Usage

```python
import pyAudioKits.audio as ak
```

### Create or load an audio

#### From NumPy array

```python
audio = ak.Audio(samples, sr)
```

Initialize an **Audio object** with a numpy array and a specific sample rate. 

* samples: The audio samples. A numpy array object. 

* sr: The sample rate. An int object. 

Return: An **Audio object**. 

#### From File

```python
audio = ak.read_Audio(direction = "audio.wav")
(audio1,audio2) = ak.read_Audio(direction = "audio_dualchannel.wav")
```

Get audio file from specified path. Dual channel audio returns a tuple.  

* direction: The path.

Return: 

* An **Audio object** when reading single channel audio.
* A tuple with two **Audio objects** when reading double channels audio. 

#### Recording

The pyAudioKits recording feature requires the PyAudio library, which is currently not properly installed online via pip.

In order to install PyAudio, please follow the following procedure.

1. Check your own Python version

2. Download the corresponding .whl file at [this link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

3. Put the downloaded .whl file into the Scripts folder under the python installation path (if using Python directly) or the Scripts folder under the Anaconda installation path (if using Anaconda)

4. Open the command line and enter the command

   ```shell
   pip install .whl file path + .whl file name
   #e.g. pip install D:\anacondaLocation\Scripts\PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
   ```

5. Continue entering command

   ```python
   pip install pyaudio
   ```

```python
import pyAudioKits.record
```

```python
audio = pyAudioKits.record.record(sr,recordSeconds)
(audio1, audio2) = pyAudioKits.record.record(sr,recordSeconds,channels=2)
```

Record audio with duration of recordSeconds from the microphone at the sample rate of sr. 

* sr: An int object for the sample rate. 
* recordSeconds: An int object (seconds). 
* channels: Number of channels for recording. 

Return: 

* An **Audio object** when recording single channel audio.
* A tuple with two **Audio objects** when recording double channels audio. 

```python
pyAudioKits.record.start_record(sr)
audio = pyAudioKits.record.end_record(sr)
```

Beginning and ending a recording thread. Can be used in interactive software designs where recording needs to be turned on or off at any time. 

* sr: An int object for the sample rate. 

Return: An **Audio object**. 

#### Simulating

```python
audio = ak.create_Single_Freq_Audio(amp,freq,sr,time,phase)
```

Generate a sine wave signal. 

* amp: A float object for amplitude. 
* freq: A float object for frequency (Hz). 
* sr: An int object for sample rate.
* time: An int object for duration (seconds).  
* phase: The phase of the sine wave. Specifying None will generate independently and uniformly distributed random phase between [-π,π] each moment.

Return: An **Audio object**. 

### The Audio object

#### Play

```python
audio.sound()
```

Play the audio. It displays the power of the audio and automatically limits the amplitude when it may damage the hearing of the human ear. 

#### Plot

```python
audio.plot(start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="t/s")
```

To draw the audio waveform on the sub graph. 

If no subgraph is passed in, it will be displayed directly. 

If imgpath is passed in, the graph will be saved. 

* start: 
  * If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
  * If xlabel is "n", then it will be the starting sample count. Default = 0.

* end: 
  * If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp of the audio. 
  * If xlabel is "n", then it will be the ending sample count. Default = The total count of samples. 

* ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 

* ax: A matplotlib.pyplot subplot to draw on. 

* imgPath: The path to save the graph. 

* xlabel: "t/s", "t/ms" or "n". 

#### Get Properties

```python
duration = audio.getDuration()
```

Get the duration of the audio (seconds). 

Return: A float object. 

```python
samples_count = len(audio)
```

Get the samples count of the audio. 

Return: An int object. 

```python
sr = audio.sr
```

Get the sample rate of the audio. 

Return: An int object. 

#### Save

```python
audio.save(direction = "audio.wav")
```

Save the audio to specified path. 

* direction: The saving path. 

```python
ak.save_Audio_DoubleTracks(audio1 = track1, audio1 = track2, direction = "audio.wav")
```

Combine two audio with the same length and sampling rate into a dual channel audio and save it to the specified path.

* audio1: An **Audio object** for the first channel. 
* audio2: An **Audio object** for the second channel. 
* direction: The saving path. 

#### Indexing and Slicing

Audio object supports **one-dimensional indexes and slices**. Any value used in the index represents the number of samples if it is an integer, or the time in seconds if it is a floating point number. 

Return: 

* If indexing is performed, a sample value is returned. 
* If slicing is performed, an **Audio object** consisting of the sliced samples is returned, but the sample rate is not changed.

Since the 1D index is overloaded, it is also possible **to modify a specific part of the audio using the 1D index**. It is possible to use a NumPy array as a source to modify, or another **Audio object** as a source. However, the sample rate of the source and target audio must be the same.

```python
audio_slice = audio.timeSelect(start = 0, end = None, step = None, index_type = "t/s")
```

Selecting part of the audio. 

* start: 
  * If index_type is "t/s" or "t/ms", then it will be the starting time stamp of slicing. 
  * If index_type is "n", then it will be the starting sample count of slicing. 
  * Default = 0.

* end: 
  * If index_type is "t/s" or "t/ms", then it will be the ending time stamp of slicing. Default = The last time stamp of the audio. 
  * If index_type is "n", then it will be the ending sample count of slicing. Default = The total count of samples. 

* step: 
  * If index_type is "t/s" or "t/ms", then it will be the time step of slicing. 
  * If index_type is "n", then it will be the samples step of slicing.
  * default = 1 sample. 

* index_type: "t/s", "t/ms" or "n". 

Return: An **Audio object** of part of the audio. 

#### Concatenate

```python
audio = ak.concatenate(lis = [audio1,audio2,...])
```

Concatenating the **Audio objects** in a tuple or a list.

* lis: A tuple or a list for **Audio objects**. 

Return: An **Audio object**. 

#### Synthesis

```python
audio = ak.synthesis(lis = [audio1,audio2,...])
```

Synthesis the **Audio objects** in a tuple or a list.

* lis: A tuple or a list for **Audio objects**. 

Return: An **Audio object**. 

```python
audio = ak.mixWithSNR(signal, noise, snr, maintain= "signal")
```

Mixing signal and noise with specific SNR(dB). The signal and noise should have the same sample rate. 

* signal: An **Audio object**. The signal without noise. 

* noise: An **Audio object**. Noise to mix with signal. 

* snr: A float object. The SNR(dB) between signal and noise. 

* maintain: maintain="signal" to maintain the intensity of signal while maintain="noise" to maintain the intensity of noise. 

Return: An **Audio object**. 

#### Arithmetic operations

Audio object overloads the arithmetic operations. 

```python
audio1=audio1+audio2	#Audio overlay. The two audio should have the same length and sample rate. 
audio1=audio1+arr	#Audio can also overlay a numpy array, whose shape is the same as audio1.samples.shape
audio1=audio1-audio2	#Audio substract. The two audio should have the same length and sample rate. 
audio1=audio1-arr	#Audio can also substract a numpy array, whose shape is the same as audio1.samples.shape
audio1=audio1*audio2	#Audio modulate. The two audio should have the same length and sample rate. 
audio1=audio1*arr	#Audio can also be modulated by a numpy array, whose shape is the same as audio1.samples.shape
audio1=audio1/audio2	#Audio demodulate. The two audio should have the same length and sample rate. 
audio1=audio1/arr	#Audio can also be demodulated by a numpy array, whose shape is the same as audio1.samples.shape

audio1=audio1*value	#Using multiplication to amplify an audio, value is float object. 

audio1=audio1/value	#Using divide to attenuate an audio, value is float object. 
```

#### Amplify

```python
audio = audio.amplify(dB)
```

Amplify the audio with gain of dB. 

* dB: A float object for the gain in dB. 

Return: An **Audio object**. 

#### Pitch shift

```python
audio = audio.pitch_shift(halfSteps) 
```

Shift the pitch of the audio. A possive value of halfSteps increase the frenquency of the audio.

* halfSteps: An int object for how much halfSteps to shift. 

Return: An **Audio object**. 

#### Resample

```python
audio = audio.resample(newRate)
```

Resample the audio with a new sample rate. 

* newRate: An int object for the new sample rate.

Return: An **Audio object**. 

#### Add Gaussian White Noise

```python
audio = audio.addWgn(dB)
```

Add Gaussian white noise of specific intensity. 

* snr: An float object for the white noise's intensity. The signal and white noise will have a signal-to-noise ratio measured in snr (in dB)

Return: An **Audio object**. 

#### Padding

```python
audio_padded = audio.padding(audioDuration, start = 0)
```

Zero-padding the audio to the given length. 

* audioDuration: If float, the duration of the padded audio (in seconds); if int, the number of samples of the padded audio. 

* start: The start position of the original audio in the padded audio. If float, it is the start time (in seconds); if int, it is the number of start samples.

Return: An **Audio object**. 

#### Framing and Windowing

```python
audioFrames=audio.framing(frameDuration=0.03,overlapRate=0.5,window=None)
```

Framing and windowing the audio. 

* frameDuration: A float object for the duration of each frame (seconds) or an int object for the length of each frame (sample points). 

* overlapRate: A float object in [0,1) for the overlapping rate of the frames. 

* window: 
  * If string, it's the name of the window function (e.g., "hann")
  * If tuple, it's the name of the window function and any parameters (e.g., ("kaiser", 4.0))
  * If numeric, it is treated as the beta parameter of the "kaiser" window, as in scipy.signal.get_window.
  * If callable, it's a function that accepts one integer argument (the window length)
  * If list-like, it's a pre-computed window of the correct length Nx

Return: An **AudioFrames object**. 

### The AudioFrames object

#### Get Properties

```python
duration = audioFrames.getDuration()
```

Get the duration of the original audio (seconds). 

Return: A float object. 

```python
samples_count = len(audioFrames)
```

Get the samples count of the original audio. 

Return: An int object. 

```python
sr = audioFrames.sr
```

Get the sample rate of the original audio. 

Return: An int object. 

```python
frame_count = audioFrames.getFrameCount()
```

Get the count of frames. 

Return: An int object. 

```python
frame_length = audioFrames.getFrameLength()
```

Get the count of samples in each frame. 

Return: An int object. 

```python
frame_count, frame_length = audioFrames.shape
```

Get the count of frames and the count of samples in each frame at the same time. 

Return: A tuple of two int objects. 

```python
frame_duration = audioFrames.getFrameDuration()
```

Get the duration of each frame in seconds. 

Return: A float object. 

```python
step_length = audioFrames.getStepLength()
```

Get the inter-frame step length in count of samples. 

Return: A int object. 

```python
step_duration = audioFrames.getStepDuration()
```

Get the inter-frame step length in seconds. 

Return: A float object. 

#### Indexing and Slicing

AudioFrames object supports **one or two-dimensional indexes and slices**. 

The first dimension is the **frame dimension**. Any value represents the number of samples in the original **Audio object** if it is an integer, or the time in the original **Audio object** in seconds if it is a floating point number. 

The second index is the **time dimension**. Any value represents the number of samples in each frame if it is an integer, or the time in each frame in seconds if it is a floating point number. 

Return: 

* If any indexing or slicing is done on the time dimension, a 2-D NumPy array will be returned. 
* If slicing on the frame dimension with step greater than the inter-frame step length, a 2-D NumPy array will be returned. 
* Otherwise, an **AudioFrames object** will be returned; in particular, if the frame dimension of the **AudioFrames object** is 1, it will be downscaled to an **Audio object**.

#### Retrieve

```python
audio = audioframes.retrieve(method = "first_half")
```

Retrieve the **AudioFrames** object back to an **Audio object**. 

* method: 
  * "last_half": When overlapping, preserve the last half of each frame. 
  * "first_half": When overlapping, preserve the first half of each frame. 

Return: An **Audio object**. 

## Time domain analyse

**For more details of this section, please refer to:** [**2. Musical tone and Noise.ipynb**](https://github.com/HarmoniaLeo/pyAudioKits/blob/main/2.%20Musical%20tone%20and%20Noise.ipynb) **and** [**4. Short-time Analysis Method for Audio Signals.ipynb**](https://github.com/HarmoniaLeo/pyAudioKits/blob/main/4.%20Short-time%20Analysis%20Method%20for%20Audio%20Signals.ipynb) 

```python
import pyAudioKits.analyse as aly
```

### Methods

#### Power and Energy

```python
power = aly.power(input, dB = False)
```

Calculate the power of the audio. 

* input: An **Audio object** or an **AudioFrames object**. 

* dB: Whether to express the result in the form of gain (dB). 

Return: 

* If input is an **Audio object**: Power of the whole audio. 

* If input is an **AudioFrames object**: A **frame_axis_ds object** containing power of each frame. 

```python
energy = aly.energy(input, dB = False)
```

Calculate the energy of the audio. 

* input: An **Audio object** or an **AudioFrames object**. 
* dB: Whether to express the result in the form of gain (dB). 

Return: 

* If input is an **Audio object**: Energy of the whole audio. 
* If input is an **AudioFrames object**: A **frame_axis_ds object** containing energy of each frame. 

```python
snr = aly.snr(signal, signalAndNoise)
```

Calculate the SNR(dB). 

* signal: An **Audio object**. The ground truth of the signal. 
* signalAndNoise: An **Audio object**. Signal mixed with noise. 

Return: A float object of the SNR(dB). 

#### Zero Crossing Rate

```python
zerocrossingrate = aly.zerocrossing(input)
```

Calculate the zero crossing rate of the audio. 

* input: An **AudioFrames object**. 

Return: A **frame_axis_ds object** containing zero crossing rate of each frame. 

#### Auto-correlation

```python
autocorrelation = aly.autocorr(input)
```

Calculate the auto correlation function of the audio. 

* input: An **Audio object** or an **AudioFrames object**. 

Return: 

* If input is an **Audio object**: A **time_delta_axis_ds object** containing auto correlation function.

* If input is an **AudioFrames object**: A **time_delta_frame_axis_ds object** containing short-time auto correlation result for each frames.

### Statistic result data structures

#### frame_axis_ds object

```python
result = frame_axis_ds.samples
```

Get statistics. 

Return: A one-dimensional NumPy array whose length is equal to the number of frames of the **AudioFrames object** to statistic. 

frame_axis_ds object supports **one-dimensional indexes and slices**. Any value used in the index represents the number of samples if it is an integer, or the time in seconds if it is a floating point number. 

Return: An 1-D NumPy array for part of the result. 

```python
result = frame_axis_ds.frameSelect(start = 0, end = None, step = None, index_type = "t/s")
```

Selecting part of the result. 

* start: 
  * If index_type is "t/s" or "t/ms", then it will be the starting time stamp of slicing. Default = 0. 
  * If index_type is "n", then it will be the starting sample count of slicing. Default = 0.
  * If index_type is "frame", then it will be the starting frame count. Default = 0.

* end: 
  * If index_type is "t/s" or "t/ms", then it will be the ending time stamp of slicing. Default = The last time stamp of the audio. 
  * If index_type is "n", then it will be the ending sample count of slicing. Default = The total count of samples. 
  * If index_type is "frame", then it will be the ending frame count of slicing. Default = The total count of frames. 

* step: 
  * If index_type is "t/s" or "t/ms", then it will be the time step of slicing. 
  * If index_type is "n", then it will be the samples step of slicing.
  * If index_type is "frame", then it will be the frames count step of slicing. 
  * Default = 1 frame. 

* index_type: "t/s", "t/ms", "n" or "frame". 

Return: An 1-D NumPy array for part of the result. 

```python
frame_axis_ds.plot(start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="t/s")
```

To draw the per frame statistic result on the sub graph. 

If no subgraph is passed in, it will be displayed directly. 

If imgpath is passed in, the graph will be saved. 

* start: 
  * If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
  * If xlabel is "n", then it will be the starting sample count. Default = 0.
  * If xlabel is "frame", then it will be the starting frame count. Default = 0.

* end: 
  * If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp of the audio. 
  * If xlabel is "n", then it will be the ending sample count. Default = The total count of samples. 
  * If xlabel is "frame", then it will be the ending frame count. Default = The total count of frames. 

* ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 
* ax: A matplotlib.pyplot subplot to draw on. 
* imgPath: The path to save the graph. 
* xlabel: "t/s", "t/ms", "n" or "frame". 

#### time_delta_axis_ds object

```python
result = time_delta_axis_ds.samples
```

Get statistics. 

Return: A one-dimensional NumPy array whose length is equal to $2\times L-1$ , where $L$ is the number of samples of the **Audio object** to statistic. 

time_delta_axis_ds object supports **one-dimensional indexes and slices**. Any value used in the index represents the sample offset if it is an integer, or the time offset in seconds if it is a floating point number. 

Return: An 1-D NumPy array for part of the result. 

```python
result = time_delta_axis_ds.timeSelect(start = 0, end = None, step = None, index_type = "t/s")
```

Selecting part of the result. 

* start: 
  * If index_type is "t/s" or "t/ms", then it will be the starting time offset of slicing. Default = 0. 
  * If index_type is "k", then it will be the starting sample offset of slicing. Default = 0.
* end: 
  * If index_type is "t/s" or "t/ms", then it will be the ending time offset of slicing. Default = The duration of the audio. 
  * If index_type is "k", then it will be the ending sample offset of slicing. Default = The max samples count of the audio. 
* step: 
  * If index_type is "t/s" or "t/ms", then it will be the time offset step of slicing. 
  * If index_type is "k", then it will be the sample offset step of slicing.
  * default = 1 sample offset. 
* index_type: "t/s", "t/ms" or "k". 

Return: An **Audio object** of part of the audio. 

```python
result = time_delta_axis_ds.plot(start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="t/s")
```

To draw the result on the sub graph. 

If no subgraph is passed in, it will be displayed directly. 

If imgpath is passed in, the graph will be saved. 

* start: 
  * If xlabel is "t/s" or "t/ms", then it will be the starting time offset. Default = 0. 
  * If xlabel is "k", then it will be the starting sample offset. Default = 0.

* end: 
  * If xlabel is "t/s" or "t/ms", then it will be the ending time offset. Default = The duration of the audio. 
  * If xlabel is "k", then it will be the ending sample offset. Default = The max samples count of the audio. 

* ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 

* ax: A matplotlib.pyplot subplot to draw on. 

* imgPath: The path to save the graph. 

* xlabel: "t/s", "t/ms" or "k". 

#### time_delta_frame_axis_ds object

```python
result = time_delta_frame_axis_ds.samples
```

Get statistics. 

Return: A two-dimensional NumPy array whose shape is $(K,2\times L-1)$ , where $L$ is the number of samples in each frame and $K$ is the number of frames of the **AudioFrames object** to statistic. 

time_delta_frame_axis_ds object supports **one or two-dimensional indexes and slices**. 

The first dimension is the **frame dimension**. Any value represents the number of samples in the original **Audio object** if it is an integer, or the time in the original **Audio object** in seconds if it is a floating point number. 

The second dimension is the **time offset dimension**. Any value represents the samples offset in each frame if it is an integer, or the time offset in each frame in seconds if it is a floating point number. 

Return: An 2-D NumPy array for part of the result. 

```python
result = time_delta_frame_axis_ds.frameSelect(start = 0, end = None, step = None, index_type = "t/s")
```

Selecting part of the result on the frame dimension. 

* start: 
  * If index_type is "t/s" or "t/ms", then it will be the starting time stamp of slicing. Default = 0. 
  * If index_type is "n", then it will be the starting sample count of slicing. Default = 0.
  * If index_type is "frame", then it will be the starting frame count. Default = 0.

* end: 
  * If index_type is "t/s" or "t/ms", then it will be the ending time stamp of slicing. Default = The last time stamp of the audio. 
  * If index_type is "n", then it will be the ending sample count of slicing. Default = The total count of samples. 
  * If index_type is "frame", then it will be the ending frame count of slicing. Default = The total count of frames. 

* step: 
  * If index_type is "t/s" or "t/ms", then it will be the time step of slicing. 
  * If index_type is "n", then it will be the samples step of slicing.
  * If index_type is "frame", then it will be the frames count step of slicing. 
  * Default = 1 frame. 

* index_type: "t/s", "t/ms", "n" or "frame". 

Return: An 2-D NumPy array for part of the result. 

```python
result = time_delta_frame_axis_ds.timeSelect(start = 0, end = None, step = None, index_type = "t/s")
```

Selecting part of the result on the time offset dimension. 

* start: 
  * If index_type is "t/s" or "t/ms", then it will be the starting time offset of slicing. Default = 0. 
  * If index_type is "k", then it will be the starting sample differenc of slicing. Default = 0.

* end: 
  * If index_type is "t/s" or "t/ms", then it will be the ending time offset of slicing. Default = The duration of the audio in each frame. 
  * If index_type is "k", then it will be the ending sample differenc of slicing. Default = The max samples count of the audio in each frame. 

* step: 
  * If index_type is "t/s" or "t/ms", then it will be the time offset step of slicing. 
  * If index_type is "k", then it will be the sample offset step of slicing.
  * default = 1 sample offset. 

* index_type: "t/s", "t/ms" or "k". 

Return:  An 2-D NumPy array for part of the result. 

```python
time_delta_frame_axis_ds.plot(xstart=0, xend=None, ystart=0, yend=None, ax=None, imgPath=None, xlabel="t/s", ylabel="t/s", cbar=True)
```

To draw the per frame statistic result on the sub graph. 

If no subgraph is passed in, it will be displayed directly. 

If imgpath is passed in, the graph will be saved. 

* xstart: 
  * If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
  * If xlabel is "n", then it will be the starting sample count. Default = 0.
  * If xlabel is "frame", then it will be the starting frame count. Default = 0.

* xend: 

  * If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp of the audio. 

  * If xlabel is "n", then it will be the ending sample count. Default = The total count of samples. 

  * If xlabel is "frame", then it will be the ending frame count. Default = The total count of frames. 

* ystart: 

  * If ylabel is "t/s" or "t/ms", then it will be the starting time offset. Default = 0. 

  * If ylabel is "k", then it will be the starting sample offset. Default = 0.

* yend: 

  * If ylabel is "t/s" or "t/ms", then it will be the ending time offset. Default = The duration of the audio. 

  * If ylabel is "k", then it will be the ending sample offset. Default = The max samples count of the audio. 

* ax: A matplotlib.pyplot subplot to draw on. 
* imgPath: The path to save the graph. 
* xlabel: "t/s", "t/ms", "n" or "frame". 
* ylabel: "t/s", "t/ms" or "k"
* cbar: True to show the color bar. 

## Frequency domain analyse

**For more details of this section, please refer to:** [**3. Fourier Transform: from Time domain to Frequency domain.ipynb**](https://github.com/HarmoniaLeo/pyAudioKits/blob/main/3.%20Fourier%20Transform:%20from%20Time%20domain%20to%20Frequency%20domain.ipynb) **and** [**4. Short-time Analysis Method for Audio Signals.ipynb**](https://github.com/HarmoniaLeo/pyAudioKits/blob/main/4.%20Short-time%20Analysis%20Method%20for%20Audio%20Signals.ipynb) 

```python
import pyAudioKits.analyse as aly
```

### Methods

#### FFT

```python
spec = aly.FFT(input, N=None)
```

Calculate the FFT of the audio. 

* input: An **Audio object** or an **AudioFrames object**. 
* N: Num of Fourier Transform points. None if use all samples. 

Return: 

* If input is an **Audio object**: A **freq_axis_ds object** containing spectrum. 

* If input is an **AudioFrames object**: A **freq_frame_axis_ds object** containing short-time spectrum. 

#### Power Spectral Density

```python
psd = aly.PSD(input, N=None, dB=False)
```

Calculate the power spectral density of the audio. 

* input: An **Audio object** or an **AudioFrames object**. 
* N: Num of Fourier Transform points. None if use all samples. 
* dB: Whether to express the output in gain (dB). 

Return: 

* If input is an **Audio object**: A **freq_axis_ds_real object** containing power spectral density of the whole audio. 

* If input is an **AudioFrames object**: A **freq_frame_axis_ds_real object** containing power spectral density of each frame. 

#### Spectral Entropy

```python
specEnt = aly.specEntropy(input, N=None)
```

Calculate the spectral entropy of the audio. 

* input: An **Audio object** or an **AudioFrames object**. 

* N: Num of Fourier Transform points. None if use all samples. 

Return: 

* If input is an **Audio object**: A float object for the spectral entropy of the whole audio. 

* If input is an **AudioFrames object**: A **frame_axis_ds object** containing spectral entropy of each frame. 

#### Spectrum Peak

```python
specPeak, peakAmp = aly.getMaxFrequency(input,N=None,dB=False)
```

Get the frequency and the amplitude of spectrum peak. 

* input: An **Audio object** or an **AudioFrames object**. 
* N: Num of Fourier Transform points. None if use all samples. 
* dB: Whether to express the output amplitude in gain (dB). 

Return: 

* If input is an **Audio object**: The frequency and the amplitude of the spectrum peak of the whole audio. 

* If input is an **AudioFrames object**: A **frame_axis_ds object** containing the frequency of the spectrum peak and a frame_axis_ds object containing the amplitude of the spectrum peak. 

### Statistic result data structures

#### freq_axis_ds object

```python
result = freq_axis_ds.samples
result = freq_axis_ds_real.samples
```

Get statistics spectrum. 

Return: A one-dimensional NumPy array whose length is equal to $N/2$ , where $N$ is the number of points in FFT. 

freq_delta_axis_ds object supports **one-dimensional indexes and slices**. Any value used in the index represents the frequency point if it is an integer, or the frequency in Hz if it is a floating point number. 

Return: An 1-D NumPy array for part of the result spectrum. 

```python
result = freq_axis_ds.freqSelect(start = 0, end = None, step = None, index_type = "frequency/Hz")
result = freq_axis_ds_real.freqSelect(start = 0, end = None, step = None, index_type = "frequency/Hz")
```

Selecting part of the spectrum. 

* start: The starting of slicing frequency. Meaning depends on index_type. Default = 0.

* end: The ending of slicing frequency. Meaning depends on index_type. Default = Half of the sample rate. 

* step: The slicing step. Meaning depends on index_type. Default = 1 freq point. 

* index_type: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 

Return: An 1-D NumPy array for part of the spectrum. 

```python
freq_axis_ds.plot(start=0, end=None, ax=None, ylim=None, imgPath=None, xlabel="frequency/Hz", plot_type="amplitude",freq_scale="linear")
freq_axis_ds_real.plot(start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="frequency/Hz",freq_scale="linear")
```

To draw the statistic result spectrogram on the sub graph. 

If no subgraph is passed in, it will be displayed directly. 

If imgpath is passed in, the graph will be saved. 

* start: The starting frequency. Meaning depends on xlabel. Default = 0. 

* end: The ending frequency. Meaning depends on xlabel. Default = Half of the sample rate. 

* ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 

* ax: A matplotlib.pyplot subplot to draw on. 

* imgPath: The path to save the graph. 

* xlabel: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 
* plot_type: "amplitude", "dB" or "phase".  (plot function of freq_axis_ds_real doesn't have this parameter)

* freq_scale: "linear", "log" or "mel". 

#### freq_frame_axis_ds object

```python
result = freq_frame_axis_ds.samples
result = freq_frame_axis_ds_real.samples
```

Get statistics short-time spectrum. 

Return: A two-dimensional NumPy array whose shape is $(K,N/2)$ , where $N$ is the number of points in FFT and $K$ is the number of frames of the **AudioFrames object** to statistic. 

freq_frame_axis_ds object supports **one or two-dimensional indexes and slices**. 

The first dimension is the **frame dimension**. Any value represents the number of samples in the original Audio object if it is an integer, or the time in the original Audio object in seconds if it is a floating point number. 

The second index is the **frequncy dimension**. Any value represents the frequency point in each frame if it is an integer, or the frequency in each frame in Hz if it is a floating point number. 

Return: An 2-D NumPy array for part of the short-time spectrum. 

```python
result = freq_frame_axis_ds.freqSelect(start = 0, end = None, step = None, index_type = "frequency/Hz")
result = freq_frame_axis_ds_real.freqSelect(start = 0, end = None, step = None, index_type = "frequency/Hz")
```

Selecting part of the short-time spectrum on the frequency dimension.  

* start: The starting of slicing frequency. Meaning depends on index_type. Default = 0.

* end: The ending of slicing frequency. Meaning depends on index_type. Default = Half of the sample rate. 

* step: The slicing step. Meaning depends on index_type. Default = 1 freq point. 

* index_type: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 

Return: An 2-D NumPy array for part of the short-time spectrum. 

```python
result = freq_frame_axis_ds.frameSelect(start = 0, end = None, step = None, index_type = "frequency/Hz")
result = freq_frame_axis_ds_real.frameSelect(start = 0, end = None, step = None, index_type = "frequency/Hz")
```

Selecting part of the short-time spectrum on the frame dimension. 

* start: 
  * If index_type is "t/s" or "t/ms", then it will be the starting time stamp of slicing. Default = 0. 
  * If index_type is "n", then it will be the starting sample count of slicing. Default = 0.
  * If index_type is "frame", then it will be the starting frame count. Default = 0.

* end: 
  * If index_type is "t/s" or "t/ms", then it will be the ending time stamp of slicing. Default = The last time stamp  in each frame. 
  * If index_type is "n", then it will be the ending sample count of slicing. Default = The total count of samples in each frame. 
  * If index_type is "frame", then it will be the ending frame count of slicing. Default = The total count of frames. 

* step: 
  * If index_type is "t/s" or "t/ms", then it will be the time step of slicing. 
  * If index_type is "n", then it will be the samples step of slicing.
  * If index_type is "frame", then it will be the frames count step of slicing. 
  * Default = 1 frame. 

* index_type: "t/s", "t/ms", "n" or "frame". 

Return: An 2-D NumPy array for part of the short-time spectrum. 

```python
freq_frame_axis_ds.plot(xstart=0, xend=None, ystart=0, yend=None, ax=None, imgPath=None, xlabel="t/s", ylabel="frequency/Hz", plot_type="amplitude", cbar=True, freq_scale="linear")
freq_frame_axis_ds_real.plot(xstart=0, xend=None, ystart=0, yend=None, ax=None, imgPath=None, xlabel="t/s", ylabel="frequency/Hz", cbar=True, freq_scale="linear")
```

To draw the short-time spectrogram on the sub graph. 

If no subgraph is passed in, it will be displayed directly. 

If imgpath is passed in, the graph will be saved. 

* xstart: 
  * If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
  * If xlabel is "n", then it will be the starting sample count. Default = 0.
  * If xlabel is "frame", then it will be the starting frame count. Default = 0.

* xend: 
  * If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp in each frame. 
  * If xlabel is "n", then it will be the ending sample count. Default = The total count of samples in each frame. 
  * If xlabel is "frame", then it will be the ending frame count. Default = The total count of frames. 

* ystart: The starting frequency. Meaning depends on ylabel. Default = 0. 

* yend: The ending frequency. Meaning depends on ylabel. Default = Half of the sample rate. 

* ax: A matplotlib.pyplot subplot to draw on. 

* imgPath: The path to save the graph. 

* xlabel: "t/s", "t/ms", "n" or "frame". 

* ylabel: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 

* plot_type: "amplitude", "dB" or "phase". (plot function of freq_frame_axis_ds_real doesn't have this parameter)

* cbar: True to show the color bar. 

* freq_scale: "linear", "log" or "mel". 

## Model based analyse

**For more details of this section, please refer to:** [**6. Endpoint Detection and Speech Recognition.ipynb**](https://github.com/HarmoniaLeo/pyAudioKits/blob/main/6.%20Endpoint%20Detection%20and%20Speech%20Recognition.ipynb)

```python
import pyAudioKits.analyse as aly
```

### MFCC

```python
mfcc_feats = aly.MFCC(input,p=13,diff1=True,diff2=True,energy=True,frameDuration = 0.03, overlapRate = 0.5)
```

Calculate the MFCC features of the audio. 

* input: An **Audio object**. 

* p: MFCC order. 

* frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each frame (sample points). 

* overlapRate: A float object in [0,1) for the overlapping rate of the frames. 

* diff1: Use first-order differential features. 

* diff2: Use second-order differential features. 

* energy: Use energy features. 

Return: A 2-D NumPy array of MFCC features. Each row will be MFCC features of one frame.

### FBank

```python
fbank_feats = aly.fBank(input, filters = 26, frameDuration = 0.03, overlapRate = 0.5)
```

Calculate the Fbank features of the Audio. 

* input: An **Audio object**. 
* filters: Number of mel filters applied. 
* frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each frame (sample points). 
* overlapRate: A float object in [0,1) for the overlapping rate of the frames. 

Return: A 2-D NumPy array of FBank features. Each row will be FBank features of one frame. 

### Mel Spectrogram

```python
melspec_feats = aly.melSpec(input, spec_h=128, spec_w=128)
```

Calculate the Mel spectrogram features of the Audio. 

* input: An **Audio object**. 
* spec_h: The height of the mel spectrogram features, which determines the frequency resolution.
* spec_w: The width of the mel spectrogram features, which determines the temporal resolution.

Return: A 2-D NumPy array of Mel spectrogram features. 

### LPC

```python
es, ws = LPC(input, p=10)
```

Matlab style LPC for each frame. 

* input: An **AudioFrames object**.
* p: LPC order. 

Return: A list object of LPC error of each frame and a list object of LPC coefficient of each frame. 

## Filter

**For more details of this section, please refer to:** [**5. LTI Filter.ipynb**](https://github.com/HarmoniaLeo/pyAudioKits/blob/main/5.%20LTI%20Filter.ipynb)

```python
import pyAudioKits.filters as flt
```

### General Design

```python
output = flt.ltiFilter(input,numerators,denominators,zero_phase=False)
```

LTI filter design by specifying the denominator and numerator coefficients of the system function. 

* input: An **Audio object** or an **AudioFrames object**. 
* numerators: An Numpy array of the numerator coefficients of the system function. 
* denominators: An Numpy array of the denominator coefficients of the system function. 
* zero_phase: Use bi-directional filtering to maintain a phase response of 0. 

Return: 

* An **Audio object** if the input is an **Audio object**. 
* An **AudioFrames object** if the input is an **AudioFrames object**. 

### Butterworth Filter Design

```python
output=flt.lowPassFilterN(input,n,f,freq_type = "frequency/Hz", zero_phase = True)	#N order low pass Butterworth filter. 
output=flt.highPassFilterN(input,n,f,freq_type = "frequency/Hz", zero_phase = True)	#N order high pass Butterworth filter.
output=flt.bandPassFilterN(input,n,fLow,fHigh,freq_type = "frequency/Hz", zero_phase = True)	#N order band pass Butterworth filter.
output=flt.bandStopFilterN(input,n,fLow,fHigh,freq_type = "frequency/Hz", zero_phase = True)	#N order band stop Butterworth filter.
output=flt.lowPassFilter(input,fPass,fStop,ripplePass,rippleStop,freq_type = "frequency/Hz", zero_phase = True)	#Low pass Butterworth filter with specified ripple. 
output=flt.highPassFilter(input,fPass,fStop,ripplePass,rippleStop,freq_type = "frequency/Hz", zero_phase = True)	#High pass Butterworth filter with specified ripple. 
output=flt.bandPassFilter(input,fLowPass,fLowStop,fHighPass,fHighStop,ripplePass,rippleStop,freq_type = "frequency/Hz", zero_phase = True)	#Band pass Butterworth filter with specified ripple. 
output=flt.bandStopFilter(input,fLowPass,fLowStop,fHighPass,fHighStop,ripplePass,rippleStop,freq_type = "frequency/Hz", zero_phase = True)	#Band stop Butterworth filter with specified ripple. 
```

* input: An **Audio object** or an **AudioFrames object**. 
* n: The order. 
* f、fLow、fHigh: The cut-off frequency. 
* fPass: The passband frequency. 
* fStop: The stopband frequency. 
* ripplePass: The passband ripple. The signal will loses no more than ripplePass dB in the passband.
* rippleStop: The stopband ripple. The signal will have at least rippleStop dB attenuation in the stopband. 
* freq_type: "frequency/Hz"[0,sr/2), "frequency/(rad/s)"[0,sr * π), "normalized frequency/Hz"[0,1) or "normalized frequency/(rad/s)"[0,π)
* Zero_phase: Use bi-directional filtering to maintain a phase response of 0. 

Return: 

* An **Audio object** if the input is an **Audio object**. 
* An **AudioFrames object** if the input is an **AudioFrames object**. 

## Algorithm

```python
import pyAudioKits.algorithm as alg
```

### Speech Endpoint Detection

**For more details of this section, please refer to:** [**6. Endpoint Detection and Speech Recognition.ipynb**](https://github.com/HarmoniaLeo/pyAudioKits/blob/main/6.%20Endpoint%20Detection%20and%20Speech%20Recognition.ipynb)

```python
vad_result = alg.VAD(input,energyThresLow,energyThresHigh,zerocrossingThres,frameDuration = 0.03,overlapRate=0.5)
```

Speech endpoint detection based on double threshold method. 

* input: An **Audio object**. 

* energyThresLow: A lower threshold of energy for distinguish between silence and voice.

* energyThresHigh: A higher threshold of energy for distinguish between unvoiced and voiced.

* zerocrossingThres: Zero crossing rate threshold.

* frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each frame (sample points). 

* overlapRate: A float object in [0,1) for the overlapping rate of the frames. 

Return: A VAD object. 

```python
vad_result.plot(imgPath=None)
```

Visualize the result of VAD. Save the figure if imgPath is given, otherwise display directly. The blue parts are silence, the magenta parts are unvoice and the orange parts are voice. 

* imgPath: The direction to save the figure. 

```python
vad_result.slices()
```

Return the slices of voices in audio. 

Return: A list for **Audio objects**. 

### Speech Enhancement

**For more details of this section, please refer to:** [**7. Speech enhancement: spectral subtraction, Wiener and Kalman.ipynb**](https://github.com/HarmoniaLeo/pyAudioKits/blob/main/7.%20Speech%20enhancement:%20spectral%20subtraction%2C%20Wiener%20and%20Kalman.ipynb)

#### Spectrum subtraction

```python
output = alg.specSubstract(input, noise, beta=0.002, frameDuration = 0.03, overlapRate = 0.5, window = None)
```

Using spectral subtraction to reduce noise. 

* input: An **Audio object** of signal + noise. 

* noise: An **Audio object** of estimate noise. 

* beta: The beta parameter. 

* frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each frame (sample points). 

* overlapRate: A float object in [0,1) for the overlapping rate of the frames. 

* window: 
  * If string, it's the name of the window function (e.g., "hann")
  * If tuple, it's the name of the window function and any parameters (e.g., ("kaiser", 4.0))
  * If numeric, it is treated as the beta parameter of the "kaiser" window, as in scipy.signal.get_window.
  * If callable, it's a function that accepts one integer argument (the window length)
  * If list-like, it's a pre-computed window of the correct length Nx

Return: An **Audio object** of filtered signal. 

#### Wiener Filtering

```python
output = alg.wienerFilter(observed_signal,desired_signal,h_length=200, frameDuration = 0.03, overlapRate = 0.5, window = None)
```

Using Wiener filtering to reduce noise. 

* observed_signal: An **Audio object** of signal + noise. 

* desired_signal: An **Audio object** or estimated signal. 

* h_length: Orders. 

* frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each frame (sample points).

* overlapRate: A float object in [0,1) for the overlapping rate of the frames. 

* window: 

  * If string, it's the name of the window function (e.g., "hann")

  * If tuple, it's the name of the window function and any parameters (e.g., ("kaiser", 4.0))

  * If numeric, it is treated as the beta parameter of the "kaiser" window, as in scipy.signal.get_window.

  * If callable, it's a function that accepts one integer argument (the window length)

  * If list-like, it's a pre-computed window of the correct length Nx

Return: An **Audio object** of filtered signal. 

#### Kalman Filtering

```python
output = alg.kalmanFilter(input,noise,numIter=7,p=20, frameDuration = 0.05)
```

Using Kalman filtering to reduce noise. 

* input: An Audio object of signal + noise.
* noise: An Audio object of estimate noise. 
* numIter: Iterating times. 
* p: Orders. 
* frameDuration: A float object for the duration of each frame (seconds) or a int object for the length of each frame (sample points). 

Return: An **Audio object** of filtered signal. 

### Speech Recognition

#### DTW

```python
distance = alg.dtw(M1, M2)
```

Use DTW to calculate the similarity distance between two MFCC features. 

* M1: The first MFCC feature. 

* M2: The first MFCC feature. 

Return: A float object of the similarity distance between two MFCC features.

#### GMM+HMM

```python
gmmhmm_model = alg.GMMHMM(features, labels, n_iter = 10)
```

Construct and train a GMM+HMM model. 

* features: A list consisting of MFCC features.
* labels: The label corresponding to each MFCC feature in the features list. 
* n_iter: Iterating times.

Return: A GMMHMM object. 

```python
predicted_labels = gmmhmm_model.predict(features)
```

Use the trained GMM+HMM model to predict the labels on test set. 

* features: A list consisting of MFCC features.

Return: A list of predicted labels. 

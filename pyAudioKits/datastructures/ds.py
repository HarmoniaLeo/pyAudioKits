from .axis import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import librosa

def tResult(values, input, unit):
    if type(values) == np.ndarray:
        return frame_axis_ds(values, input.sr, input.step, unit)
    else:
        return values

def fResult(values, input, unit):
    if unit == "complex":
        if len(values.shape) == 2:
            return freq_frame_axis_ds(values, input.sr, input.step, unit)
        else:
            return freq_axis_ds(values, input.sr, unit)
    else:
        if len(values.shape) == 2:
            return freq_frame_axis_ds_real(values, input.sr, input.step, unit)
        else:
            return freq_axis_ds_real(values, input.sr, unit)

class one_axis_ds:
    def __init__(self, samples, sr, axis, unit):
        self.samples = samples
        self.sr = sr
        self.axis = axis
        self.shape = samples.shape
        self.unit = unit
    
    def __getitem__(self, index):
        if type(index) == tuple:
            raise Exception("Only support 1-D indices. ")
        else:
            return self.samples[self.axis[index]]
    
    def __len__(self):
        return self.samples.shape[0]
    
    def plot(self,start=0,end=None,ylim=None,ax=None,imgPath=None,xlabel="t/s",plot_transform=lambda x:x,slicing=lambda a,b,c,d,e:e.slicing(a,b,c,d),unit=None):
        indice = slicing(start,end,None,xlabel,self.axis)
        y = self.samples[indice]
        x = self.axis.getScale(xlabel)[indice]
        if ax==None:
            plt.xlabel(xlabel)
            if unit is None:
                plt.ylabel(self.unit)
            else:
                plt.ylabel(unit)
            if not ylim is None:
                plt.ylim(ylim[0],ylim[1])
            plt.plot(x, plot_transform(y))
            if imgPath==None:
                plt.show()
            else:
                plt.savefig(imgPath,dpi=500, bbox_inches = 'tight')
            plt.clf()
        else:
            ax.plot(x, y)
    
class time_axis_ds(one_axis_ds):
    def __init__(self, samples, sr, unit):
        super().__init__(samples, sr, time_axis(len(samples), sr), unit)
    
    def plot(self, start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="t/s"):
        """To draw the audio waveform on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        start: 
            If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
            If xlabel is "n", then it will be the starting sample count. Default = 0.
        end: 
            If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp of the audio. 
            If xlabel is "n", then it will be the ending sample count. Default = The total count of samples. 
        ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 
        ax: A matplotlib.pyplot subplot to draw on. 
        imgPath: The path to save the graph. 
        xlabel: "t/s", "t/ms" or "n". 
        """
        return super().plot(start, end, ylim, ax, imgPath, xlabel)
        

class frame_axis_ds(one_axis_ds):
    def __init__(self, samples, sr, step, unit):
        super().__init__(samples, sr, frame_axis(len(samples), sr, step), unit)
    
    def plot(self, start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="t/s"):
        """To draw the per frame statistic result on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        start: 
            If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
            If xlabel is "n", then it will be the starting sample count. Default = 0.
            If xlabel is "frame", then it will be the starting frame count. Default = 0.
        end: 
            If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp of the audio. 
            If xlabel is "n", then it will be the ending sample count. Default = The total count of samples. 
            If xlabel is "frame", then it will be the ending frame count. Default = The total count of frames. 
        ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 
        ax: A matplotlib.pyplot subplot to draw on. 
        imgPath: The path to save the graph. 
        xlabel: "t/s", "t/ms", "n" or "frame". 
        """
        return super().plot(start, end, ylim, ax, imgPath, xlabel)
    
    def frameSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting part of the result. 
        
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
            An 1-D NumPy array for part of the result. 
        """
        return self.samples[self.axis.slicing(start, end, step, index_type)]

class time_delta_axis_ds(one_axis_ds):
    def __init__(self, samples, sr, unit):
        super().__init__(samples, sr, time_delta_axis(len(samples), sr), unit)
    
    def plot(self, start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="t/s"):
        """To draw the auto correlation result on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        start: 
            If xlabel is "t/s" or "t/ms", then it will be the starting time offset. Default = 0. 
            If xlabel is "k", then it will be the starting sample offset. Default = 0.
        end: 
            If xlabel is "t/s" or "t/ms", then it will be the ending time offset. Default = The duration of the audio. 
            If xlabel is "k", then it will be the ending sample offset. Default = The max samples count of the audio. 
        ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 
        ax: A matplotlib.pyplot subplot to draw on. 
        imgPath: The path to save the graph. 
        xlabel: "t/s", "t/ms" or "k". 
        """
        return super().plot(start, end, ylim, ax, imgPath, xlabel)
    
    def timeSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting part of the result. 
        
        start: 
            If index_type is "t/s" or "t/ms", then it will be the starting time offset of slicing. Default = 0. 
            If index_type is "k", then it will be the starting sample offset of slicing. Default = 0.
        end: 
            If index_type is "t/s" or "t/ms", then it will be the ending time offset of slicing. Default = The duration of the audio. 
            If index_type is "k", then it will be the ending sample offset of slicing. Default = The max samples count of the audio. 
        step: 
            If index_type is "t/s" or "t/ms", then it will be the time offset step of slicing. 
            If index_type is "k", then it will be the sample offset step of slicing.
            default = 1 sample offset. 

        index_type: "t/s", "t/ms" or "k". 

        return:
            An Audio object of part of the audio. 
        """
        return self.samples[self.axis.slicing(start, end, step, index_type)]

class freq_axis_ds_base(one_axis_ds):
    def __init__(self, samples, sr, unit):
        super().__init__(samples, sr, freq_axis(len(samples), sr), unit)
        self.scale_transform = {"linear": lambda a,b,c,d,e:e.slicing(a,b,c,d), "log": lambda a,b,c,d,e:e.logSlicing(a,b,c,d), "mel": lambda a,b,c,d,e:e.melSlicing(a,b,c,d)}
    
    def freqSelect(self, start = 0, end = None, step = None, index_type = "frequency/Hz"):
        """Selecting part of the spectrum. 
        
        start: The starting of slicing frequency. Meaning depends on index_type. Default = 0.
        end: The ending of slicing frequency. Meaning depends on index_type. Default = Half of the sample rate. 
        step: The slicing step. Meaning depends on index_type. Default = 1 freq point. 

        index_type: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 

        return:
            An 1-D NumPy array for part of the spectrum. 
        """
        return self.samples[self.axis.slicing(start, end, step, index_type)]

class freq_axis_ds(freq_axis_ds_base):
    
    def plot(self, start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="frequency/Hz", plot_type="amplitude",freq_scale="linear"):
        """To draw the spectrogram on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        start: The starting frequency. Meaning depends on xlabel. Default = 0. 
        end: The ending frequency. Meaning depends on xlabel. Default = Half of the sample rate. 
        ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 
        ax: A matplotlib.pyplot subplot to draw on. 
        imgPath: The path to save the graph. 
        xlabel: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 
        plot_type: "amplitude", "dB" or "phase". 
        freq_scale: "linear", "log" or "mel". 
        """
        plot_transform = {"amplitude": lambda x: np.abs(x), "dB": lambda x: librosa.amplitude_to_db(np.abs(x)), "phase": lambda x: np.angle(x)}
        
        return super().plot(start, end, ylim, ax, imgPath, xlabel, plot_transform[plot_type], slicing=self.scale_transform[freq_scale],unit=plot_type)

class freq_axis_ds_real(freq_axis_ds_base):
    def plot(self, start=0, end=None, ylim=None, ax=None, imgPath=None, xlabel="frequency/Hz",freq_scale="linear"):
        """To draw the statistic result spectrogram on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        start: The starting frequency. Meaning depends on xlabel. Default = 0. 
        end: The ending frequency. Meaning depends on xlabel. Default = Half of the sample rate. 
        ylim: A tuple (y_start,y_end) for display range on y-axis. The default (None) is adaptive. 
        ax: A matplotlib.pyplot subplot to draw on. 
        imgPath: The path to save the graph. 
        xlabel: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 
        freq_scale: "linear", "log" or "mel". 
        """
        return super().plot(start, end, ylim, ax, imgPath, xlabel, slicing=self.scale_transform[freq_scale])

class two_axis_ds:
    def __init__(self, samples, sr, axis1, axis2, unit):
        self.samples = samples
        self.sr = sr
        self.axis1 = axis1
        self.axis2 = axis2
        self.shape = samples.shape
        self.unit = unit
    
    def __getitem__(self, index):
        if type(index) == tuple:
            if len(index) > 2:
                raise Exception("Only support 1-D or 2-D indices. ")
            index1 = self.axis1[index[0]]
            index2 = self.axis2[index[1]]
            if type(index1) == np.array:
                index1 = index1[:,None]
            if type(index2) == np.array:
                index2 = index2[None,:]
            return self.samples[index1, index2]
        else:
            return self.samples[self.axis1[index]]
    
    def __len__(self):
        return self.samples.shape[0]
    
    def plot(self,xstart=0,xend=None,ystart=0,yend=None,
                    ax=None,imgPath=None,
                    xlabel="t/s",ylabel="t/s",
                    cbar=True,
                    plot_transform = lambda x:x,
                    slicing_axis1=lambda a,b,c,d,e:e.slicing(a,b,c,d),slicing_axis2=lambda a,b,c,d,e:e.slicing(a,b,c,d),unit=None):

        ax1_indice = slicing_axis1(xstart,xend,None,xlabel,self.axis1)
        ax1 = self.axis1.getScale(xlabel)[ax1_indice]
        ax2_indice = slicing_axis2(ystart,yend,None,ylabel,self.axis2)
        ax2 = self.axis2.getScale(ylabel)[ax2_indice]

        mat = self.samples[ax1_indice[:,None], ax2_indice[None,:]]
        mat = pd.DataFrame(plot_transform(mat.T),index=[round(i,2) for i in ax2],columns=[round(i,2) for i in ax1])
        if ax==None:
            plt.figure(figsize=(12, 4))
            sns.heatmap(mat,cmap=plt.cm.plasma,cbar=cbar)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=60)
            # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            if imgPath==None:
                plt.show()
            else:
                plt.savefig(imgPath,dpi=500, bbox_inches = 'tight')
            plt.clf()
        else:
            sns.heatmap(mat,cmap=plt.cm.plasma,xticklabels=ax1,yticklabels=ax2,ax=ax,cbar=cbar)

class time_frame_axis_ds(two_axis_ds):
    def __init__(self, samples, sr, step, unit):
        super().__init__(samples, sr, frame_axis(samples.shape[0], sr, step), time_axis(samples.shape[1], sr), unit)
    
    def frameSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting part of the result on the frame axis. 
        
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
            An 2-D NumPy array for part of the result. 
        """
        return self.samples[self.axis1.slicing(start, end, step, index_type)]
    
    def timeSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting part of the result on the time axis. 
        
        start: 
            If index_type is "t/s" or "t/ms", then it will be the starting time stamp of slicing. Default = 0. 
            If index_type is "n", then it will be the starting sample count of slicing. Default = 0.
        end: 
            If index_type is "t/s" or "t/ms", then it will be the ending time stamp of slicing. Default = The last time stamp in each frame. 
            If index_type is "n", then it will be the ending sample count of slicing. Default = The total count of samples in each frame. 
        step: 
            If index_type is "t/s" or "t/ms", then it will be the time step of slicing. 
            If index_type is "n", then it will be the samples step of slicing.
            Default = 1 sample. 

        index_type: "t/s", "t/ms" or "n". 

        return:
            An 2-D NumPy array for part of the result. 
        """
        return self.samples[:,self.axis2.slicing(start, end, step, index_type)]

class time_delta_frame_axis_ds(two_axis_ds):
    def __init__(self, samples, sr, step, unit):
        super().__init__(samples, sr, frame_axis(samples.shape[0], sr, step), time_delta_axis(samples.shape[1],sr), unit)
    
    def plot(self, xstart=0, xend=None, ystart=0, yend=None, ax=None, imgPath=None, xlabel="t/s", ylabel="t/s", cbar=True):
        """To draw the per frame statistic result on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        xstart: 
            If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
            If xlabel is "n", then it will be the starting sample count. Default = 0.
            If xlabel is "frame", then it will be the starting frame count. Default = 0.
        xend: 
            If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp of the audio. 
            If xlabel is "n", then it will be the ending sample count. Default = The total count of samples. 
            If xlabel is "frame", then it will be the ending frame count. Default = The total count of frames. 
        ystart: 
            If ylabel is "t/s" or "t/ms", then it will be the starting time offset. Default = 0. 
            If ylabel is "k", then it will be the starting sample offset. Default = 0.
        yend: 
            If ylabel is "t/s" or "t/ms", then it will be the ending time offset. Default = The duration of the audio in each frame. 
            If ylabel is "k", then it will be the ending sample offset. Default = The max samples count of the audio in each frame. 
        ax: A matplotlib.pyplot subplot to draw on. 
        imgPath: The path to save the graph. 
        xlabel: "t/s", "t/ms", "n" or "frame". 
        ylabel: "t/s", "t/ms" or "k"
        cbar: True to show the color bar. 
        """
        return super().plot(xstart, xend, ystart, yend, ax, imgPath, xlabel, ylabel,cbar)
    
    def frameSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting part of the result on the frame dimension. 
        
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
            An 2-D NumPy array for part of the result. 
        """
        return self.samples[self.axis1.slicing(start, end, step, index_type)]
    
    def timeSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting part of the result on the time offset dimension. 
        
        start: 
            If index_type is "t/s" or "t/ms", then it will be the starting time offset of slicing. Default = 0. 
            If index_type is "k", then it will be the starting sample differenc of slicing. Default = 0.
        end: 
            If index_type is "t/s" or "t/ms", then it will be the ending time offset of slicing. Default = The duration of the audio in each frame. 
            If index_type is "k", then it will be the ending sample differenc of slicing. Default = The max samples count of the audio in each frame. 
        step: 
            If index_type is "t/s" or "t/ms", then it will be the time offset step of slicing. 
            If index_type is "k", then it will be the sample offset step of slicing.
            default = 1 sample offset. 

        index_type: "t/s", "t/ms" or "k". 

        return:
            An 2-D NumPy array for part of the result. 
        """
        return self.samples[:,self.axis2.slicing(start, end, step, index_type)]

class freq_frame_axis_ds_base(two_axis_ds):
    def __init__(self, samples, sr, step, unit):
        super().__init__(samples, sr, frame_axis(samples.shape[0], sr, step), freq_axis(samples.shape[1], sr), unit)
        self.scale_transform = {"linear": lambda a,b,c,d,e:e.slicing(a,b,c,d), "log": lambda a,b,c,d,e:e.logSlicing(a,b,c,d), "mel": lambda a,b,c,d,e:e.melSlicing(a,b,c,d)}
    
    def frameSelect(self, start = 0, end = None, step = None, index_type = "t/s"):
        """Selecting part of the short-time spectrum on the frame dimension. 
        
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
            An 2-D NumPy array for part of the short-time spectrum. 
        """
        return self.samples[self.axis1.slicing(start, end, step, index_type)]
    
    def freqSelect(self, start = 0, end = None, step = None, index_type = "frequency/Hz"):
        """Selecting part of the short-time spectrum on the frequency dimension.  
        
        start: The starting of slicing frequency. Meaning depends on index_type. Default = 0.
        end: The ending of slicing frequency. Meaning depends on index_type. Default = Half of the sample rate. 
        step: The slicing step. Meaning depends on index_type. Default = 1 freq point. 

        index_type: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 

        return:
            An 2-D NumPy array for part of the short-time spectrum. 
        """
        return self.samples[:,self.axis2.slicing(start, end, step, index_type)]

class freq_frame_axis_ds(freq_frame_axis_ds_base):
    
    def plot(self, xstart=0, xend=None, ystart=0, yend=None, ax=None, imgPath=None, xlabel="t/s", ylabel="frequency/Hz", plot_type="amplitude", cbar=True, freq_scale="linear"):
        """To draw the short-time spectrogram on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        xstart: 
            If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
            If xlabel is "n", then it will be the starting sample count. Default = 0.
            If xlabel is "frame", then it will be the starting frame count. Default = 0.
        xend: 
            If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp of the audio. 
            If xlabel is "n", then it will be the ending sample count. Default = The total count of samples. 
            If xlabel is "frame", then it will be the ending frame count. Default = The total count of frames. 
        ystart: The starting frequency. Meaning depends on ylabel. Default = 0. 
        yend: The ending frequency. Meaning depends on ylabel. Default = Half of the sample rate. 
        ax: A matplotlib.pyplot subplot to draw on. 
        imgPath: The path to save the graph. 
        xlabel: "t/s", "t/ms", "n" or "frame". 
        ylabel: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 
        plot_type: "amplitude", "dB" or "phase". 
        cbar: True to show the color bar. 
        freq_scale: "linear", "log" or "mel". 
        """
        plot_transform = {"amplitude": lambda x: np.abs(x), "dB": lambda x: librosa.amplitude_to_db(np.abs(x)), "phase": lambda x: np.angle(x)}
        return super().plot(xstart, xend, ystart, yend, ax, imgPath, xlabel, ylabel, cbar, plot_transform[plot_type], slicing_axis2=self.scale_transform[freq_scale], unit=plot_type)
    
    

class freq_frame_axis_ds_real(freq_frame_axis_ds_base):
    def plot(self, xstart=0, xend=None, ystart=0, yend=None, ax=None, imgPath=None, xlabel="t/s", ylabel="frequency/Hz", cbar=True, freq_scale="linear"):
        """To draw the short-time statistic result spectrogram on the sub graph. 
        If no subgraph is passed in, it will be displayed directly. 
        If imgpath is passed in, the graph will be saved. 

        xstart: 
            If xlabel is "t/s" or "t/ms", then it will be the starting time stamp. Default = 0. 
            If xlabel is "n", then it will be the starting sample count. Default = 0.
            If xlabel is "frame", then it will be the starting frame count. Default = 0.
        xend: 
            If xlabel is "t/s" or "t/ms", then it will be the ending time stamp. Default = The last time stamp of the audio. 
            If xlabel is "n", then it will be the ending sample count. Default = The total count of samples. 
            If xlabel is "frame", then it will be the ending frame count. Default = The total count of frames. 
        ystart: The starting frequency. Meaning depends on ylabel. Default = 0. 
        yend: The ending frequency. Meaning depends on ylabel. Default = Half of the sample rate. 
        ax: A matplotlib.pyplot subplot to draw on. 
        imgPath: The path to save the graph. 
        xlabel: "t/s", "t/ms", "n" or "frame". 
        ylabel: "frequency/Hz", "frequency/(rad/s)", "normalized frequency/Hz", "normalized frequency/(rad/s)" or "freq point". 
        cbar: True to show the color bar. 
        freq_scale: "linear", "log" or "mel". 
        """
        return super().plot(xstart, xend, ystart, yend, ax, imgPath, xlabel, ylabel, cbar, slicing_axis2=self.scale_transform[freq_scale])
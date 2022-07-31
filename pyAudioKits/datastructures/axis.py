import numpy as np
from math import *

class axis:
    def __init__(self, length, sr, int_index_map, float_index_map, index_map, scales_map):
        self.scale = np.arange(0, length, 1)
        self.sr = sr
        self.default_start = 0
        self.default_end = self.scale.shape[0]
        self.default_step = 1
        self.int_index_map = int_index_map
        self.float_index_map = float_index_map
        self.index_map = index_map
        self.scales_map = scales_map
    
    def __getitem__(self, index):
        if type(index) == slice:
            start = index.start
            end = index.stop
            step = index.step
            if start is None:
                start = self.default_start
            elif type(start) == float:
                start = self.float_index_map(start)
            else:
                start = self.int_index_map(start)
            if end is None:
                end = self.default_end
            elif type(end) == float:
                end = self.float_index_map(end)
            else:
                end = self.int_index_map(end)
            if step is None:
                step = self.default_step
            elif type(step) == float:
                step = self.float_index_map(step)
            else:
                step = self.int_index_map(step)
            return self.scale[start:end:step]
        else:
            if type(index) == float:
                index = self.float_index_map(index)
            else:
                index = self.int_index_map(index)
            return self.scale[index]
    
    def slicing(self, start, end, step, index_type):
        start = self.index_map[index_type](start)
        if end is None:
            end = self.default_end
        else:
            end = self.index_map[index_type](end)
        if step is None:
            step = self.default_step
        else:
            step = self.index_map[index_type](step)
        return self.scale[start:end:step]
            
    def getScale(self, index_type):
        return self.scales_map[index_type](self.scale)

class time_axis(axis):
    def __init__(self, length, sr):
        super().__init__(length, sr, 
        lambda x:x, 
        lambda x: int(self.sr * x), 
        {"t/s": lambda x: int(self.sr * x), "t/ms": lambda x: int(self.sr * x / 1000), "n": lambda x:x}, 
        {"n": lambda x: x, "t/s": lambda x: x / sr, "t/ms": lambda x: x * 1000 / sr})

class time_delta_axis(axis):
    def __init__(self, length, sr):
        max_diff = int((length + 1) / 2) - 1
        super().__init__(length, sr, 
        lambda x:x + max_diff, 
        lambda x: int(self.sr * x + max_diff), 
        {"t/s": lambda x: int(self.sr * x + max_diff), "t/ms": lambda x: int(self.sr * x / 1000 + max_diff), "k": lambda x:x + max_diff}, 
        {"t/s": lambda x: (x - max_diff)/sr, "t/ms": lambda x: 1000*(x - max_diff)/sr, "k": lambda x: x - max_diff})

class frame_axis(axis):
    def __init__(self, length, sr, step):
        time_step = step / sr
        super().__init__(length, sr, 
        lambda x: int(x / step), 
        lambda x: int(x / time_step), 
        {"t/s": lambda x: int(x / time_step), "t/ms": lambda x: int(x / time_step / 1000), "n": lambda x: int(x / step), "frame": lambda x:x}, 
        {"t/s": lambda x: x * time_step, "t/ms": lambda x: 1000 * x * time_step, "n": lambda x: x * step, "frame": lambda x:x})

class freq_axis(axis):
    def __init__(self, length, sr):
        super().__init__(length, sr, 
        lambda x: x, 
        lambda x: int(x/sr*length), 
        {"frequency/Hz": lambda x: int(x/sr*length), "frequency/(rad/s)": lambda x: int(x/sr*length/2/np.pi), "normalized frequency/Hz": lambda x:int(x*length), "normalized frequency/(rad/s)": lambda x:int(x*length/np.pi/2), "freq point": lambda x:x}, 
        {"frequency/Hz": lambda x: x*sr/length, "frequency/(rad/s)": lambda x: x*sr/length*2*np.pi, "normalized frequency/Hz": lambda x:x/length, "normalized frequency/(rad/s)": lambda x:2*x/length*np.pi, "freq point": lambda x:x})
        self.default_end = int(length / 2)
    
    def melSlicing(self, start, end, step, index_type):
        start = self.index_map[index_type](start)
        if end is None:
            end = self.default_end
        else:
            end = self.index_map[index_type](end)
        if step is None:
            step = self.default_step
        else:
            step = self.index_map[index_type](step)
        
        steps = int((end - start) / step)
        start = self.scales_map["frequency/Hz"](start)
        end = self.scales_map["frequency/Hz"](end)

        slicing = 700 * (np.power(10, np.linspace(2595 * np.log10(1 + start/700), 2595 * np.log10(1 + end/700), steps)/2595) - 1)
        slicing = np.array([self.index_map["frequency/Hz"](s) for s in slicing.tolist()])
        return self.scale[slicing.astype(int)]
    
    def logSlicing(self, start, end, step, index_type):
        start = self.index_map[index_type](start)
        if end is None:
            end = self.default_end
        else:
            end = self.index_map[index_type](end)
        if step is None:
            step = self.default_step
        else:
            step = self.index_map[index_type](step)
        
        steps = int((end - start) / step)

        slicing = np.power(10, np.linspace(np.log10(1 + start), np.log10(1 + end), steps)) - 1
        return self.scale[slicing.astype(int)]
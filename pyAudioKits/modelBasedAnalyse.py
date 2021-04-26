from python_speech_features import mfcc,delta
from sklearn.linear_model import LinearRegression
import numpy as np
from audio import Audio,concatenate
import pandas as pd

def MFCC(audio,p=13,frameDuration=0.03,overlapRate=0.5):
    windowLength=int(frameDuration*audio.sr)
    step=int(windowLength*(1-overlapRate))
    wav_feature=mfcc(audio.samples,audio.sr,numcep=p,winlen=windowLength/audio.sr,winstep=step/audio.sr)
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    return np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    
def LPC(audio,p=10,frameDuration=0.03,overlapRate=0.5):
    windowLength=int(frameDuration*audio.sr)
    step=int(windowLength*(1-overlapRate))
    y1=audio.samples[:-1]-0.97*audio.samples[:-1]
    l=audio.samples.shape[0]
    indexer = np.arange(windowLength)[None, :] + step*np.arange(int((l-windowLength)/step))[:, None]
    flatten=y1[indexer]
    ys=[]
    es=[]
    for i in range(0,flatten.shape[0]):
        x=flatten[i]
        indexer = np.arange(p)[None, :] + np.arange(windowLength-p-1)[:, None]
        X=x[indexer]
        y=pd.Series(x[p+1:])
        X=pd.DataFrame(X)
        linreg = LinearRegression(fit_intercept=0)
        linreg.fit(X, y)
        lpc=np.array(linreg.coef_)
        y2=np.sum(lpc*X,axis=1)
        e=y-y2
        ys.append(Audio(y2[:step],audio.sr))
        es.append(Audio(e[:step],audio.sr))
    return concatenate(ys),concatenate(es)
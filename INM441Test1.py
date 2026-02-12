import wave
import numpy as np
import pylab as pl
waveAbsPath="mic.wav"
f=wave.open(waveAbsPath,"rb")
params=f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
str_data = f.readframes(nframes)
wave_data = np.frombuffer(str_data, "int16")
wave_data.shape = -1, 2
#将其转置得到：
wave_data = wave_data.T
#通过取样点数和取样频率计算出每个取样的时间：
time = np.arange(0, nframes)/framerate
pl.plot(time, wave_data.reshape((wave_data.shape[0] * wave_data.shape[1])))
pl.show()

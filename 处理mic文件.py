import wave
import numpy as np
import array
import struct
import pandas as pd

mic = pd.read_csv('mic.csv', header = None)
data  = mic.values
mic = pd.read_csv('mic.csv', header = None)
data  = mic.values
data=data[0]
rate=8000
pcmData16Bit=data[0:16000]
print(pcmData16Bit)
pcmData16Bit_f=pcmData16Bit.astype(np.float32)

name="mic.wav"

def waveSetOut(outPath,pcmArray,pcmRate):
    waveF=wave.open(outPath,"wb")
    waveF.setnchannels(1)
    waveF.setsampwidth(2)
    waveF.setframerate(pcmRate)
    for val in pcmArray:
        val=round(float(val))
        dataStruct =struct.pack("<h",val)
        waveF.writeframesraw(dataStruct)
    waveF.writeframes(b"")
    waveF.close()

waveSetOut(name,pcmData16Bit_f,rate)
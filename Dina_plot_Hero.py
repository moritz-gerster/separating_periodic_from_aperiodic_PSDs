#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:18:38 2021

@author: moritzgerster
"""

#import the pyplot and wavfile modules 

import matplotlib.pyplot as plot

from scipy.io import wavfile

 
plt.specgram(signal_real[:srate*1], srate, mode="psd")
spectrum, f, t, im = plt.specgram(signal_real[:], srate, mode="psd", scale="dB")
plt.specgram(signal_real[:srate*1], srate, mode="angle")
plt.specgram(signal_real[:srate*1], srate, mode="phase")

# Read the wav file (mono)

samplingFrequency, signalData = wavfile.read('y.wav')

 

# Plot the signal read from wav file

plot.subplot(211)

plot.title('Spectrogram of a wav file with piano music')

 

plot.plot(signalData)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

 

plot.subplot(212)

plot.specgram(signalData,Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')

 

plot.show()
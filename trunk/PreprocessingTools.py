# -*- coding: utf-8 -*-
'''
Based on previous works by Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''
import numpy as np
from scipy import signal

## method correct_offset
#
# Eliminates displacement of the measurement data originated by initial tension
# by subtracting the average value of the x first measurements from every
# value.

def correct_offset(measurement, x=None):
    assert measurement.shape [0] > measurement.shape [1]
    for ii in range(measurement.shape[1]):
        tmp = measurement[:,ii]
        if x is not None:
            measurement[:,ii] = tmp - tmp[0:x].mean(0)
        else:
            measurement[:,ii] = tmp - tmp.mean(0)
    return measurement


def decimate_data(measurement, decimate_factor):    
    assert measurement.shape [0] > measurement.shape [1]   
    
    num_channels = measurement.shape[1]
    num_time_steps = measurement.shape[0]    
    
    for ii in range(measurement.shape [1]):
        tmp = measurement[:,ii]
        tmp = signal.decimate(tmp, decimate_factor, axis = 0)        
        if ii == 0:
            meas_decimated = np.zeros((tmp.shape[0],num_channels))       
        meas_decimated[:,ii] = tmp      
    return meas_decimated

def correct_time_lag(measurement, channel, lag, sampling_rate):
    #lag in ms
    #sampling rate in 1/s
    
    def gcd(a, b):
        """Return greatest common divisor using Euclid's Algorithm."""
        
        while b:      
            a, b = b, a % b
        
        return a
    
    def lcm(a, b):
        print(a,b)
        """Return lowest common multiple."""
        return a * b // gcd(a, b)
    
    delta_t=1/sampling_rate*1000 #ms
    sig_num=2
    factor = lcm(int(delta_t*10**sig_num), int(lag*10**sig_num))/(10**sig_num)
    print(factor)
    from scipy.signal import resample, decimate
    import matplotlib.pyplot as plot
    plot.figure()
    plot.plot(measurement[:,channel])
    print(factor*measurement.shape[0])
    resampled_col=resample(measurement[:,channel], factor*measurement.shape[0])
    num_shift = int(sampling_rate*factor*lag/1000)
    shifted_col= resampled_col[num_shift:]
    decimated_col=decimate(shifted_col, factor)
    measurement=measurement[:decimated_col.shape[1],:]
    measurement[:,channel]=decimated_col
    plot.figure()
    plot.plot(decimated_col)
    plot.show()
    return measurement
# -*- coding: utf-8 -*-
'''
Written by Volkmar Zabel 2016
'''

import numpy as np
#import sys
import os
#import json

import multiprocessing as mp
import ctypes as c
from collections import deque
import datetime
from scipy.signal import signaltools
from scipy.signal.windows import get_window
import warnings
#from scipy.lib.six import string_types

from copy import deepcopy

from PreprocessingTools import PreprocessData
from _ast import Str
#from StabilDiagram import main_stabil, StabilPlot, nearly_equal

#import pydevd
    
'''TO DO:
Modify all methods for the calculation of half spectra such that they can be used here!
'''    
    

class PLSCF(object):
    
    def __init__(self,prep_data):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_data, PreprocessData)
        self.prep_data =prep_data
        self.setup_name = prep_data.setup_name
        self.start_time = prep_data.start_time
        
        #             0             1     
        #self.state= [Half_spectra, Modal Par.
        self.state  =[False,    False]
        
        self.begin_frequency = None
        self.end_frequency = None
        self.nperseg = None
        self.factor_a = None
        self.selected_omega_vector = None
        self.num_omega = None
        self.spectrum_tensor = None
        
        self.max_model_order = None
        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None
            
    @classmethod
    def init_from_config(cls,conf_file, prep_data):
        assert os.path.exists(conf_file)
        assert isinstance(prep_data, PreprocessData)
        
        with open(conf_file, 'r') as f:
            
            assert f.__next__().strip('\n').strip(' ') == 'Begin Frequency:'
            begin_frequency = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'End Frequency:'
            end_frequency = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Samples per time segment:'
            nperseg = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Maximum Model Order:'
            max_model_order= int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Use Multiprocessing:'
            multiprocessing= f.__next__().strip('\n').strip(' ')=='yes'
            
        pLSCF_object = cls(prep_data)
        pLSCF_object.build_half_spectra(begin_frequency, end_frequency, nperseg, multiprocess=multiprocessing)
        pLSCF_object.compute_modal_params(max_model_order, multiprocessing)
        
        return pLSCF_object
        
    def build_half_spectra(self, begin_frequency, end_frequency, nperseg, multiprocess=True):

         
        ###############################################################################
        ##################### Construct half-spectrum matrix ##########################
        ###############################################################################
        
        '''
        Builds a 3D tensor with cross spectral densities:
        Dimensions: number_of_all_channels x number_of_references x number_of_freq_lines
        '''
                
        print('Constructing half-spectrum matrix ... ')
        assert isinstance(begin_frequency, float)
        assert isinstance(end_frequency, float)
        assert isinstance(nperseg, int)

        self.begin_frequency=begin_frequency
        self.end_frequency=end_frequency
        self.nperseg=nperseg
        total_time_steps = self.prep_data.total_time_steps
        ref_channels = sorted(self.prep_data.ref_channels)
        roving_channels = self.prep_data.roving_channels
        measurement = self.prep_data.measurement
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels
        
        sampling_rate = self.prep_data.sampling_rate

                
        # Extract reference time series for half spectra 
        
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        #print(all_channels)
        
        refs = (measurement[:,ref_channels])
           
        
        begin_omega = begin_frequency * 2 * np.pi
        end_omega = end_frequency * 2 * np.pi
        factor_a = 0
        
        for ii in range(num_ref_channels):
            
            this_ref = refs[:,ii]
                        
            for jj in range(num_analised_channels):
                
                this_response = measurement[:,jj]
        
        
                #### spectra based on cross correlations ####
                
                '''
                num_ref_samples = int(len(this_ref) * 0.9) # 10% of signal length as time shift length
                #x_corr = np.flipud(np.correlate(this_ref[0:num_ref_samples], this_response, mode='valid'))
                x_corr = np.correlate(this_ref[0:num_ref_samples], this_response, mode='valid')    
        
                fig = plt.figure(figsize = [10,5])
                
                ax1 = fig.add_subplot(2,1,1)
                ax2 = fig.add_subplot(2,1,2)
        
                ax1.plot( x_corr, 'b-', label='$R_xy_inv_FFT$')
                               
                (x_corr, factor_a) = fcl.Exp_Win(x_corr, 0.001)
                
                ax2.plot( x_corr, 'b-', label='$R_xy_inv_FFT_1$')
                plt.show
               
                print('factor_a = ', factor_a)
               
                num_samples = int(len(x_corr))
        
                #frequency_vector, this_half_spec = fcl.FFT_Average(x_corr, fs=sampling_rate, window='boxcar', \
                #    nperseg=4096, noverlap=(4096/2), return_onesided=True)        
                frequency_vector, this_half_spec = fcl.FFT_Average(x_corr, fs=sampling_rate, window='boxcar', \
                    nperseg=num_samples, noverlap=None, return_onesided=True)        
                omega_vector = frequency_vector * 2 * np.pi
                '''
 
                
                #### spectra from correlation functions of independent sections ####
                '''
                '''
               
                #num_sections = 200  # number of independent sections for averaging
                #blocklength = int(len(this_ref) * 1/num_sections) 
                blocklength = int(len(this_ref) * 0.025) 
                #print('blocklength = ', blocklength)
                            
                tmp_freq, tmp_cross_spec = self.CrossSpectrum(this_ref, this_response, \
                        fs=sampling_rate, window='boxcar', nperseg=blocklength, \
                        noverlap=None, detrend='constant', zeropad = True, return_onesided=True, \
                        scaling='spectrum')
                        
                R_xy = np.fft.irfft(tmp_cross_spec)
                R_xy = R_xy[0:int(len(R_xy)/2)]
        
                (R_xy, factor_a) = self.Exp_Win(R_xy, 0.001)
                #print('factor_a = ', factor_a)
                
                frequency_vector, this_half_spec = self.FFT_Average(R_xy, fs=sampling_rate, window='boxcar', \
                    nperseg=len(R_xy), noverlap=None, return_onesided=True)        
                omega_vector = frequency_vector * 2 * np.pi
                
                #fig = plt.figure(figsize = [10,5])
                
                #ax1 = fig.add_subplot(2,1,1)
                #ax2 = fig.add_subplot(2,1,2)
        
                #ax1.plot( R_xy, 'b-', label='$R_xy_inv_FFT$')
                       
                #ax2.plot( frequency_vector, abs(this_half_spec), 'b-')
                #plt.show
           
        
                #### spectra based on Welch's method ####
                '''
                frequency_vector, this_half_spec = self.CrossSpectrum(this_ref, this_response,\
                    fs=sampling_rate, window='boxcar', nperseg=nperseg, noverlap = nperseg/2, \
                    return_onesided=True, scaling='spectrum')        
                omega_vector = frequency_vector * 2 * np.pi
                
               
                fig = plt.figure(figsize = [10,5])
                ax1 = fig.add_subplot(1,1,1)
                ax1.plot( frequency_vector, abs(this_half_spec), 'b-')
                plt.show
                '''
        
                if ii == jj == 0:
                                       
                    # Store only frequency range to be analysed
        
                    cond_1 = [omega_vector >= begin_omega]
                    omega_list = [omega_vector]
                    omega_extract_1 = np.select(cond_1, omega_list)
                    cond_2 = [omega_vector <= end_omega]
                    omega_list = [omega_extract_1]
                    omega_extracted = np.select(cond_2, omega_list)
                    index_begin_omega = np.amin(np.nonzero(omega_extracted))
                    index_end_omega = np.amax(np.nonzero(omega_extracted))
        
                    selected_omega_vector = omega_vector[index_begin_omega : index_end_omega+1]
                    num_omega = len(selected_omega_vector)
                                                
                    spectrum_tensor = np.zeros((num_analised_channels, num_ref_channels, len(selected_omega_vector)), dtype = complex)
                    
                spectrum_tensor[jj,ii,:] = this_half_spec[index_begin_omega : index_end_omega+1]
    
        self.selected_omega_vector = selected_omega_vector
        self.num_omega = num_omega
        self.spectrum_tensor = spectrum_tensor
        self.factor_a = factor_a

        
        self.state[0]=True

        
    
    ###############################################################################
    ####### One-sided cross spectra based on Welch's method #######################
    ###############################################################################
    
    def CrossSpectrum(self, x, y, fs=1.0, window='boxcar', nperseg=1024, noverlap=None, 
    detrend='constant', zeropad = False, return_onesided=True, scaling='spectrum', axis=-1):
        
        """
        Estimate cross power spectral density using Welch's method.
        Welch's method [1]_ computes an estimate of the power spectral density
        by dividing the data into overlapping segments, computing a modified
        periodogram for each segment and averaging the periodograms.
        Parameters
        ----------
        x : array_like
        y : array_like
        Time series of measurement values
        fs : float, optional
        Sampling frequency of the `x` and `y` time series in units of Hz. Defaults
        to 1.0.
        window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length will be used for nperseg.
        Defaults to 'hanning'.
        nperseg : int, optional
        Length of each segment. Defaults to 256.
        noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg / 2``. Defaults to None.
        nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is `nperseg`. Defaults to None.
        detrend : str or function, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`. If it is a
        function, it takes a segment and returns a detrended segment.
        Defaults to 'constant'.
        return_onesided : bool, optional
        If True, return a one-sided spectrum. If False return
        a two-sided spectrum. Note that for complex data, a two-sided
        spectrum is always returned.
        scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where Pxx has units of V**2/Hz if x is measured in V and computing
        the power spectrum ('spectrum') where Pxx has units of V**2 if x is
        measured in V. Defaults to 'density'.
        axis : int, optional
        Axis along which the periodogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
        Returns
        -------
        f : ndarray
        Array of sample frequencies.
        Pxy : ndarray
        Cross power spectral density or cross power spectrum of x.
        See Also
        --------
        periodogram: Simple, optionally modified periodogram
        lombscargle: Lomb-Scargle periodogram for unevenly sampled data
        Notes
        -----
        An appropriate amount of overlap will depend on the choice of window
        and on your requirements. For the default 'hanning' window an
        overlap of 50% is a reasonable trade off between accurately estimating
        the signal power, while not over counting any of the data. Narrower
        windows may require a larger overlap.
        If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.
        .. versionadded:: 0.12.0
        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
        estimation of power spectra: A method based on time averaging
        over short, modified periodograms", IEEE Trans. Audio
        Electroacoust. vol. 15, pp. 70-73, 1967.
        .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
        Biometrika, vol. 37, pp. 1-16, 1950.
        Examples
        --------
        >>> from scipy import signal
        >>> import matplotlib.pyplot as plt
        Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
        0.001 V**2/Hz of white noise sampled at 10 kHz.
        >>> fs = 10e3
        >>> N = 1e5
        >>> amp = 2*np.sqrt(2)
        >>> freq = 1234.0
        >>> noise_power = 0.001 * fs / 2
        >>> time = np.arange(N) / fs
        >>> x = amp*np.sin(2*np.pi*freq*time)
        >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        Compute and plot the power spectral density.
        >>> f, Pxx_den = signal.welch(x, fs, nperseg=1024)
        >>> plt.semilogy(f, Pxx_den)
        >>> plt.ylim([0.5e-3, 1])
        >>> plt.xlabel('frequency [Hz]')
        >>> plt.ylabel('PSD [V**2/Hz]')
        >>> plt.show()
        If we average the last half of the spectral density, to exclude the
        peak, we can recover the noise power on the signal.
        >>> np.mean(Pxx_den[256:])
        0.0009924865443739191
        Now compute and plot the power spectrum.
        >>> f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
        >>> plt.figure()
        >>> plt.semilogy(f, np.sqrt(Pxx_spec))
        >>> plt.xlabel('frequency [Hz]')
        >>> plt.ylabel('Linear spectrum [V RMS]')
        >>> plt.show()
        The peak height in the power spectrum is an estimate of the RMS amplitude.
        >>> np.sqrt(Pxx_spec.max())
        2.0077340678640727
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape)
        
        if y.size == 0:
            return np.empty(y.shape), np.empty(y.shape)
        
        if y.size != x.size:
            raise ValueError('Signals x and y are not of the same length.')
    
        if axis != -1:
            x = np.rollaxis(x, axis, len(x.shape))
            y = np.rollaxis(y, axis, len(y.shape))
    
        if x.shape[-1] < nperseg:
            warnings.warn('nperseg = %d, is greater than x.shape[%d] = %d, using '
            'nperseg = x.shape[%d]'
            % (nperseg, axis, x.shape[axis], axis))
            nperseg = x.shape[-1]
    
        if y.shape[-1] < nperseg:
            warnings.warn('nperseg = %d, is greater than y.shape[%d] = %d, using '
                            'nperseg = y.shape[%d]'
                            % (nperseg, axis, y.shape[axis], axis))
            nperseg = y.shape[-1]
        
        if zeropad:
            winlength = 2 * nperseg
        else:
            winlength = nperseg
    
        #if isinstance(window, string_types) or type(window) is tuple:
        if isinstance(window, str) or type(window) is tuple:
            win = get_window(window, winlength)
        else:
            win = np.asarray(window)
            if len(win.shape) != 1:
                raise ValueError('window must be 1-D')
            if win.shape[0] > x.shape[-1]:
                raise ValueError('window is longer than x.')
            if win.shape[0] > y.shape[-1]:
                raise ValueError('window is longer than y.')
            nperseg = win.shape[0]
    
        if scaling == 'density':
            scale = 1.0 / (fs * (win*win).sum())
        elif scaling == 'spectrum':
            scale = 1.0 / win.sum()**2
        else:
            raise ValueError('Unknown scaling: %r' % scaling)
    
        if noverlap is None:
            #noverlap = nperseg // 2
            noverlap = 0
        elif noverlap >= nperseg:
            raise ValueError('noverlap must be less than nperseg.')
    
        if not hasattr(detrend, '__call__'):
            detrend_func = lambda seg: signaltools.detrend(seg, type=detrend)
        elif axis != -1:
            # Wrap this function so that it receives a shape that it could
            # reasonably expect to receive.
            def detrend_func(seg):
                seg = np.rollaxis(seg, -1, axis)
                seg = detrend(seg)
                return np.rollaxis(seg, axis, len(seg.shape))
        else:
            detrend_func = detrend
    
        step = nperseg - int(noverlap)
        indices = np.arange(0, x.shape[-1]-nperseg+1, step)
        
    
        if np.isrealobj(x) and np.isrealobj(y) and return_onesided:
             
            for k, ind in enumerate(indices):
                x_dt = detrend_func(x[..., ind:ind+nperseg])
                y_dt = detrend_func(y[..., ind:ind+nperseg])
                    
                if zeropad:
                    x_tmp = np.zeros((2 * len(x_dt)))
                    x_tmp[0:len(x_dt)] = x_dt
                    x_dt = x_tmp
                    y_tmp = np.zeros((2 * len(y_dt)))
                    y_tmp[0:len(y_dt)] = y_dt
                    y_dt = y_tmp
                
                xft = 2 * np.fft.rfft(x_dt*win)
                yft = np.fft.rfft(y_dt*win)
    
                if k == 0:
                    Pxy = xft * np.conj(yft)
    
                else:  #Averaging
                    Pxy *= k/(k+1.0)
                    Pxy += (xft * np.conj(yft)) / (k+1.0)
           
            Pxy *= scale
            f = np.fft.rfftfreq(nperseg, d = (1/fs))
        else:
            for k, ind in enumerate(indices):
                x_dt = detrend_func(x[..., ind:ind+nperseg])
                y_dt = detrend_func(y[..., ind:ind+nperseg])
                
                if zeropad:
                    x_tmp = np.zeros((2 * len(x_dt)))
                    x_tmp[0:len(x_dt)] = x_dt
                    x_dt = x_tmp
                    y_tmp = np.zeros((2 * len(y_dt)))
                    y_tmp[0:len(y_dt)] = y_dt
                    y_dt = y_tmp
                    
                xft = np.fft.fft(x_dt*win)
                yft = np.fft.fft(y_dt*win)
    
                if k == 0:
                    Pxy = xft * np.conj(yft)
                else:  #Averaging
                    Pxy *= k/(k+1.0)
                    Pxy += (xft * np.conj(yft)) / (k+1.0)
            Pxy *= scale
            f = np.fft.fftfreq(nperseg, d = (1/fs))
    
        if axis != -1:
            Pxy = np.rollaxis(Pxy, -1, axis)
        return f, Pxy
        
        
    
    
    ###############################################################################
    ####### Averaged FFT of real data #######################
    ###############################################################################
    
    def FFT_Average(self, x, fs=1.0, window='boxcar', nperseg=1024, noverlap=None, 
    detrend='constant', zeropad = False, return_onesided=True, axis=-1):
        
        """
        Estimate the averaged FFT of a signal x.
        The routine computes an estimate of the FFT
        by dividing the data into overlapping segments, computing the FFT and 
        averaging them.
        Parameters
        ----------
        x : array_like
        Time series of measurement values
        fs : float, optional
        Sampling frequency of the `x` time series in units of Hz. Defaults
        to 1.0.
        window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length will be used for nperseg.
        Defaults to 'none'.
        nperseg : int, optional
        Length of each segment. Defaults to 1024.
        noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg / 2``. Defaults to None.
        nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is `nperseg`. Defaults to None.
        detrend : str or function, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`. If it is a
        function, it takes a segment and returns a detrended segment.
        Defaults to 'constant'.
        return_onesided : bool, optional
        If True, return a one-sided spectrum. If False return
        a two-sided spectrum. Note that for complex data, a two-sided
        spectrum is always returned.
        axis : int, optional
        Axis along which the periodogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
    
        Returns
        -------
        f : ndarray
        Array of sample frequencies.
        FTx : ndarray
        DFT of x.
    
        Notes
        -----
        An appropriate amount of overlap will depend on the choice of window
        and on your requirements. For the default 'hanning' window an
        overlap of 50% is a reasonable trade off between accurately estimating
        the signal power, while not over counting any of the data. Narrower
        windows may require a larger overlap.
        If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.
        .. versionadded:: 0.12.0
    
        Examples
        --------
        >>> from scipy import signal
        >>> import matplotlib.pyplot as plt
        Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
        0.001 V**2/Hz of white noise sampled at 10 kHz.
        >>> fs = 10e3
        >>> N = 1e5
        >>> amp = 2*np.sqrt(2)
        >>> freq = 1234.0
        >>> noise_power = 0.001 * fs / 2
        >>> time = np.arange(N) / fs
        >>> x = amp*np.sin(2*np.pi*freq*time)
        >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        Compute and plot the power spectral density.
        >>> f, Pxx_den = signal.welch(x, fs, nperseg=1024)
        >>> plt.semilogy(f, Pxx_den)
        >>> plt.ylim([0.5e-3, 1])
        >>> plt.xlabel('frequency [Hz]')
        >>> plt.ylabel('PSD [V**2/Hz]')
        >>> plt.show()
        If we average the last half of the spectral density, to exclude the
        peak, we can recover the noise power on the signal.
        >>> np.mean(Pxx_den[256:])
        0.0009924865443739191
        Now compute and plot the power spectrum.
        >>> f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
        >>> plt.figure()
        >>> plt.semilogy(f, np.sqrt(Pxx_spec))
        >>> plt.xlabel('frequency [Hz]')
        >>> plt.ylabel('Linear spectrum [V RMS]')
        >>> plt.show()
        The peak height in the power spectrum is an estimate of the RMS amplitude.
        >>> np.sqrt(Pxx_spec.max())
        2.0077340678640727
        """
        x = np.asarray(x)
        
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape)
    
        if axis != -1:
            x = np.rollaxis(x, axis, len(x.shape))
    
        if x.shape[-1] < nperseg:
            warnings.warn('nperseg = %d, is greater than x.shape[%d] = %d, using '
            'nperseg = x.shape[%d]'
            % (nperseg, axis, x.shape[axis], axis))
            nperseg = x.shape[-1]
        
        if zeropad:
            winlength = 2 * nperseg
        else:
            winlength = nperseg
    
        #if isinstance(window, string_types) or type(window) is tuple:
        if isinstance(window, str) or type(window) is tuple:
            win = get_window(window, winlength)
        else:
            win = np.asarray(window)
            if len(win.shape) != 1:
                raise ValueError('window must be 1-D')
            if win.shape[0] > x.shape[-1]:
                raise ValueError('window is longer than x.')
    
        if noverlap is None:
            #noverlap = nperseg // 2
            noverlap = 0
        elif noverlap >= nperseg:
            raise ValueError('noverlap must be less than nperseg.')
    
        if not hasattr(detrend, '__call__'):
            detrend_func = lambda seg: signaltools.detrend(seg, type=detrend)
        elif axis != -1:
            # Wrap this function so that it receives a shape that it could
            # reasonably expect to receive.
            def detrend_func(seg):
                seg = np.rollaxis(seg, -1, axis)
                seg = detrend(seg)
                return np.rollaxis(seg, axis, len(seg.shape))
        else:
            detrend_func = detrend
    
        step = nperseg - noverlap
        indices = np.arange(0, x.shape[-1]-nperseg+1, step)
        
    
        if np.isrealobj(x) and return_onesided:
             
            for k, ind in enumerate(indices):
                x_dt = detrend_func(x[..., ind:ind+nperseg])
                    
                if zeropad:
                    x_tmp = np.zeros((2 * len(x_dt)))
                    x_tmp[0:len(x_dt)] = x_dt
                    x_dt = x_tmp
    
                xft = np.fft.rfft(x_dt*win)
    
                if k == 0:
                    FTx = xft
    
                else:  #Averaging
                    print('average count = ', k)
                    FTx *= k/(k+1.0)
                    FTx += (xft) / (k+1.0)
           
            f = np.fft.rfftfreq(nperseg, d = (1/fs))
            
        else:
            for k, ind in enumerate(indices):
                x_dt = detrend_func(x[..., ind:ind+nperseg])
                    
                if zeropad:
                    x_tmp = np.zeros((2 * len(x_dt)))
                    x_tmp[0:len(x_dt)] = x_dt
                    x_dt = x_tmp
                xft = np.fft.fft(x_dt*win)
    
                if k == 0:
                    FTx = xft
    
                else:  #Averaging
                    print('average count = ', k)
                    FTx *= k/(k+1.0)
                    FTx += (xft) / (k+1.0)
    
            f = np.fft.rfftfreq(nperseg, d = (1/fs))
    
        if axis != -1:
            FTx = np.rollaxis(FTx, -1, axis)
        return f, FTx
        
    
    ###############################################################################
    ####### Exponential window function #######################
    ###############################################################################
    
    def Exp_Win(self, x, fin_val):
    
        '''
        Applies an exponential window function to a signal x.
        The exponential window function is composed as:
        win[ii]=exp(a * ii) with 
        a = (ln(f) / (n-1)) where n = length of x
        
        return:
        windowed signal x_win
        '''
        x = np.asarray(x)
        
        n = len(x)
        #print('n= ', n)
        a = np.log(fin_val) / (n-1)
    
        win = np.arange(n)
        win =  win * a
        win = np.exp(win)
        
        x_win = x * win
        
        return x_win, a
    
   
   
   
   
    def compute_modal_params(self, max_model_order, multiprocessing):
                        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
            
        assert self.state[0]

        self.max_model_order=max_model_order
        factor_a = self.factor_a
        
        print('Computing modal parameters...')
        
        #ref_channels = sorted(self.prep_data.ref_channels)
        #roving_channels = self.prep_data.roving_channels
        #measurement = self.prep_data.measurement
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels
        selected_omega_vector = self.selected_omega_vector
        num_omega = self.num_omega
        spectrum_tensor = self.spectrum_tensor

        sampling_rate = self.prep_data.sampling_rate
        Delta_t = 1 / sampling_rate

        ###############################################################################
        ######################### Go through model orders #############################
        ###############################################################################
                
        # Compute the modal solutions for all model orders
        
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros((num_analised_channels, max_model_order, max_model_order),dtype=complex)
                          
        for this_model_order in range(max_model_order+1):
            
            # minimal model order should be 2 !!!
            
            if this_model_order >> 1:
                print("this_model_order: ", this_model_order)
        
                # Create matrices X_0 and Y_0
            
                X_o = np.zeros((num_omega, (this_model_order+1)), dtype = complex)
                
                for jj in range(this_model_order+1):                   
                    X_o[:,jj] = selected_omega_vector * jj * Delta_t * 1j
                            
                X_o = np.exp(X_o)
                
                for n_o in range(num_analised_channels):
                    Y_o = np.zeros((num_omega, ((this_model_order+1)*num_ref_channels)), dtype= complex)
                    
                    for kk in range(num_omega):
                        this_Syy = spectrum_tensor[n_o,:,kk]
                        
                        for ll in range(this_model_order+1):
                            Y_o[kk,(ll*num_ref_channels):((ll+1)*num_ref_channels)] = X_o[kk,ll]*this_Syy.T
                            
                    X_o_H = (np.conj(X_o)).T        
                    Y_o_H = (np.conj(Y_o)).T
                    R_o = np.real(np.dot(X_o_H, X_o))
                    R_o_inv = np.linalg.inv(R_o)
                    S_o = np.real(np.dot(X_o_H, Y_o))
                    T_o = np.real(np.dot(Y_o_H, Y_o))
                    
                    if n_o == 0:
                        M = 2 * (T_o - (np.dot(np.dot(S_o.T, R_o_inv), S_o)))
                        R_o_rows = R_o_inv.shape[0]
                        R_o_cols = R_o_inv.shape[1]
                        R_o_inv_tensor = np.zeros((R_o_rows, R_o_cols, num_analised_channels))
                        R_o_inv_tensor[:,:,n_o] = R_o_inv
                        S_o_rows = S_o.shape[0]
                        S_o_cols = S_o.shape[1]
                        S_o_tensor = np.zeros((S_o_rows, S_o_cols, num_analised_channels))
                        S_o_tensor[:,:,n_o] = S_o
                        
                    else:
                        
                        M = M + 2*(T_o - (np.dot(np.dot(S_o.T, R_o_inv), S_o)))
                        R_o_inv_tensor[:,:,n_o] = R_o_inv
                        S_o_tensor[:,:,n_o] = S_o
                        
                # Compute alpha, beta
        
                M_ba = M[num_ref_channels :, : num_ref_channels]
                M_bb = M[num_ref_channels :, num_ref_channels :]
                alpha_b = -np.dot(np.linalg.inv(M_bb), M_ba)
                alpha = np.eye(num_ref_channels)
                alpha = np.concatenate((alpha, alpha_b), axis=0)
                        
                for n_o in range(num_analised_channels):
                    R_o_inv = R_o_inv_tensor[:,:,n_o]
                    S_o = S_o_tensor[:,:,n_o]
                    beta_o = - np.dot(R_o_inv, (np.dot(S_o, alpha)))
                    
                    if n_o == 0:
                        beta_o_rows = beta_o.shape[0]
                        beta_o_cols = beta_o.shape[1]
                        beta_o_tensor = np.zeros((beta_o_rows, beta_o_cols, num_analised_channels))
                        beta_o_tensor[:,:,n_o] = beta_o
                        
                    else:
                        beta_o_tensor[:,:,n_o] = beta_o
                        
                # Create matrices A_c and C_c
                
                A_p = alpha[(this_model_order*num_ref_channels):((this_model_order+1)*num_ref_channels),:]
                A_p_inv = np.linalg.inv(A_p)
                B_p = beta_o_tensor[this_model_order,:,:]
                B_p = np.transpose(B_p)
                size_A_c = this_model_order*num_ref_channels
                A_c = np.zeros((size_A_c, size_A_c))
                C_c = np.zeros((B_p.shape[0], size_A_c))
                
                for p_i in range(this_model_order):
                    A_p_i = alpha[((this_model_order - p_i - 1) * num_ref_channels):((this_model_order - p_i) * num_ref_channels),:]
                    this_A_c_block = - (np.dot(A_p_inv, A_p_i))
                    A_c[0:num_ref_channels, (p_i * num_ref_channels):((p_i+1) * num_ref_channels)] = this_A_c_block
                    B_p_i = beta_o_tensor[(this_model_order - p_i - 1),:,:]
                    B_p_i = np.transpose(B_p_i)
                    this_C_c_block = B_p_i - (np.dot(B_p, this_A_c_block))
                    C_c[:, (p_i * num_ref_channels):((p_i+1) * num_ref_channels)] = this_C_c_block
                    
                A_c_rest = np.eye((this_model_order-1) * num_ref_channels)
                A_c[num_ref_channels:,0:((this_model_order-1) * num_ref_channels)] = A_c_rest
                        
                # Compute modal parameters from matrices A_c and C_c
                   
                lambda_k = np.array([], dtype=np.complex)                    
                    
                eigenvalues_paired, eigenvectors_paired = np.linalg.eig(A_c)
                eigenvectors_single = []
                eigenvalues_single = []
                    
                eigenvectors_single,eigenvalues_single = \
                    self.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
                eigenvectors_single = np.array(eigenvectors_single)
                eigenvalues_single = np.array(eigenvalues_single)
                
                                
                #print('dim. of eigenvectors_paired = ', eigenvectors_paired.shape)
                #print('dim. of eigenvectors_single = ', eigenvectors_single.shape)
                #print('dim. of C_c = ', C_c.shape)

                
                current_frequencies = np.zeros((1,max_model_order))
                current_damping = np.zeros((1,max_model_order))
                current_mode_shapes = np.zeros((num_analised_channels,max_model_order), dtype=complex)
                                 
                for jj in range(len(eigenvalues_single)):
                    k = eigenvalues_single[jj]        
                    lambda_k = np.log(complex(k)) * sampling_rate
                    freq_j = np.abs(lambda_k) / (2*np.pi)     
        
                    #damping without correction if no exponential window was applied
                    '''
                    damping_j = np.real(lambda_k)/np.abs(lambda_k) * (-100) 
                    '''
                    #damping with correction if exponential window was applied to corr. fct.
                    
                    damping_j = (np.real(lambda_k)/np.abs(lambda_k) - factor_a * (sampling_rate) / (freq_j * 2*np.pi)) * (-100)        
                    #damping_j = (np.real(lambda_k)/np.abs(lambda_k) + factor_a * (freq_j * 2*np.pi)) * (-100)        
                    #damping_j = (np.real(lambda_k)/np.abs(lambda_k) - factor_a ) * (-100)        
                    
                            
                    mode_shapes_j = np.dot(C_c[:,:], eigenvectors_single[:,jj])
                    mode_shapes_j = mode_shapes_j.reshape((num_analised_channels,1))
                    # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                    #mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))                
                    
                    current_frequencies[0:1,jj:jj+1] = freq_j
                    current_damping[0:1,jj:jj+1] = damping_j
                    current_mode_shapes[:,jj:jj+1] = mode_shapes_j
                    
                modal_frequencies[(this_model_order-1),:] = current_frequencies
                modal_damping[(this_model_order-1),:] = current_damping
                mode_shapes[:,:,(this_model_order-1)] = current_mode_shapes
        
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
            
        self.state[1]=True

################################################################################################################################        
    '''    
    def init_child_process(self, refs_memory_, measurement_memory_, toeplitz_memory_):
        #make the  memory arrays available to the child processes
        global refs_memory
        refs_memory = refs_memory_
        
        global measurement_memory
        measurement_memory = measurement_memory_   
        
        global toeplitz_memory
        toeplitz_memory = toeplitz_memory_
    '''
        
    def multiprocess_evd(self, a, truncation_orders, return_dict):
        
        for truncation_order in truncation_orders:
            eigenvalues_paired, eigenvectors_paired = np.linalg.eig(a[0:truncation_order+1, 0:truncation_order+1])
    
            eigenvectors_single,eigenvalues_single = \
                    PLSCF.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            return_dict[truncation_order] = (eigenvalues_single, eigenvectors_single)
        
        return
                    
    @staticmethod
    def remove_conjugates_new (vectors, values):
        '''
        removes conjugates and marks the vectors which appear in pairs
        
        vectors.shape = [order+1, order+1]
        values.shape = [order+1,1]
        '''
        num_val=vectors.shape[1]
        conj_indices=deque()
        
        for i in range(num_val):
            this_vec=vectors[:,i]
            this_conj_vec = np.conj(this_vec)
            this_val=values[i]
            this_conj_val = np.conj(this_val)
            if this_val == this_conj_val: #remove real eigenvalues
                continue
            for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
                if vectors[0,j] == this_conj_vec[0] and \
                   vectors[-1,j] == this_conj_vec[-1] and \
                   values[j] == this_conj_val:
                    # saves computation time this function gets called many times and 
                    #numpy's np.all() function causes a lot of computation time
                    conj_indices.append(i)
                    break
        conj_indices=list(conj_indices)
        vector = vectors[:,conj_indices]
        value = values[conj_indices]

        return vector,value
    
    @staticmethod
    def integrate_quantities(vector, accel_channels, velo_channels, omega):
        # input quantities = [a, v, d]
        # output quantities = [d, d, d]
        # converts amplitude and phase
        #                     phase + 180; magn / omega^2
        
        vector[accel_channels] *= -1       / (omega ** 2)
        #                    phase + 90; magn / omega
        vector[velo_channels] *=  1j        / omega
        
        return vector   
    
    def save_state(self, fname):
        
        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        #             0             1     
        #self.state= [Half_spectra, Modal Par.
        self.state  =[False,    False]
        out_dict={'self.state':self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time']=self.start_time
        #out_dict['self.prep_data']=self.prep_data
        if self.state[0]:# spectral tensor
            out_dict['self.begin_frequency'] = self.begin_frequency
            out_dict['self.end_frequency'] = self.end_frequency
            out_dict['self.nperseg'] = self.nperseg
            out_dict['self.selected_omega_vector'] = self.selected_omega_vector
            out_dict['self.num_omega'] = self.num_omega
            out_dict['self.spectrum_tensor'] = self.spectrum_tensor
        if self.state[1]:# modal params
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            
        np.savez_compressed(fname, **out_dict)
        
    @classmethod 
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname)    
        #             0         1           2          
        #self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state= list(in_dict['self.state'])
        else:
            return
        
        for this_state, state_string in zip(state, ['Sprectral Tensor Built',
                                                    'Modal Parameters Computed',
                                                    ]):
            if this_state: print(state_string)
        
        assert isinstance(prep_data, PreprocessData)
        setup_name= str(in_dict['self.setup_name'].item())
        start_time=in_dict['self.start_time'].item()
        assert setup_name == prep_data.setup_name
        start_time = prep_data.start_time
        
        assert start_time == prep_data.start_time
        #prep_data = in_dict['self.prep_data'].item()
        pLSCF_object = cls(prep_data)
        pLSCF_object.state = state
        if state[0]:# spectral tensor
            pLSCF_object.begin_frequency = in_dict['self.begin_frequency']
            pLSCF_object.end_frequency = int(in_dict['self.end_frequency'])
            pLSCF_object.nperseg = int(in_dict['self.nperseg'])
            pLSCF_object.selected_omega_vector= in_dict['self.selected_omega_vector']
            pLSCF_object.num_omega = in_dict['self.num_omega']
            pLSCF_object.spectrum_tensor = in_dict['self.spectrum_tensor']
        if state[2]:# modal params
            pLSCF_object.max_model_order = int(in_dict['self.max_model_order'])
            pLSCF_object.modal_frequencies = in_dict['self.modal_frequencies']
            pLSCF_object.modal_damping = in_dict['self.modal_damping']
            pLSCF_object.mode_shapes = in_dict['self.mode_shapes']
        
        return pLSCF_object
     
    @staticmethod
    def rescale_mode_shape(modeshape):
        #scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape


    
def main():
    pass

if __name__ =='__main__':
    main()
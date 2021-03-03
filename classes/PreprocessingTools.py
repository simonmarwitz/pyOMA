# 
# -*- coding: utf-8 -*-
'''
Based on previous works by Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015

TODO
 * correct linear,.... offsets as well
 * implement filter functions
 * implement loading of different filetypes ascii, lvm, ...
 * currently loading geometry, etc. files will overwrite existing assignments implement "load and append"
 * implement fft, psd, covariance (auto,cross), coherence, frf (to be used with a preprocessor gui)
 * implement integration
 * implement windowing functions
 
'''

import numpy as np
from scipy import signal
import scipy.signal.ltisys
import scipy.integrate
import os
import csv
import sys
import datetime
import time

import multiprocessing as mp
import ctypes as c
import warnings

import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import warnings

def nearly_equal(a,b,sig_fig=5):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )
class GeometryProcessor(object):
    '''
        conventions:

        * chan_dofs=[(chan, node, (x_amplif,y_amplif,z_amplif)),...]

        * channels = 0 ... #, starting at channel 0, should be a complete sequence

        * nodes = 1 ... #, starting at node 1, can be a sequence with missing entries

        * lines = [(node_start, node_end),...], unordered

        * master_slaves = [(node_master, x_master, y_master, z_master, 
                            node_slave, x_slave, y_slave, z_slave),...], unordered

        (Load it into VisualizeGui for browsing the dict's structure!)
        (If you get an error bad_magic_number or segmentation fault you 
        cannot open the shelves on the computer you are currently using
        It has been tested to work on arwen.bauing.uni-weimar.de, but this
        depends on were the shelve file was created and may not always work)
    '''
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.lines = []
        self.master_slaves = []
        
    @staticmethod 
    def nodes_loader(filename):
        '''
        nodes file uses one header line
        tab-separated file
        node is treated as a string
        x,y,z are treated as floats (in scientific format)
        '''
        nodes = {}
        with open(filename, 'r') as f:
            f.__next__()
            for line1 in csv.reader(f, delimiter='\t', skipinitialspace=True):
                line=[]
                for val in line1:
                    if not val: continue
                    line+=val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                node, x, y, z =  \
                    [float(line[i]) if i >=1 else line[i].strip(' ') for i in range(4) ]  # cut trailing empty columns
                nodes[node]= [x, y, z]
        return nodes
    
    @staticmethod     
    def lines_loader(filename):
        '''
        lines file uses one header line
        tab-separated file
        nodenames are treated as strings
        '''
        lines = []
        with open(filename, 'r') as f:
            f.__next__()
            for line1 in csv.reader(f, delimiter='\t', skipinitialspace=True):
                line=[]
                for val in line1:
                    if not val: continue
                    line+=val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                node_start, node_end = \
                    [line[i] for i in range(2)]# cut trailing empty columns
                lines.append((node_start, node_end))
        return lines
    
    @staticmethod 
    def master_slaves_loader(filename):
        '''
        lines file uses one header line
        tab-separated file
        nodenames are treated as strings
        amplification factors are treated as floats
        '''
        master_slaves = []
        with open(filename, 'r') as f:
            f.__next__()
            reader = csv.reader(f, delimiter='\t', skipinitialspace=True)
            for line1 in reader:
                line=[]
                for val in line1:
                    if not val: continue
                    line+=val.split()
                if not line:
                    continue                    
                if line[0].startswith('#'):
                    break
                i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl = \
                        [float(line[i]) if i not in [0,4] else line[i].strip(' ') for i in range(8) ]
                master_slaves.append(
                    (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl))
        return master_slaves
    
    @classmethod     
    def load_geometry(cls, nodes_file, lines_file=None, master_slaves_file=None):
        '''
        inititalizes a geometry object, to be passed along in the preprocessed data object
        '''    
        
        geometry_data = cls()
        
        nodes = geometry_data.nodes_loader(nodes_file)
        geometry_data.add_nodes(nodes)
        
        if lines_file is not None:
            lines = geometry_data.lines_loader(lines_file)
            geometry_data.add_lines(lines)       
            
        if master_slaves_file is not None:
            master_slaves = geometry_data.master_slaves_loader(master_slaves_file)
            geometry_data.add_master_slaves(master_slaves)
        
        return geometry_data 

    def add_nodes(self, nodes):
        for item in nodes.items():
            try:
                self.add_node(*item)
            except:
                print('Something was wrong while adding node {}. Continuing!'.format(item))
                continue

    def add_node(self, node_name, coordinate_list):
        node_name = str(node_name)
        if node_name in self.nodes.keys():
            print('Node {} is already defined. Overwriting.'.format(node_name))
            
        if not isinstance(coordinate_list, (list,tuple)):
            raise RuntimeError('Coordinates must be provided as (x,y,z) tuples/lists.')
        if len(coordinate_list) != 3:
            raise RuntimeError('Coordinates must be provided as (x,y,z) tuples/lists.')
        
        try:
            node_name = str(node_name)
            coordinate_list= list(coordinate_list)
            for i in range(3):
                coordinate_list[i] = float(coordinate_list[i])
        except ValueError:
            raise RuntimeError('Coordinate {} at position {} could not be converted to float.'.format(coordinate_list[i],i))
        except:
            raise
        
        self.nodes[node_name]=tuple(coordinate_list)
    
    def take_node(self, node_name):
        if not node_name in self.nodes:
            print('Node not defined. Exiting')
            return
        
        while True: # check if any line is connected to this node
            for j in range(len(self.lines)):
                line = self.lines[j]
                if node_name in line:
                    del self.lines[j]
                    break
            else:
                break

        while True: # check if this node is a master or slave for another node
            for j, master_slave in enumerate(self.master_slaves):
                if node_name == master_slave[0] or node_name == master_slave[4]:
                    m = master_slave
                    del self.master_slaves[j]
                    break
            else:
                break
        del self.nodes[node_name]
        
        print('Node {} removed.'.format(node_name))

    def add_lines(self, lines):
        
        for line in lines:
            try:
                self.add_line(line)
            except:
                print('Something was wrong while adding line {}. Continuing!'.format(line))
                continue
            

    def add_line(self, line):
        if not isinstance(line, (list,tuple)):
            raise RuntimeError('Line has to be provided in format (start_node, end_node).')
        if len(line) != 2:
            raise RuntimeError('Line has to be provided in format (start_node, end_node).')
        
        line = [str(line[0]), str(line[1])]        
        if line[0] not in self.nodes or line[1] not in self.nodes:
            print('One of the end-nodes of line {} not defined!'.format(line))
        else:
            for line_ in self.lines:
                if line_[0] == line[0] and line_[1] == line[1]:
                    print('Line {} was defined, already.'.format(line))
            else:
                self.lines.append(line)
            
    def take_line(self, line=None, line_ind=None):
        assert line is None or line_ind is None
        
        if line is not None:
            for line_ind in range(len(self.lines)):
                line_ = self.lines[line_ind]
                if line[0]==line_[0] and line[1] == line_[1]:
                    break
            else:
                print('Line {} was not found.'.format(line))
                return
        del self.lines[line_ind]
        print('Line {} at index {} removed.'.format(line, line_ind))
                        
    def add_master_slaves(self,master_slaves):
        for ms in master_slaves:
            try:
                self.add_master_slave(ms)
            except:
                print('Something was wrong while adding master-slave-definition {}. Continuing!'.format(ms))
                continue
            
    def add_master_slave(self, ms):
        if not isinstance(ms, (list,tuple)):
            raise RuntimeError('master slave definition has to be provided in format (start_node, end_node).')
        if len(ms) != 8:
            raise RuntimeError('master slave definition has to be provided in format (master_node, x_ampli, y_ampli, z_ampli, slave_node, x_ampli, y_ampli, z_ampli).')
        ms = (str(ms[0]), float(ms[1]), float(ms[2]), float(ms[3]), str(ms[4]),float(ms[5]), float(ms[6]), float(ms[7])) 
        if ms[0] not in self.nodes or ms[4] not in self.nodes:
            print('One of the nodes of master slave definition {} not defined!'.format(ms))
        else:
            for ms_ in self.master_slaves:
                b = False
                for i in range(8):
                    b = b and ms_[i] == ms[i]
                if b:
                    print('master slave definition {} was defined, already.'.format(ms))
            else:
                self.master_slaves.append(ms)
                
    def take_master_slave(self, ms=None, ms_ind=None):
        assert ms is None or ms_ind is None
        
        if ms is not None:
            for ms_ind in range(len(self.master_slaves)):
                ms_ = self.master_slaves[ms_ind]
                b = False
                for i in range(8):
                    b = b and ms_[i] == ms[i]
                if b:
                    break
            else:
                print('master slave definition {} was not found.'.format(ms))
                return 
              
        del self.master_slaves[ms_ind]
        print('master slave definition {} at index {} removed.'.format(ms, ms_ind))
    
    def rescale_geometry(self, factor):
        pass
    
class PreprocessData(object):
    '''
    A simple Data PreProcessor
    * load ascii datafiles
    * specify sampling rate, reference channels and roving channels
    * specify geometry, channel-dof-assignments
    * specify channel quantities such as acceleration, velocity, etc 
    * remove channels, cut time histories
    * remove (constant) offsets from time history data
    * decimate time histories
    future:
    * apply several filters
    * calculate fft, psd, covariance, coherence, frf
    * integrate
    * apply windowing functions
    '''
    def __init__(self, measurement, sampling_rate, total_time_steps=None, 
                 #num_channels=None,
                 ref_channels=None,
                 accel_channels=None, velo_channels=None, disp_channels=None,
                 setup_name=None, channel_headers=None, start_time=None, 
                 ft_freq=None, sum_ft=None, F=None, **kwargs):
        
        super().__init__()
        
        assert isinstance(measurement, np.ndarray)
        assert measurement.shape[0] > measurement.shape[1]
        self.measurement = measurement
        self.measurement_filt = measurement
        
        assert isinstance(sampling_rate, (int,float))
        self.sampling_rate = sampling_rate
        
        #added by anil
        if F is not None:
            assert isinstance(F, np.ndarray)
        self.F = F
        
        if total_time_steps is None:
            total_time_steps = measurement.shape[0]
        
        assert  measurement.shape[0] >= total_time_steps

        self.total_time_steps = total_time_steps
        
        if ref_channels is None:
            ref_channels = list(range(measurement.shape[1]))
        self.ref_channels = ref_channels
#         if roving_channels is None:
#             roving_channels = [i for i in range(measurement.shape[1]) if i not in ref_channels]
#         self.roving_channels = roving_channels
        
        self.num_ref_channels = len(self.ref_channels)
#         self.num_roving_channels = len(self.roving_channels)
        self.num_analised_channels = measurement.shape[1]#self.num_ref_channels + self.num_roving_channels
        
        #if num_channels is None:
        #    num_channels = self.num_analised_channels
            
        #assert num_channels <= self.measurement.shape[1]    
        
        #if ((self.num_ref_channels + self.num_roving_channels) > num_channels):
        #        sys.exit('The sum of reference and roving channels is greater than the number of all channels!')
        
        for ref_channel in self.ref_channels:
            if (ref_channel < 0):
                sys.exit('A reference channel number cannot be negative!')
            if (ref_channel > (self.num_analised_channels - 1)):
                sys.exit('A reference channel number cannot be greater than the number of all channels!')
            #for rov_channel in self.roving_channels:
            #    if (rov_channel < 0):
            #        sys.exit('A roving channel number cannot be negative!')
            #    if (rov_channel > (num_channels - 1)):
            #        sys.exit('A roving channel number cannot be greater than the number of all channels!')
            #    if (ref_channel == rov_channel):
            #       sys.exit('Any channel can be either a reference OR a roving channel. Check your definitions!')
        
        if disp_channels is None:
            disp_channels = []
        if velo_channels is None:
            velo_channels = []
        if accel_channels is None:
            #accel_channels = [c for c in self.ref_channels+self.roving_channels if c not in disp_channels or c not in velo_channels]
            accel_channels = [c for c in range(self.num_analised_channels) if c not in disp_channels and c not in velo_channels]
        
        for channel in range(self.num_analised_channels):
            if (channel in accel_channels) + (channel in velo_channels) + (channel in disp_channels) != 1:
                
                warnings.warn('Measuring quantity of channel {} is not defined.'.format(channel))  
            
        self.accel_channels = accel_channels
        self.velo_channels = velo_channels
        self.disp_channels = disp_channels
        
        #print(self.accel_channels,self.velo_channels,self.disp_channels)
        
        if setup_name is None:            
            setup_name = ''
        assert isinstance(setup_name, str)
            
        self.setup_name = setup_name
        
        if channel_headers is not None:
            assert len(channel_headers) == self.num_analised_channels
        else:
            channel_headers=list(range(self.num_analised_channels))
            
        self.channel_headers=channel_headers
        
        if start_time is not None:
            assert isinstance(start_time, datetime.datetime)
        else:
            start_time=datetime.datetime.now()
        self.start_time=start_time
        #print(self.start_time)
        #self.geometry_data = None
        
        self.channel_factors = [1 for channel in range(self.measurement.shape[1])]
        
        self.chan_dofs = []
        
        self.ft_freq = ft_freq
        self.sum_ft = sum_ft
        
        self.tau_max = 0
        
        self.corr_matrix = None
        self.psd_mats = None
        self.freqs = None
        self.s_vals_cf = None
        self.s_vals_psd = None
        
    @classmethod
    def init_from_config(cls, conf_file, meas_file, chan_dofs_file=None, **kwargs):
        '''
        initializes the PreProcessor object with a configuration file
        
        to remove channels at loading time use 'usecols' keyword argument
        if delete_channels are specified, these will be checked against 
        all other channel definitions, which will be adjusted accordingly
        '''
        if not os.path.exists(conf_file):
            raise RuntimeError('Conf File does not exist: {}'.format(conf_file))
        
        with open(conf_file, 'r') as f:
            
            assert f.__next__().strip('\n').strip(' ') == 'Setup Name:'
            name = f. __next__().strip('\n')
            assert f.__next__().strip('\n').strip(' ')== 'Sampling Rate [Hz]:'
            sampling_rate= float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Reference Channels:'
            ref_channels=f.__next__().strip('\n').split(' ')
            if ref_channels:
                ref_channels=[int(val) for val in ref_channels if val.isnumeric()]
            assert f.__next__().strip('\n').strip(' ')== 'Delete Channels:'
            delete_channels=f.__next__().strip('\n ').split(' ')
            if delete_channels:
                delete_channels=[int(val) for val in delete_channels  if val.isnumeric()]
            assert f.__next__().strip('\n').strip(' ')== 'Accel. Channels:'
            accel_channels=f.__next__().strip('\n ').split()
            if accel_channels:
                accel_channels=[int(val) for val in accel_channels]
            assert f.__next__().strip('\n').strip(' ')== 'Velo. Channels:'
            velo_channels = f.__next__().strip('\n ').split()
            if velo_channels:
                velo_channels=[int(val) for val in velo_channels]
            assert f.__next__().strip('\n').strip(' ')== 'Disp. Channels:'
            disp_channels=f.__next__().strip('\n ').split()
            if disp_channels:
                disp_channels=[int(val) for val in disp_channels]
        
        loaded_data   = cls.load_measurement_file(meas_file, **kwargs)
        
        if not isinstance(loaded_data, np.ndarray):
            headers, units, start_time, sample_rate, measurement = loaded_data
        else:
            measurement = loaded_data
            start_time=datetime.datetime.now()
            sample_rate = sampling_rate
            headers = ['Channel_{}'.format(i) for i in range(measurement.shape[1])]
        if not sample_rate == sampling_rate:
            warnings.warn('Sampling Rate from file: {} does not correspond with specified Sampling Rate from configuration {}'.format(sample_rate, sampling_rate))
        #print(headers)
        
                    
        if chan_dofs_file is not None:
            chan_dofs = cls.load_chan_dofs(chan_dofs_file)
        else:
            chan_dofs = None
            
        if chan_dofs is not None:
            for chan_dof in chan_dofs:
                if len(chan_dof)==5:
                    chan = chan_dof[0]
                    chan_name = chan_dof[4]
                    if len(chan_name)==0:
                        continue
                    elif headers[chan] == 'Channel_{}'.format(chan):
                        headers[chan] = chan_name
                    elif headers[chan] != chan_name:
                        print('Different headers for channel {} in measurement file ({}) and in channel-DOF-assignment ({}).'.format(chan, headers[chan], chan_name))
                    else:
                        continue
                    
        #print(delete_channels)
        if delete_channels:
            #delete_channels.sort(reverse=True)
            
            names=['Reference Channels', 'Accel. Channels', 'Velo. Channels', 'Disp. Channels']
            channel_lists=[ref_channels, accel_channels, velo_channels, disp_channels]
            #print(chan_dofs)
            
            num_all_channels = measurement.shape[1]
            #print(chan_dofs, ref_channels, accel_channels, velo_channels,disp_channels, headers)
            new_chan_dofs = []
            new_ref_channels = []
            new_accel_channels = []
            new_velo_channels = []
            new_disp_channels =[]
            new_headers = []
            new_channel = 0
            for channel in range(num_all_channels):
                if channel in delete_channels:
                    print('Now removing Channel {} (no. {})!'.format(headers[channel], channel)) 
                    continue
                else:
                    for chan_dof in chan_dofs:
                        if chan_dof[0] == channel:
                            node, az, elev = chan_dof[1:4]
                            if len(chan_dof)==5:
                                cname = chan_dof[4]
                            else:
                                cname = ''
                            break
                    else:
                        print('Could not find channel in chan_dofs')
                        continue
                    
                    new_chan_dofs.append([new_channel, node, az, elev, cname])
                    if channel in ref_channels:
                        new_ref_channels.append(new_channel)
                    if channel in accel_channels:
                        new_accel_channels.append(new_channel)
                    if channel in velo_channels:
                        new_velo_channels.append(new_channel)
                    if channel in disp_channels:
                        new_disp_channels.append(new_channel)
                    new_headers.append(headers[channel])
                    
                    new_channel += 1
                                
            measurement = np.delete(measurement, delete_channels, axis=1)      
                  
            chan_dofs = new_chan_dofs
            ref_channels = new_ref_channels
            accel_channels = new_accel_channels
            velo_channels = new_velo_channels
            disp_channels = new_disp_channels
            headers = new_headers
            #print(chan_dofs, ref_channels, accel_channels, velo_channels,disp_channels, headers)
            
            
            
            
            
#             channel = measurement.shape[1]
#             #num_channels = measurement.shape[1]
#             while channel >= 0:
#                 
#                 if channel in delete_channels:
#                     # affected lists: ref_channels, accel_channels, velo_channels, disp_channels + chan_dofs
#                     # remove channel from all lists
#                     # decrement all channels higher than channel in all lists
#                     #num_channels -= 1
#                     for channel_list in channel_lists:
#                         if channel in channel_list:
#                             channel_list.remove(channel)
#                             print('Channel {} removed from {} list'.format(channel, names[channel_lists.index(channel_list)]))
#                         for channel_ind in range(len(channel_list)):
#                             if channel_list[channel_ind] > channel:
#                                 channel_list[channel_ind] -= 1 
#                                 
#                     if chan_dofs:
#                         this_num_channels = len(chan_dofs)
#                         chan_dof_ind = 0
#                         while chan_dof_ind < this_num_channels:
#                             if channel==chan_dofs[chan_dof_ind][0]:
#                                 print('Channel-DOF-Assignment {} removed.'.format(chan_dofs[chan_dof_ind]))
#                                 del chan_dofs[chan_dof_ind]
#                                 this_num_channels -= 1
#                             elif channel < chan_dofs[chan_dof_ind][0]:
#                                 chan_dofs[chan_dof_ind][0] -= 1
#                             chan_dof_ind += 1
#                     print('Now removing Channel {} (no. {})!'.format(headers[channel], channel))  
#                     del headers[channel]
#                 channel -= 1
#             #print(chan_dofs)   
#             
#             measurement=np.delete(measurement, delete_channels, axis=1)
        total_time_steps = measurement.shape[0]
        num_channels = measurement.shape[1]
        #roving_channels = [i for i in range(num_channels) if i not in ref_channels]
        if not accel_channels and not velo_channels and not disp_channels:
            accel_channels = [i for i in range(num_channels)]
        #print(measurement.shape, ref_channels)
        #print(measurement)
        prep_data = cls(measurement, sampling_rate, total_time_steps, 
                 #num_channels, 
                 ref_channels, #roving_channels,
                 accel_channels, velo_channels, disp_channels, 
                 channel_headers=headers, start_time=start_time,
                 setup_name = name, **kwargs)
        if chan_dofs:
            prep_data.add_chan_dofs(chan_dofs)
        
        return prep_data
    
    @staticmethod
    def load_chan_dofs(fname):
        '''
        chan_dofs[i] = (chan_num, node_name, az, elev, chan_name)
                    = (int,       str,       float,float, str)
                    
        azimuth angle starting from x axis towards y axis
        elevation defined from x-y plane up
        x: 0.0, 0.0
        y: 90.0, 0.0
        z: 0.0, 90.0
        channels not  present in the file will be removed later
        nodes do not have to, but should exist, as this information is 
        also used for merging multiple setups, which does not rely on 
        any "real" geometry
        '''        
        chan_dofs = []
        with open(fname, 'r') as f:
            f.__next__()
            for line1 in csv.reader(f, delimiter='\t', skipinitialspace=True):
                line=[]
                for val in line1:
                    if not val: continue
                    line+=val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                while len(line)<=5:line.append('')
                chan_num, node, az, elev, chan_name = [line[i].strip(' ') for i in range(5)]
                chan_num, az, elev = int(float(chan_num)), float(az), float(elev)
                #print(chan_num, node, az, elev)
                if node == 'None':
                    node=None
                    #print(None)
                chan_dofs.append([chan_num, node, az, elev, chan_name])
        return chan_dofs
    
    @staticmethod
    def load_measurement_file(fname, ftype='ascii', **kwargs):
        
        if ftype!='ascii':
            raise RuntimeError('Specified filetype: {} not implemented yet!'.format(ftype))
        
        with open(fname, 'r', encoding='iso-8859-1') as f:
            var_num = 1
            headers = ['Delta Start']
            units = []
            i=-1
            while True:
                s = f.__next__()
                i+=1                
                if 'StartTime:' in s:
                    s=s.split('(')[-1]
                    s.strip(')')
                    date, time, ampm = s.split(' ')
                    month,day,year=[int(d) for d in date.split('/')]
                    
                    hour,min,sec=[int(t) for t in time.split(':')]
                    if 'PM' in ampm and hour < 12:
                        hour += 12
                    start_time = datetime.datetime(year,month,day,hour, min,sec)
                elif 'SampleRate' in s:
                    sample_rate = float(s.split(' ')[-1])
                elif 'VarCount' in s:
                    num_vars = int(s.split(' ')[-1])
                elif 'NameLen' in s:
                    for s in s.split(' / '):
                        if 'Name:' in s:
                            while len(headers)<=var_num:
                                headers.append(None)
                            headers[var_num]=s.split(':')[-1].strip(' ')
                        elif 'Unit:' in s:
                            while len(units)<=var_num:
                                units.append(None)
                            units[var_num]=s.split(':')[-1].strip(' ')
                    var_num += 1
                elif '**' in s:
                    break
            assert var_num == num_vars+1
            
            s=f.__next__()
            i+=1
            start_sec = s.split(' ')[0].split('\t')[0]
            start_time += datetime.timedelta(seconds = float(start_sec))
            f.seek(0)
            #print(i, num_vars)
            #print(kwargs)
            measurement=np.loadtxt(f, 
                          dtype=kwargs.get('dtype',float), 
                          comments=kwargs.get('comments','#'), 
                          delimiter=kwargs.get('delimiter',None),
                          converters=kwargs.get('converters',None), 
                          skiprows=kwargs.get('skiprows',i), 
                          usecols=kwargs.get('usecols',None), 
                          unpack=kwargs.get('unpack',False),
                          ndmin=kwargs.get('ndmin',0))
            
            assert measurement.shape [0] > measurement.shape [1]
        #print(measurement[0,:])    
        print(headers, measurement.shape)
            
        return headers, units, start_time, sample_rate, measurement  

        
    def add_chan_dofs(self,chan_dofs):
        '''
        chan_dofs = [ (chan_num, node_name, az, elev, chan_name) ,  ... ]
        '''
        for chan_dof in chan_dofs:
            chan_dof[0]=int(chan_dof[0])
            if chan_dof[1] is not None:
                chan_dof[1]=str(chan_dof[1])
            chan_dof[2]=float(chan_dof[2])
            chan_dof[3]=float(chan_dof[3])
            if len(chan_dof)==4:
                chan_dof.append('')
            self.chan_dofs.append(chan_dof)
        #self.chan_dofs=chan_dofs
        
    def take_chan_dof(self, chan, node, dof):
        
        for j in range(len(self.chan_dofs)):
            if self.chan_dofs[j][0] == chan and \
               self.chan_dofs[j][1] == node  and \
               nearly_equal(self.chan_dofs[j][2][0], dof[0], 3) and \
               nearly_equal(self.chan_dofs[j][2][1], dof[1], 3) and \
               nearly_equal(self.chan_dofs[j][2][2], dof[2], 3):
                del self.chan_dofs[j]
                break
        else:
            if self.chan_dofs:
                print('chandof not found')

    def save_state(self, fname):
        
        #print('fname = ', fname)
        
        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        
        out_dict={}
        # measuremt infos
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.measurement'] = self.measurement
        out_dict['self.sampling_rate'] = self.sampling_rate
        out_dict['self.total_time_steps'] = self.total_time_steps
        out_dict['self.ref_channels'] = self.ref_channels
        #out_dict['self.roving_channels'] = self.roving_channels
        out_dict['self.num_ref_channels'] = self.num_ref_channels
        #out_dict['self.num_roving_channels'] = self.num_roving_channels
        out_dict['self.num_analised_channels'] = self.num_analised_channels
        out_dict['self.accel_channels']=self.accel_channels
        out_dict['self.velo_channels']=self.velo_channels
        out_dict['self.disp_channels']=self.disp_channels
        out_dict['self.chan_dofs']=self.chan_dofs
        out_dict['self.channel_headers'] = self.channel_headers
        out_dict['self.start_time']=self.start_time
        out_dict['self.ft_freq']=self.ft_freq
        out_dict['self.sum_ft']=self.sum_ft
        out_dict['self.tau_max']=self.tau_max
        out_dict['self.corr_matrix'] =self.corr_matrix
        out_dict['self.psd_mats']=self.psd_mats
        out_dict['self.s_vals_cf']=self.s_vals_cf
        out_dict['self.s_vals_psd']=self.s_vals_psd
        
        #out_dict['self.geometry_data'] = self.geometry_data

        np.savez_compressed(fname, **out_dict)
         
    @classmethod
    def load_state(cls, fname):
        
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname, allow_pickle=True)    

        setup_name= str(in_dict['self.setup_name'].item())
        measurement = in_dict['self.measurement']
        sampling_rate = float(in_dict['self.sampling_rate'])
        total_time_steps = int(in_dict['self.total_time_steps'])
        ref_channels = list(in_dict['self.ref_channels'])
        #roving_channels = list(in_dict['self.roving_channels'])
        if in_dict['self.channel_headers'].shape:
            channel_headers = list(in_dict['self.channel_headers'])
        else:
            channel_headers =['' for chan in range(int(in_dict['self.num_analised_channels']))]
        start_time=in_dict['self.start_time'].item()
        
        accel_channels =  list(in_dict['self.accel_channels'])
        velo_channels = list(in_dict['self.velo_channels'])
        disp_channels = list(in_dict['self.disp_channels'])
        
        if 'self.ft_freq' in in_dict:
            ft_freq = in_dict['self.ft_freq']
            if not ft_freq.shape:
                ft_freq = ft_freq.item()
        else:
            ft_freq = None
            
        #ft_freq = in_dict.get('self.ft_freq', None)
        if 'self.sum_ft' in in_dict:
            sum_ft = in_dict['self.sum_ft']
            if not sum_ft.shape:
                sum_ft = sum_ft.item()
        else:
            sum_ft = None
        spectral_values = [None,None,None,None,None]
        for obj_num,name in enumerate(['self.corr_matrix', 'self.psd_mats', 'self.s_vals_cf','self.s_vals_psd', 'self.tau_max']):
            try:
                spectral_values[obj_num] = in_dict[name]
            except Exception as e:
                print(e)
            
        
        #sum_ft = in_dict.get( 'self.sum_ft', None)
        
        preprocessor = cls(measurement, sampling_rate, total_time_steps, 
                 ref_channels=ref_channels, #roving_channels=roving_channels,
                 accel_channels=accel_channels, velo_channels=velo_channels, 
                 disp_channels=disp_channels, setup_name=setup_name,
                 channel_headers=channel_headers, start_time=start_time, 
                 ft_freq=ft_freq, sum_ft = sum_ft)
        
        chan_dofs = [[int(float(chan_dof[0])), str(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3]), str(chan_dof[4] if 5 == len(chan_dof) else '')] for chan_dof in in_dict['self.chan_dofs']]
        preprocessor.add_chan_dofs(chan_dofs)
        
        preprocessor.corr_matrix = spectral_values[0]
        preprocessor.pds_mats = spectral_values[1]
        preprocessor.s_vals_cf = spectral_values[2]
        preprocessor.s_vals_psd = spectral_values[3]
        preprocessor.tau_max = int(spectral_values[4])
        
        assert preprocessor.num_ref_channels == int(in_dict['self.num_ref_channels'])
        #assert preprocessor.num_roving_channels == int(in_dict['self.num_roving_channels'])
        assert preprocessor.num_analised_channels == int(in_dict['self.num_analised_channels'])
        
        #preprocessor.add_geometry_data(in_dict['self.geometry_data'].item())  
        return preprocessor
    
    def filter_data(self, lowpass=None, highpass=None, overwrite=False, order=4, ftype='butter',  RpRs = [None, None], plot_filter=False):
        print('Filtering data in the band: {} .. {}.'.format(highpass, lowpass))
        
        ''' checks '''
        error = 0
        if (highpass is None) and (lowpass is None): error += 1
        ftype_list = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        if not (isinstance(order, int) and (ftype in ftype_list) and (order > 1)): error += 1
        if (ftype=='cheby1' or ftype=='cheby2' or ftype=='ellip') and (RpRs[0]==None or RpRs[1]==None): error += 1
        
        if error > 0: 
            raise RuntimeError('Invalid arguments.') 
            return
        
        #print("Filtering data...")
        
        nyq = self.sampling_rate/2
        
        freqs = []
        if lowpass is not None:
            freqs.append(float(lowpass))
            btype = 'lowpass'
        if highpass is not None:
            freqs.append(float(highpass))
            btype = 'highpass'
        if len(freqs) == 2:
            btype = 'bandpass'
            freqs.sort()

        freqs[:] = [x/nyq for x in freqs]
        
        #print(freqs)
        order = int(order)
        measurement = self.measurement
        
        b, a = signal.iirfilter(order, freqs, rp=RpRs[0], rs=RpRs[1], btype=btype, ftype=ftype)
        self.measurement_filt = signal.filtfilt(b, a, measurement, axis=0, padlen=0)
        if self.F is not None:
            self.F_filt = signal.filtfilt(b, a, self.F, axis=0, padlen=0)
        
        if np.isnan(self.measurement_filt).any():
            RuntimeWarning('Your filtered data contains NaNs. Check your filter settings! Continuing...')
        
        if plot_filter:
            w, h = scipy.signal.freqz(b, a, worN=2000)
            _, f_plot = plt.subplots(2)
            f_plot[0].set_title('Filter data')           
            f_plot[0].plot((nyq / np.pi) * w, abs(h))
            f_plot[0].plot([0, nyq], [np.sqrt(0.5), np.sqrt(0.5)], '--')
            f_plot[0].axvline(nyq, color='green') # cutoff frequency
            f_plot[1].plot(self.measurement[:,1], label='Original signal (Hz)')
            f_plot[1].plot(self.measurement_filt[:,1], label='Filtered signal (Hz)')
            plt.show()

        if overwrite:
            self.measurement = self.measurement_filt
            if self.F is not None:
                self.F = self.F_filt
            
        return self.measurement_filt
            
    def plot_data(self, channels=None, single_channels = False,f_max = None, NFFT = 512, timescale='time'):
        
        t = np.linspace(start=0, stop=self.total_time_steps/self.sampling_rate, num=self.total_time_steps)
        if timescale=='samples':
            t = np.linspace(start=0, stop=self.total_time_steps, num=self.total_time_steps)
        else:
            assert timescale=='time'
        
        if channels is None:
            channels = list(range(self.num_analised_channels))
            
        if single_channels:
            fig, axes = plot.subplots(nrows=len(channels), ncols=2, sharey='col', sharex='col', tight_layout = True)
        else:
            fig, axes = plot.subplots(nrows=2, ncols=1, squeeze=False, tight_layout = True)
            
        for i, channel in enumerate(channels):
            if not single_channels:
                i = 0
            axes[i, 0].plot(t,self.measurement[:,channel], alpha=.5, label=str(channel))
            axes[i, 0].grid(True, axis='y',ls='dotted')
        axes[-1,0].set_xlabel('t [s]')
        axes[-1,0].set_ylabel('')
        if not single_channels:
            axes[0,0].set_xlabel('Time [\si{\second}]')
            axes[0,0].set_ylabel('Magnitude [\si{\metre\per\second\squared}]')
            axes[0,0].set_ylim([-0.15,0.15])
            axes[0,0].set_xlim([0,1800])
            
        
        if single_channels:
            psd_mats, freqs = self.psd_welch(NFFT,False)
            for i,channel in enumerate(channels):
                axes[i, 1].plot(freqs,np.abs(psd_mats[channel,channel,:]), alpha=.5, label=str(channel))
                axes[i,1].grid(True, axis='x',ls='dotted')
            axes[-1,1].set_xlabel('f [Hz]')
            axes[-1,1].set_ylabel('')
            axes[-1,1].set_yscale('log')
            if f_max:
                axes[-1,1].set_xlim((0,f_max))
        else:
            self.plot_svd_spectrum(NFFT, log_scale=True,ax=axes[1,0])
            axes[1,0].set_ylim([-80,0])
            if f_max:
                axes[1,0].set_xlim((0,f_max))
        #for ax in axes.flat:
        #    ax.legend()
        
        return fig
    
    def correct_offset(self, x=None):
        '''
        corrects a constant offset from measurement data
        Eliminates displacement of the measurement data originated by initial tension
        by subtracting the average value of the x first measurements from every
        value
        '''
        print('Correcting offset of measurements.')
        #print(self.measurement.mean(axis=0))
        self.measurement -= self.measurement.mean(axis=0)
        #print(self.measurement.mean(axis=0))
        return
    
        for ii in range(self.measurement.shape[1]):
            tmp = self.measurement[:,ii]
            if x is not None:
                self.measurement[:,ii] = tmp - tmp[0:x].mean(0)
            else:
                self.measurement[:,ii] = tmp - tmp.mean(0)
    
    def precondition_data(self, method='iqr'):
        
        assert method in ['iqr', 'range']
        
        self.correct_offset()
        
        for i in range(self.measurement.shape[1]):
            tmp = self.measurement[:,i]
            if method == 'iqr':
                factor = np.subtract(*np.percentile(tmp, [95, 5]))
            elif method == 'range':
                factor = np.max(tmp) - np.min(tmp)
            self.measurement[:,i] /= factor
            self.channel_factors[i]=factor
        
    
    def decimate_data(self, decimate_factor, highpass=None,  order=8, filter_type='cheby1'):
        if highpass:
            print('Decimating data with factor {} and additional highpass filtering at {}!'.format(decimate_factor, highpass))
        else:
            print('Decimating data with factor {}!'.format(decimate_factor))
        '''
        decimates measurement data
        filter type and order are choosable (order 8 and type cheby1 are standard for scipy signal.decimate function)
        maximum ripple in the passband (rp) and minimum attenuation in the stop band (rs) are modifiable
        '''
        #signal.decimate()
        #input validation
        decimate_factor = abs(decimate_factor)
        order = abs(order)
        ftype_list = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        if not ((isinstance(decimate_factor, int) or isinstance(decimate_factor, int)) and isinstance(order, int) and (filter_type in ftype_list)
                and (decimate_factor > 1) and (order > 1)):
            raise RuntimeError('Invalid arguments.')
            return
        
        
        RpRs = [None, None]
        if filter_type=='cheby1' or filter_type=='cheby2' or filter_type=='ellip':
            RpRs = [0.05, 0.05] #standard for signal.decimate
        
        nyq = self.sampling_rate/2
        
        meas_decimated = self.filter_data(lowpass= nyq*0.8/decimate_factor, highpass=highpass, overwrite=False, order=order, ftype=filter_type,  RpRs = RpRs,  plot_filter=False)
        

        self.sampling_rate /=decimate_factor
        meas_decimated = meas_decimated[slice(None, None, decimate_factor)]
        if self.F is not None:
            F_decimated = self.F_filt[slice(None, None, decimate_factor)]
            self.F = F_decimated
        self.total_time_steps = meas_decimated.shape[0]
        self.measurement = meas_decimated 
    
    def psd_welch(self, n_lines=2048, refs_only=True, window='hamm'):
        '''
        DONE:
        
        * modify to compute one-sided PSD only, to save computation time
        * make possible to pass arguments to signal.csd
        * compute cross-psd of all  channels only with reference channels (i.e. replace 'numdof' with num_analised_channels or ref_channels, respectively)
        
        '''
        
        print("Estimating Correlation Function and Power Spectral Density by Welch's method...")
        
        measurement = self.measurement
        sampling_rate = self.sampling_rate
        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
        else:
            num_ref_channels = num_analised_channels
        
        psd_mats_shape = (num_analised_channels, num_ref_channels, n_lines+1 )      
        psd_mats = np.zeros(psd_mats_shape, dtype=complex)
        
        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                _, Pxy_den = scipy.signal.csd(measurement[:,channel_1],measurement[:,channel_2], 
                                              sampling_rate, nperseg=2*n_lines, window=window, scaling='spectrum', return_onesided=True)    
                Pxy_den *= n_lines
                psd_mats[channel_1, channel_2, :] = Pxy_den
                     
        freqs = np.fft.rfftfreq(2*n_lines , 1/sampling_rate) 
        
        self.psd_mats = psd_mats
        self.freqs = freqs
        self.n_lines = n_lines

        return psd_mats, freqs   
        
        
    def corr_welch(self, tau_max, window='hamming'):
        
        psd_mats, freqs = self.psd_welch(n_lines = tau_max, window=window)
        
        '''
        DONE:
        * compute cross-correlations of all channels only with reference channels (i.e. replace 'numdof' with num_analised_channels or ref_channels, respectively)
        '''
        
        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        corr_mats_shape = (num_analised_channels, num_ref_channels, tau_max)
            
        corr_matrix = np.zeros(corr_mats_shape)
        
        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                this_psd = psd_mats[channel_1, channel_2,:]
                this_corr = np.fft.irfft(this_psd)
                this_corr = this_corr[:tau_max].real
                
                corr_matrix[channel_1, channel_2, :] = this_corr

        self.corr_matrix = corr_matrix
        self.tau_max = tau_max
        return corr_matrix
        
    
    def psd_blackman_tukey(self, tau_max=256, window = 'bartlett'):
        print("Estimating Correlation Function and Power Spectral Density by Blackman-Tukey's method...")
        
        '''
        TO DO:
        * use rfft
        * why was 2*... removed from the amplitude correction?
        * compare with psd_welch
        * check energy in time domain and frequency domain with parsevals theorem 
        * compute cross-psd of all  channels only with reference channels (i.e. replace 'numdof' with num_analised_channels or ref_channels, respectively)
        * compute only one-sided psd (i.e. length only tau_max - 1 or similar)
        * read about the window choices in the reference that is mentioned in the comment and try to implement other windows that ensure non-negative fourier transform
        
        '''
        
        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        ref_channels = self.ref_channels
        corr_matrix = self.compute_correlation_matrices(tau_max)  
        
        psd_mats_shape = (num_analised_channels, num_ref_channels, tau_max)  
        psd_mats = np.zeros(psd_mats_shape, dtype=complex)

        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                this_correlation_function = corr_matrix[channel_1, channel_2,:]
                # create window, use Bartlett for nonnegative Fourier transform
                # otherwise Coherence becomes invalid
                # another option is to convolve another window with itself
                # from : SPECTRAL ANALYSIS OF SIGNALS, Petre Stoica and Randolph Moses, pp. 42 ff
                # window options: bartlett, blackman, hamming, hanning
                if window == 'bartlett':
                    win = np.bartlett(len(this_correlation_function))
                elif window=='blackman':
                    win = np.blackman(len(this_correlation_function))
                    win = np.convolve(win, win, 'same')
                    win /= np.max(win)
                elif window=='hamming':
                    win = np.hamming(len(this_correlation_function))
                    win = np.convolve(win, win, 'same')
                    win /= np.max(win)
                elif window=='hanning':
                    win = np.hanning(len(this_correlation_function))
                    win = np.convolve(win, win, 'same')
                    win /= np.max(win)
                elif window == 'rect':
                    win = np.ones(len(this_correlation_function))
                else:
                    raise RuntimeError('Invalid window.')
                    return
                #test with coherence, should be between 0 and 1
                #coherence = np.abs(G12)**2/G11/G22
                
                # applies window and calculates fft
                fft = np.fft.rfft(this_correlation_function*win, n=2*tau_max-1)
                #print(this_correlation_function.shape, fft.shape)
                # corrections
                fft = fft[:tau_max]
                ampl_correction= (tau_max)/(win).sum()
                fft *= ampl_correction
                
                if channel_1 == channel_2:
                    fft = np.abs(fft)

                psd_mats[channel_1, channel_2, :] = fft

        
        freqs = np.fft.rfftfreq(2*tau_max - 1, 1/self.sampling_rate) 

        self.psd_mats = psd_mats
        self.freqs = freqs
        self.n_lines = tau_max
        
        return psd_mats, freqs

    def welch(self, n_lines):
        print("Estimating Correlation Function and Power Spectral Density by Welch's method...")
        
        
        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        
        ref_channels = self.ref_channels
        dofs = list(range(self.measurement.shape[1]))
        
        signal = self.measurement
        
        #psd_mats_shape = (num_analised_channels, num_analised_channels, 2*n_lines)  
        psd_mats_shape = (num_analised_channels, num_ref_channels, 2*n_lines//2+1 )     
            
        psd_mats = np.zeros(psd_mats_shape, dtype=complex)
        
        for channel_1 in range(num_analised_channels):
            for channel_2, ref_channel in enumerate(ref_channels):
                #f, Pxy_den = scipy.signal.csd(signal[:,channel_1],signal[:,channel_2], self.sampling_rate, nperseg=n_lines*2, window='hamm', scaling='spectrum', return_onesided=False)
                f, Pxy_den = scipy.signal.csd(signal[:,channel_1],signal[:,ref_channel], self.sampling_rate, nperseg=n_lines*2, window='hamm', scaling='spectrum', return_onesided=True)
                
                if channel_1 == channel_2:
                    assert (Pxy_den.imag==0).all()
                    
                #Pxy_den *= 2*n_lines-1
                Pxy_den *= n_lines
                psd_mats[channel_1, channel_2, :] = Pxy_den
                
        corr_mats_shape = (num_analised_channels, num_ref_channels, n_lines)
            
        corr_matrix = np.zeros(corr_mats_shape)
        
        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                this_psd = psd_mats[channel_1, channel_2,:]
                this_corr = np.fft.ifft(this_psd)
                this_corr = this_corr[:n_lines].real
                
                #win = np.hamming(2*n_lines-1)[n_lines-1:]
                #this_corr /= win
                
                
                corr_matrix[channel_1, channel_2, :] = this_corr
                #corr_matrix[channel_1, channel_2, n_lines-1:] = this_corr
                #corr_matrix[channel_1, channel_2, :n_lines] = np.flip(this_corr,axis=-1)
                
        '''
        s_vals_cf = np.zeros((len(dofs), corr_matrix.shape[2]))
        for t in range(corr_matrix.shape[2]):
            s_vals_cf[:,t] = np.linalg.svd(corr_matrix[:,:,t],True,False)
    #     
        s_vals_psd = np.zeros((num_analised_channels, psd_mats.shape[2]))
        for t in range(psd_mats.shape[2]):
            s_vals_psd[:,t] = np.linalg.svd(psd_mats[:,:,t],True,False)     
          '''   
        self.corr_matrix = corr_matrix
        self.psd_mats = psd_mats
        self.n_lines = n_lines
        '''self.s_vals_cf = s_vals_cf
        self.s_vals_psd = s_vals_psd'''
        
        return corr_matrix, psd_mats#, s_vals_cf, s_vals_psd        
          
    def get_s_vals_psd(self, n_lines=256, window='hamm'):
        num_analised_channels = self.num_analised_channels
        psd_mats, freqs = self.psd_welch(n_lines=n_lines, refs_only=False, window=window)
        s_vals_psd = np.zeros((num_analised_channels, psd_mats.shape[2]))
        for t in range(psd_mats.shape[2]):
            # might use only real part to account for slightly asynchronous data 
            # see [Au (2017): OMA, Chapter 7.5]
            s_vals_psd[:,t] = np.linalg.svd(psd_mats[:,:,t],True,False)
        return s_vals_psd, freqs
    
    def compute_correlation_matrices(self, tau_max, num_blocks=False):
        '''
        This function computes correlation functions of all channels with
        selected reference channels up to a time lag of tau_max
        if num_blocks is greater than 1 the correlation functions are computed in
        blocks, which allows the estimation of variances for each time-lag at 
        the expense of a slightly reduced quality of the correlation function
        i.e. blocks may not overlap and therefore for the higher lags, less samples
        are used for the estimation of the correlation        
        '''
        print('Computing Correlation Matrices with tau_max {}...'.format(tau_max))
        total_time_steps = self.total_time_steps
        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        measurement = self.measurement
        ref_channels = self.ref_channels        
        #roving_channels = self.roving_channels
        
        self.tau_max = tau_max
        
        all_channels = list(range(num_analised_channels))#ref_channels + roving_channels
        #all_channels.sort()
        
        if not num_blocks:
            num_blocks = 1
            
        block_length = int(np.floor(total_time_steps/num_blocks))
        
        if block_length <= tau_max:
            raise RuntimeError('Block length (={}) must be greater or equal to max time lag (={})'.format(block_length, tau_max))

        corr_matrices_mem = []
        
        corr_mats_shape = (num_analised_channels, num_ref_channels, tau_max)
        
        for n_block in range(num_blocks):
            corr_memory = mp.Array(c.c_double, np.zeros((np.product(corr_mats_shape)))) # shared memory, can be used by multiple processes @UndefinedVariable
            corr_matrices_mem.append(corr_memory)
            
        measurement_shape=measurement.shape
        measurement_memory = mp.Array(c.c_double, measurement.reshape(measurement.size, 1))# @UndefinedVariable
                
        #each process should have at least 10 blocks to compute, to reduce overhead associated with spawning new processes 
        n_proc = min(int(tau_max*num_blocks/10), os.cpu_count())
        pool=mp.Pool(processes=n_proc, initializer=self.init_child_process, initargs=(measurement_memory, corr_matrices_mem)) # @UndefinedVariable
        
        iterators = []            
        it_len = int(np.ceil(tau_max*num_blocks/n_proc))
        printsteps = np.linspace(0,tau_max*num_blocks,100, dtype=int)
        
        curr_it = []
        i = 0
        for n_block in range(num_blocks):
            for tau in range(1,tau_max+1):
                i += 1
                if i in printsteps:                        
                    curr_it.append([n_block, tau, True])
                else:
                    curr_it.append((n_block, tau))
                if len(curr_it)>it_len:
                    iterators.append(curr_it)
                    curr_it = []
        else:
            iterators.append(curr_it)

        self.init_child_process(measurement_memory, corr_matrices_mem)
        for curr_it in iterators:
#             pool.apply_async(self.compute_covariance , args=(curr_it,
#                                                         tau_max,
#                                                         block_length, 
#                                                         ref_channels, 
#                                                         all_channels, 
#                                                         measurement_shape,
#                                                         corr_mats_shape))
            self.compute_covariance(curr_it, tau_max, block_length, ref_channels, all_channels, measurement_shape, corr_mats_shape)
                                  
        pool.close()
        pool.join()               

        corr_matrices = []
        for corr_mats_mem in corr_matrices_mem:
            corr_mats = np.frombuffer(corr_mats_mem.get_obj()).reshape(corr_mats_shape) 
            corr_matrices.append(corr_mats)
            
        self.corr_matrices = corr_matrices      
        
        corr_mats_mean = np.mean(corr_matrices, axis=0)
        #corr_mats_mean = np.sum(corr_matrices, axis=0)
        #corr_mats_mean /= num_blocks - 1
        self.corr_matrix = corr_mats_mean
        
        #self.corr_mats_std = np.std(corr_matrices, axis=0)
    
        print('.',end='\n', flush=True)   
        
        return corr_mats_mean
        
    def compute_covariance(self, curr_it, tau_max, block_length, ref_channels, all_channels, measurement_shape, corr_mats_shape):
        
        for this_it in curr_it:
            if len(this_it) > 2:
                print('.',end='', flush=True)
                del this_it[2]
            n_block, tau = this_it
            
            measurement = np.frombuffer(measurement_memory.get_obj()).reshape(measurement_shape)

            this_measurement = measurement[(n_block)*block_length:(n_block+1)*block_length,:]
                
            refs = this_measurement[:-tau,ref_channels]
            
            current_signals = this_measurement[tau:, all_channels]
            
            this_block = np.dot(current_signals.T, refs)/current_signals.shape[0]
#             for i, ref_channel in enumerate(ref_channels):
#                 print(this_block[ref_channel,i])
#                 for chan_dof in self.chan_dofs:
#                     if chan_dof[0]==ref_channel:
#                         print(chan_dof)
                        
            corr_memory = corr_matrices_mem[n_block]
            
            corr_mats = np.frombuffer(corr_memory.get_obj()).reshape(corr_mats_shape)
            
            with corr_memory.get_lock():
                corr_mats[:,:,tau-1] = this_block
        
    def init_child_process(self, measurement_memory_, corr_matrices_mem_):
        #make the  memory arrays available to the child processes
        global measurement_memory
        measurement_memory = measurement_memory_   
        
        global corr_matrices_mem
        corr_matrices_mem = corr_matrices_mem_
    
    def get_corr_0(self):
        
        ref_channels = self.ref_channels        
        all_channels = list(range(self.num_analised_channels))#
        
        measurement = self.measurement

        refs = measurement[:,ref_channels]
        
        current_signals = measurement[:, all_channels]
        
        this_block = np.dot(current_signals.T, refs)/current_signals.shape[0]
        
        return this_block
    
    
    def get_rms(self):
        self.correct_offset()
        return np.sqrt(np.mean(np.square(self.measurement),axis=0))
    
    def add_noise(self,amplitude=0,snr=0):
        print('Adding Noise with Amplitude {} and {} percent RMS'.format(amplitude,snr*100))
        assert amplitude !=0 or snr !=0
        
        if snr != 0 and amplitude == 0:
            rms = self.get_rms()
            amplitude = rms*snr
        else:
            amplitude = [amplitude for channel in range(self.num_analised_channels)]
            
        for channel in range(self.num_analised_channels):            
            self.measurement[:,channel] += np.random.normal(0,amplitude[channel],self.total_time_steps)
        
    def get_fft(self,svd=True, NFFT=2048):
        
        if self.ft_freq is None or self.sum_ft is None:
            ft, self.ft_freq  = self.psd_welch(n_lines=NFFT, refs_only=False)
            if not svd:
                self.sum_ft = np.abs(np.sum(ft, axis=0))
            else:
                self.sum_ft = np.zeros((self.num_analised_channels, len(self.ft_freq )))
                for i in range(len(self.ft_freq )):
                    #might use only real parts of psd o account for slightly asynchronous data 
                    # see [Au (2017): OMA, Chapter 7.5]
                    u,s,vt = np.linalg.svd(ft[:,:,i])
                    self.sum_ft[:,i]=10*np.log(s)
                    #print(10*np.log(s))
                    
        #print(self.ft_freq.shape, self.sum_ft.shape)
        return self.ft_freq, self.sum_ft
    
    def get_time_accel(self, channel):
        time_vec = np.linspace(0, self.total_time_steps/self.sampling_rate, self.total_time_steps)
        accel_vel = self.measurement[:,channel]
        return time_vec, accel_vel   
     
    def plot_svd_spectrum(self,NFFT=512, log_scale=False, ax=None):

        if ax is None:
            ax=plot.subplot(111)
        
        psd_matrix, freq = self.psd_welch(NFFT,False)
        svd_matrix = np.zeros((self.num_analised_channels, len(freq)))
        #print(freq)
        for i in range(len(freq)):
            # might use only real part to account for slightly asynchronous data 
            # see [Au (2017): OMA, Chapter 7.5]
            u,s,vt = np.linalg.svd(psd_matrix[:,:,i])
            if log_scale: s = 10*np.log10(s)
            svd_matrix[:,i]=s#10*np.log(s)
            

        for i in range(self.num_analised_channels):
            ax.plot(freq, svd_matrix[i,:])
            #if i>3: break
             

        ax.set_xlim((0,self.sampling_rate/2))
        #plot.grid(1)
        #ax.set_xlabel('Frequency [\si{\hertz}]')
        if log_scale: ax.set_ylabel('Singular Value Magnitude [\si{\decibel}]')
        else: ax.set_ylabel('Singul\\"arwert Magnitude')
        #plot.yticks([0,-25,-50,-75,-100,-125,-150,-175,-200,-225,-250])
        #plot.ylim((-225,0))
        #plot.xlim((0.1,5))

        #plot.grid(b=0)
        
        #plot.show()
    def plot_correlation(self, tau_max=None, num_blocks=False, ax=None):
        
        assert tau_max or self.corr_matrix.shape
        
        if tau_max is not None:
            if not self.corr_matrix:
                self.compute_correlation_matrices(tau_max, num_blocks)
            elif self.corr_matrix.shape[2]<=tau_max:
                self.compute_correlation_matrices(tau_max, num_blocks)
            corr_matrix = self.corr_matrix[:,:,:tau_max]
        else:
            corr_matrix = self.corr_matrix
            tau_max = corr_matrix.shape[2]
            
        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        
        if ax is None:
            ax=plot.subplot(111)
        
        
        
        for ref_channel in range(num_ref_channels):
            for channel in range(num_analised_channels):
                ax.plot(corr_matrix[channel,ref_channel,:])
        ax.set_xlim((0,tau_max))
        ax.set_xlabel('$\tau_{\text{max}}$')
        ax.set_ylabel('$R_{i,j}(\tau) [\si{\milli\metre\squared\per\second\tothe{4}}]')
        
    def plot_psd(self, tau_max = None, n_lines=None, method='blackman', ax=None, **kwargs):
        
        assert tau_max or self.psd_mats is not None or n_lines
        assert method in ['blackman', 'welch']
        
        if tau_max is None and n_lines is not None:
            tau_max = n_lines
            
        if tau_max is not None:
            if not self.psd_mats.shape:
                if method == 'blackman':
                    self.psd_blackman_tukey(tau_max, **kwargs)
                else:
                    self.psd_welch(tau_max, **kwargs)
            elif self.psd_mats.shape[2]<= tau_max:
                if method == 'blackman':
                    self.psd_blackman_tukey(tau_max, **kwargs)
                else:
                    self.psd_welch(tau_max, **kwargs)
                        
            psd_mats = self.psd_mats[:,:,tau_max]
        else:
            psd_mats = self.psd_mats
        
        freqs = self.freqs

        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        
        if ax is None:
            ax=plot.subplot(111)
            
        for ref_channel in range(num_ref_channels):
            for channel in range(num_analised_channels):
                ax.plot(freqs,np.abs(psd_mats[channel,ref_channel,:]))
        ax.set_xlim((0,freqs.max()))
        if plot.rc('latex.usetex'):
            ax.set_xlabel('$f [\si{\hertz}]$')
            ax.set_ylabel('$S_{i,j}(f) [\si{\milli\metre\squared\per\second\tothe{4}}\per\hertz]$')
        else:
            ax.set_xlabel('f [Hz]')
            ax.set_ylabel('S_{i,j}(f) [mm^2/s^4/Hz]')
            
def load_measurement_file(fname, **kwargs):
    # assign this function to the class before instantiating the object
    # PreprocessData.load_measurement_file = load_measurement_file
    
    # define a function to return the following variables
    headers=['channel_name','channel_name']
    units=['unit','unit',]
    start_time=datetime.datetime()
    sample_rate = float()
    measurement=np.array()
    
    #channels im columns
    assert measurement.shape[0]>measurement.shape[1]
        
    return headers, units, start_time, sample_rate, measurement  

def main():
    pass
#     def handler(msg_type, msg_string):
#         pass
# 
#     if not 'app' in globals().keys():
#         global app
#         app = QApplication(sys.argv)
#     if not isinstance(app, QApplication):
#         app = QApplication(sys.argv)
# 
#     # qInstallMessageHandler(handler) #suppress unimportant error msg
#     prep_data = PreprocessData.load_state('/vegas/scratch/womo1998/towerdata/towerdata_results_var/Wind_kontinuierlich__9_2016-10-05_04-00-00_000000/prep_data.npz')
#     #prep_data = None
#     preprocess_gui = PreProcessGUI(prep_data)
#     loop = QEventLoop()
#     preprocess_gui.destroyed.connect(loop.quit)
#     loop.exec_()
#     print('Exiting GUI')
# 
#     return

def example_filter():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_data = PreprocessData(measurement, sampling_rate=128)
    
    prep_data.filter_data(order = 4, ftype='cheby1', lowpass=20, highpass=None, RpRs = [0.1, 0.1], overwrite=False, plot_filter=True)

def example_decimate():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_data = PreprocessData(measurement, sampling_rate=128)
    
    _, f_plot = plt.subplots(2)
    f_plot[0].set_title('Decimate data')           
    f_plot[0].plot(np.linspace(0, 1, len(prep_data.measurement[:,1])), prep_data.measurement[:,1])
    print('Original sampling rate: ', prep_data.sampling_rate, 'Hz')
    print('Original number of time steps: ', prep_data.total_time_steps)
    
    prep_data.decimate_data(5, order=8, filter_type='cheby1')
    
    print(prep_data.measurement[:,1])
    f_plot[1].plot(np.linspace(0, 1, len(prep_data.measurement[:,1])), prep_data.measurement[:,1])
    print('Decimated sampling rate: ', prep_data.sampling_rate, 'Hz')
    print('Decimated number of time steps: ', prep_data.total_time_steps)
    plt.show()
    
def example_welch():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_data = PreprocessData(measurement, sampling_rate=128, ref_channels=[0, 1])
    
    startA = time.time()
    corr_matrix, psd_mats = prep_data.welch(256)
    print('Function A - Time elapsed: ', time.time() - startA)

    startB = time.time()
    corr_matrix_new = prep_data.corr_welch(256)
    psd_mats_new, freqs = prep_data.psd_mats, prep_data.freqs # were certainly generated during function call by corr_welch
    print('Function B - Time elapsed: ', time.time() - startB)
 
    chA = 7
    chB = 1 

    _, f_plot = plt.subplots(2)
    f_plot[0].set_title('Correlation Values - different functions')  
    f_plot[0].plot(corr_matrix[chA,chB,:])
    f_plot[1].plot(corr_matrix_new[chA,chB,:])
    plt.figure()
    plt.title('Correlation Values - Superimposed graph')  
    plt.plot(corr_matrix[chA,chB,:])
    plt.plot(corr_matrix_new[chA,chB,:])
    #plt.show()
    

    _, f_plot = plt.subplots(2)
    f_plot[0].set_title('PSD Values - different functions')  
    f_plot[0].plot(np.abs(psd_mats[chA,chB,:]))
    f_plot[1].plot(np.abs(psd_mats_new[chA,chB,:]))
    plt.figure()
    plt.title('PSD Values - Superimposed graph')  
    plt.plot(np.abs(psd_mats[chA,chB,:]))
    plt.plot(np.abs(psd_mats_new[chA,chB,:]))
            
    plt.show()

def example_blackman_tukey():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_data = PreprocessData(measurement, sampling_rate=128, ref_channels=[0, 1])
    
    startA = time.time()
    psd_mats, freqs = prep_data.psd_blackman_tukey(tau_max=256, window = 'bartlett')
    corr_matrix = prep_data.corr_matrix
    print('Time elapsed: ', time.time() - startA)
    #print(freqs)
    chA = 7
    chB = 1
    
    plt.figure()
    plt.title('Correlation Values')  
    plt.plot(corr_matrix[chA,chB,:])
    plt.figure()
    plt.title('PSD Values')  
    plt.plot(freqs,np.abs(psd_mats[chA,chB,:]))
    plt.show()

def compare_PSD_Corr():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_data = PreprocessData(measurement, sampling_rate=128, ref_channels=[0, 1])
    
    prep_data.filter_data(lowpass=10, highpass=0.1, overwrite=True)
    
    startA = time.time()
    psd_mats_b, freqs_b = prep_data.psd_blackman_tukey(tau_max=2048, window = 'hamming')
    corr_matrix_b = prep_data.corr_matrix 
    print('Blackman-Tukey - Time elapsed: ', time.time() - startA)
    
    startB = time.time()
    corr_matrix_w = prep_data.corr_welch(2048, window='hamming')
    psd_mats_w, freqs_w = prep_data.psd_mats, prep_data.freqs
    print('Welch - Time elapsed: ', time.time() - startB)
    
    
    chA = 7
    chB = 1
    
    plt.figure()
    plt.title('Correlation Values')  
    plt.plot(corr_matrix_b[chA,chB,:])
    plt.plot(corr_matrix_w[chA,chB,:])
    plt.figure()
    plt.title('PSD Values')  
    plt.plot(freqs_b,np.abs(psd_mats_b[chA,chB,:]))
    plt.plot(freqs_w,np.abs(psd_mats_w[chA,chB,:]))
    plt.show()    
    
if __name__ =='__main__':
    import os
    path = 'E:/OneDrive/BHU_NHRE/Python/2017_PreProcessGUI/'
    path = '/ismhome/staff/womo1998/Projects/2017_PreProcessGUI/'
    os.chdir(path)
    
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_data = PreprocessData(measurement, sampling_rate=128, ref_channels=[0, 1])
    prep_data.plot_svd_spectrum(8192)
    plot.show()
    #example_filter()
    #example_decimate()
    #example_welch()
    #example_blackman_tukey()
    compare_PSD_Corr()
    main()

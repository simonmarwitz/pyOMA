# 
# -*- coding: utf-8 -*-
'''
Based on previous works by Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
TODO:
- correct linear,.... offsets as well
- implement filter functions
- implement loading of different filetypes ascii, lvm, ...
- currently loading geometry, etc. files will overwrite existing assignments
    implement "load and append"
- implement fft, psd, covariance (auto,cross), coherence, frf (to be used with a preprocessor gui)
- implement integration
- implement windowing functions

'''
import numpy as np
from scipy import signal
import scipy.signal.ltisys
import os
import csv
import sys
import datetime

import multiprocessing as mp
import ctypes as c
import warnings

def nearly_equal(a,b,sig_fig=5):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )
class GeometryProcessor(object):
    '''
        conventions:

        - chan_dofs=[(chan, node, (x_amplif,y_amplif,z_amplif)),...]

        - channels = 0 ... #, starting at channel 0, should be a complete sequence

        - nodes = 1 ... #, starting at node 1, can be a sequence with missing entries

        - lines = [(node_start, node_end),...], unordered

        - master_slaves = [(node_master, x_master, y_master, z_master, 
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
    - load ascii datafiles
    - specify sampling rate, reference channels and roving channels
    - specify geometry, channel-dof-assignments
    - specify channel quantities such as acceleration, velocity, etc 
    - remove channels, cut time histories
    - remove (constant) offsets from time history data
    - decimate time histories
    future:
    - apply several filters
    - calculate fft, psd, covariance, coherence, frf
    - integrate
    - apply windowing functions
    '''
    def __init__(self, measurement, sampling_rate, total_time_steps=None, 
                 num_channels=None,ref_channels=None, roving_channels=None,
                 accel_channels=None, velo_channels=None, disp_channels=None,
                 setup_name=None, channel_headers=None, start_time=None, ft_freq=None, sum_ft=None):
        
        super().__init__()
        
        assert isinstance(measurement, np.ndarray)
        assert measurement.shape[0] > measurement.shape[1]
        self.measurement = measurement
        self.measurement_filt = measurement
        
        assert isinstance(sampling_rate, (int,float))
        self.sampling_rate = sampling_rate
        
        if total_time_steps is None:
            total_time_steps = measurement.shape[0]
        
        assert  measurement.shape[0] >= total_time_steps

        self.total_time_steps = total_time_steps
        
        if ref_channels is None:
            ref_channels = list(range(measurement.shape[1]))
        self.ref_channels = ref_channels
        if roving_channels is None:
            roving_channels = [i for i in range(measurement.shape[1]) if i not in ref_channels]
        self.roving_channels = roving_channels
        
        self.num_ref_channels = len(self.ref_channels)
        self.num_roving_channels = len(self.roving_channels)
        self.num_analised_channels = self.num_ref_channels + self.num_roving_channels
        
        if num_channels is None:
            num_channels = self.num_analised_channels
            
        assert num_channels <= self.measurement.shape[1]    
        
        if ((self.num_ref_channels + self.num_roving_channels) > num_channels):
                sys.exit('The sum of reference and roving channels is greater than the number of all channels!')
        
        for ref_channel in self.ref_channels:
            if (ref_channel < 0):
                sys.exit('A reference channel number cannot be negative!')
            if (ref_channel > (num_channels - 1)):
                sys.exit('A reference channel number cannot be greater than the number of all channels!')
            for rov_channel in self.roving_channels:
                if (rov_channel < 0):
                    sys.exit('A roving channel number cannot be negative!')
                if (rov_channel > (num_channels - 1)):
                    sys.exit('A roving channel number cannot be greater than the number of all channels!')
                if (ref_channel == rov_channel):
                    sys.exit('Any channel can be either a reference OR a roving channel. Check your definitions!')
        
        if disp_channels is None:
            disp_channels = []
        if velo_channels is None:
            velo_channels = []
        if accel_channels is None:
            accel_channels = [c for c in self.ref_channels+self.roving_channels if c not in disp_channels or c not in velo_channels]
        
        for channel in self.ref_channels+self.roving_channels:
            if (channel in accel_channels) + (channel in velo_channels) + (channel in disp_channels) != 1:
                import warnings
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
        
    @classmethod
    def init_from_config(cls, conf_file, meas_file, chan_dofs_file=None, **kwargs):
        '''
        initializes the PreProcessor object with a configuration file
        
        to remove channels at loading time use 'usecols' keyword argument
        if delete_channels are specified, these will be checked against 
        all other channel definitions, which will be adjusted accordingly
        '''
        assert os.path.exists(conf_file)
        
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
        #print(delete_channels)
        if delete_channels:
            #delete_channels.sort(reverse=True)
            
            names=['Reference Channels', 'Accel. Channels', 'Velo. Channels', 'Disp. Channels']
            channel_lists=[ref_channels, accel_channels, velo_channels, disp_channels]
            #print(chan_dofs)
            channel = measurement.shape[1]
            #num_channels = measurement.shape[1]
            while channel >= 0:
                
                if channel in delete_channels:
                    # affected lists: ref_channels, accel_channels, velo_channels, disp_channels + chan_dofs
                    # remove channel from all lists
                    # decrement all channels higher than channel in all lists
                    #num_channels -= 1
                    for channel_list in channel_lists:
                        if channel in channel_list:
                            channel_list.remove(channel)
                            print('Channel {} removed from {} list'.format(channel, names[channel_lists.index(channel_list)]))
                        for channel_ind in range(len(channel_list)):
                            if channel_list[channel_ind] > channel:
                                channel_list[channel_ind] -= 1 
                                
                    if chan_dofs:
                        this_num_channels = len(chan_dofs)
                        chan_dof_ind = 0
                        while chan_dof_ind < this_num_channels:
                            if channel==chan_dofs[chan_dof_ind][0]:
                                print('Channel-DOF-Assignment {} removed.'.format(chan_dofs[chan_dof_ind]))
                                del chan_dofs[chan_dof_ind]
                                this_num_channels -= 1
                            elif channel < chan_dofs[chan_dof_ind][0]:
                                chan_dofs[chan_dof_ind][0] -= 1
                            chan_dof_ind += 1
                    print('Now removing Channel {} (no. {})!'.format(headers[channel], channel))  
                    del headers[channel]
                channel -= 1
            #print(chan_dofs)   
            
            measurement=np.delete(measurement, delete_channels, axis=1)
        total_time_steps = measurement.shape[0]
        num_channels = measurement.shape[1]
        roving_channels = [i for i in range(num_channels) if i not in ref_channels]
        if not accel_channels and not velo_channels and not disp_channels:
            accel_channels = [i for i in range(num_channels)]
        #print(measurement.shape, ref_channels)
        #print(measurement)
        prep_data = cls(measurement, sampling_rate, total_time_steps, 
                 num_channels, ref_channels, roving_channels,
                 accel_channels, velo_channels, disp_channels, channel_headers=headers, start_time=start_time )
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
                chan_num, node, az, elev, chan_name = [float(line[i]) if not i in [1,4] else line[i].strip(' ') for i in range(5)]
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
        out_dict['self.roving_channels'] = self.roving_channels
        out_dict['self.num_ref_channels'] = self.num_ref_channels
        out_dict['self.num_roving_channels'] = self.num_roving_channels
        out_dict['self.num_analised_channels'] = self.num_analised_channels
        out_dict['self.accel_channels']=self.accel_channels
        out_dict['self.velo_channels']=self.velo_channels
        out_dict['self.disp_channels']=self.disp_channels
        out_dict['self.chan_dofs']=self.chan_dofs
        out_dict['self.channel_headers'] = self.channel_headers
        out_dict['self.start_time']=self.start_time
        out_dict['self.ft_freq']=self.ft_freq
        out_dict['self.sum_ft']=self.sum_ft
        
        #out_dict['self.geometry_data'] = self.geometry_data

        np.savez_compressed(fname, **out_dict)
         
    @classmethod
    def load_state(cls, fname):
        
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname)    

        setup_name= str(in_dict['self.setup_name'].item())
        measurement = in_dict['self.measurement']
        sampling_rate = float(in_dict['self.sampling_rate'])
        total_time_steps = int(in_dict['self.total_time_steps'])
        ref_channels = list(in_dict['self.ref_channels'])
        roving_channels = list(in_dict['self.roving_channels'])
        if in_dict['self.channel_headers'].shape:
            channel_headers = list(in_dict['self.channel_headers'])
        else:
            channel_headers =['' for chan in ref_channels+roving_channels]
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
        #sum_ft = in_dict.get( 'self.sum_ft', None)
        
        preprocessor = cls(measurement, sampling_rate, total_time_steps, 
                 ref_channels=ref_channels, roving_channels=roving_channels,
                 accel_channels=accel_channels, velo_channels=velo_channels, 
                 disp_channels=disp_channels, setup_name=setup_name,
                 channel_headers=channel_headers, start_time=start_time, 
                 ft_freq=ft_freq, sum_ft = sum_ft)
        
        chan_dofs = [[int(float(chan_dof[0])), str(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3]), str(chan_dof[-1])] for chan_dof in in_dict['self.chan_dofs']]
        preprocessor.add_chan_dofs(chan_dofs)
        assert preprocessor.num_ref_channels == int(in_dict['self.num_ref_channels'])
        assert preprocessor.num_roving_channels == int(in_dict['self.num_roving_channels'])
        assert preprocessor.num_analised_channels == int(in_dict['self.num_analised_channels'])
        
        #preprocessor.add_geometry_data(in_dict['self.geometry_data'].item())  
        return preprocessor
    
    def filter_data(self, lowpass=None, highpass=None, filt_order=4, num_int=0,overwrite=False,plot_filter=False):
        '''
        Applies various filters to the data
        '''

        if lowpass is not None: lowpass=float(lowpass)
        if highpass is not None: highpass=float(highpass)
        filt_order=int(filt_order)
        print('filtering data, with {},{}'.format(lowpass,highpass))
        import scipy.signal
        import scipy.integrate
        
        nyq=self.sampling_rate*0.5
        
        if lowpass is not None and highpass is not None:            
            b, a = scipy.signal.butter(filt_order, [highpass/nyq, lowpass/nyq], btype='band')
        elif lowpass is not None:
            b, a = scipy.signal.butter(filt_order, lowpass/nyq, btype = 'low')
        elif highpass is not None:
            b, a = scipy.signal.butter(filt_order, highpass/nyq, btype = 'high')
            
        if plot_filter:
            import matplotlib.pyplot as plt
            
            w, h = scipy.signal.freqz(b, a, worN=2000)
            plt.close()
            plt.figure()
            plt.plot((nyq / np.pi) * w, abs(h))
            plt.plot([0, nyq], [np.sqrt(0.5), np.sqrt(0.5)],
                 '--', label='sqrt(0.5)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Gain')
            plt.grid(True)
            plt.legend(loc='best')
            plt.grid(which='both', axis='both')
            plt.axvline(100, color='green') # cutoff frequency
            plt.show()
            plt.figure()
            plt.plot(self.measurement[:,1], label='Original signal (Hz)')
        self.measurement_filt = scipy.signal.filtfilt(b,a,self.measurement,axis=0, padlen=0)
        if overwrite:
            self.measurement = self.measurement_filt
        #for ii in range(self.measurement_filt.shape[1]):
            #pass
        #    self.measurement_filt[:,ii]  -= self.measurement_filt[:,ii].mean(0)
        if plot_filter:
            plt.plot(self.measurement_filt[:,1], label='Filtered signal (Hz)')
            plt.legend()
            plt.show()
        return
    
        for i in range(num_int):
            #size = 2**np.floor(np.log2(self.measurement_filt.shape[0]))+1
            for ii in range(self.measurement_filt.shape[1]):
                #self.measurement_filt[:,ii] = scipy.integrate.romb(self.measurement_filt[:size,ii], 1/self.sampling_rate)
                temp = scipy.integrate.cumtrapz(self.measurement_filt[:,ii], dx=1/self.sampling_rate,axis=0)
                self.measurement_filt[:temp.shape[0],ii] = temp
                self.measurement_filt[:,ii]  -= self.measurement_filt[:,ii].mean(0)            
        
            #plt.plot(self.measurement_filt[:,1], label='Integrated {} signal (Hz)'.format(i))
        #plt.legend()
        #plt.show()
        
    def plot_data(self, channels=[], figsize=None):
        if len(channels)==0:
            channels=list(range(self.num_analised_channels))
        import matplotlib.pyplot as plot
        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, len(channels)))
        
#         plot.rc('font', size=10)    
#         plot.rc('legend', fontsize=10, labelspacing=0.1)
#         plot.rc('axes',linewidth=0.2)
#         plot.rc('xtick.major',width=0.2)
#         plot.rc('ytick.major',width=0.2)
# 
#         plot.rc('text', usetex=True)
#         plot.rc('text.latex',preamble=r"\usepackage{siunitx}")
#         
#         figsize[0]/=3
#         figsize[1]/=3
#         plot.figure(figsize=figsize, tight_layout=True)
#         plot.plot(np.linspace(0,512/self.sampling_rate,512),self.measurement[512:1024,4], color=colors[5])
#         ylims = plot.ylim()
#         plot.xlim((0,512/self.sampling_rate))
#         plot.yticks([])
#         plot.ylabel('accel. [\si{\meter\per\second\squared}]')
#         plot.xlabel('time [\si{second}]')
#         ovl_meas = self.measurement[512:1024,4]
#         ovl_meas[ovl_meas>0.005]=0.005
#         ovl_meas[ovl_meas<-0.005]=-0.005
#         plot.figure(figsize=figsize, tight_layout=True)
#         plot.plot(np.linspace(0,512/self.sampling_rate,512),ovl_meas, color=colors[2])
#         plot.ylim(ylims)
#         plot.xlim((0,512/self.sampling_rate))
#         plot.yticks([])
#         plot.ylabel('accel. [\si{\meter\per\second\squared}]')
#         plot.xlabel('time [\si{second}]')
#         plot.show()
#         
#         
#         #plot.rc('xtick.major',pad=-10)
#         #plot.rc('ytick.major',pad=-10)
#         fig=plot.figure(figsize=figsize, tight_layout=True)
#         from mpl_toolkits.mplot3d import Axes3D
#         ax=fig.add_subplot(111,projection='3d')
#         for channel in range(3):
#             ax.plot3D(xs=np.array(range(self.measurement.shape[0]))/self.sampling_rate,zs=self.measurement[:,channel]/self.measurement[:,channel].max(), ys=np.repeat(-1*channel, self.measurement.shape[0]), label=self.channel_headers[channel],color=colors[channel*2], alpha=0.65)
#         #plot.plot(self.measurement[:,0],alpha=0.75)
#         ax.set_xlim((0,self.measurement.shape[0]/self.sampling_rate),)
#         ax.set_xticks([])
#         ax.set_zticks([])
#         ax.set_yticks([])
#         ymin,ymax=ax.get_zlim()
#         numblocks=7
#         xstep = self.measurement.shape[0]/numblocks/self.sampling_rate
#         #for channel in range(3):
#          #   for block in range(1,numblocks):
#                 #print(block*xstep,ymin,ymax,channel)
#                 #ax.plot3D(xs=(block*xstep,block*xstep), zs=(ymin,ymax), ys = (-channel,-channel), color='red')
#                 #ax.plot_surface(X=(block*xstep,block*xstep), Z=(ymin,ymax), Y = (-channel,-channel), color='red')
# 
#         for block in range(0,numblocks+1):
#             print(block*xstep,ymin,ymax,channel)
#             #ax.plot3D(xs=(block*xstep,block*xstep), zs=(ymin,ymax), ys = (-channel,-channel), color='red')
#             ax.plot_surface(X=([block*xstep,block*xstep],[block*xstep,block*xstep]), Z=([ymin,ymax],[ymin,ymax]), Y = ([0,0],[-channel,-channel]), color='red',alpha=0.25)
# 
#         ax.set_zlim((ymin,ymax))
#         ax.set_ylim((-2,0))
#         ax.set_xlabel('time')
#         ax.set_ylabel('channel')
#         ax.set_zlabel('acceleration')
#         
#         #ax.yaxis._axinfo['label']['space_factor']=2.8
#         #ax.yaxis.set_label_coords(-0.5, -4)
#         #ax.zaxis.set_label_coords(-0.5, -4)
#         plot.show()
        
        
        fig=plot.figure()
        from mpl_toolkits.mplot3d import Axes3D
        ax=fig.add_subplot(111,projection='3d')
        for channel in channels:
            ax.plot3D(xs=np.array(range(self.measurement.shape[0]))/self.sampling_rate,zs=self.measurement[:,channel]/self.measurement[:,channel].max(), ys=np.repeat(-1*channel, self.measurement.shape[0]), label=self.channel_headers[channel], color=colors[channel], alpha=0.5)
        ax.set_xlim((0,self.measurement.shape[0]/self.sampling_rate))
        ax.set_xlabel('time $t$ [s]')
        ax.set_zlabel('acceleration [$m/s^2$]')
        ax.set_ylabel('Channel Nr.')
        plot.show()
    
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
        
        #import matplotlib.pyplot as plot
        #print(self.channel_factors)
        #for i in range(self.measurement.shape[1]):
        #    plot.hist(self.measurement[:,i],bins=50,alpha=0.1)
        #plot.show()
        
    
    def decimate_data(self, decimate_factor, highpass=None):  
        '''
        decimates the measurement data by the supplied factor
        makes use of scipy's decimate filter
        '''  

        
        if highpass is not None:
            nyq = self.sampling_rate/2
            target_fs = self.sampling_rate/decimate_factor
            lowpass = target_fs/2
            b, a = signal.butter(4, [highpass/nyq, lowpass/nyq], btype='band')   
            ftype=scipy.signal.ltisys.dlti(b,a)
        else:
            ftype='iir'
            
        meas_decimated = scipy.signal.decimate(self.measurement, decimate_factor, axis=0,ftype=ftype, zero_phase=True)
        
        self.sampling_rate /=decimate_factor
        self.total_time_steps = meas_decimated.shape[0]
        self.measurement = meas_decimated
        
    def compute_correlation_matrices(self, tau_max, num_blocks=False):
        print('Computing Correlation Matrices...')
        total_time_steps = self.total_time_steps
        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        measurement = self.measurement
        ref_channels = self.ref_channels        
        roving_channels = self.roving_channels
        
        self.tau_max = tau_max
        
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        
                
        if not num_blocks:
            num_blocks = 1
            
        block_length = int(np.floor(total_time_steps/num_blocks))
        #tau_max = num_block_columns+num_block_rows
        if block_length <= tau_max:
            raise RuntimeError('Block length (={}) must be greater or equal to max time lag (={})'.format(block_length, tau_max))
        #extract_length = block_length - tau_max

        corr_matrices_mem = []
        
        corr_mats_shape = (num_analised_channels, tau_max * num_ref_channels)
        for n_block in range(num_blocks):
            corr_memory = mp.Array(c.c_double, np.zeros((np.product(corr_mats_shape)))) # shared memory, can be used by multiple processes @UndefinedVariable
            corr_matrices_mem.append(corr_memory)
            
        #measurement*=float(np.sqrt(block_length))
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

        
        for curr_it in iterators:
            pool.apply_async(self.compute_covariance , args=(curr_it,
                                                        tau_max,
                                                        block_length, 
                                                        ref_channels, 
                                                        all_channels, 
                                                        measurement_shape,
                                                        corr_mats_shape))
                                  
        pool.close()
        pool.join()               


        corr_matrices = []
        for corr_mats_mem in corr_matrices_mem:
            corr_mats = np.frombuffer(corr_mats_mem.get_obj()).reshape(corr_mats_shape) 
            corr_matrices.append(corr_mats*num_blocks)
            
        self.corr_matrices = corr_matrices      
        
        corr_mats_mean = np.mean(corr_matrices, axis=0)
        #corr_mats_mean = np.sum(corr_matrices, axis=0)
        #corr_mats_mean /= num_blocks - 1
        self.corr_mats_mean = corr_mats_mean
        #self.corr_mats_std = np.std(corr_matrices, axis=0)
        
        print('.',end='\n', flush=True)   
        
    def compute_covariance(self, curr_it, tau_max, block_length, ref_channels, all_channels, measurement_shape, corr_mats_shape, detrend=False):
        
        overlap = True
        
        #sys.stdout.flush()
        #normalize=False
        for this_it in curr_it:
            if len(this_it) > 2:
                print('.',end='', flush=True)
                del this_it[2]
            n_block, tau = this_it
            num_analised_channels = len(all_channels)
            num_ref_channels =len(ref_channels)
            
            measurement = np.frombuffer(measurement_memory.get_obj()).reshape(measurement_shape)
            if overlap:
                this_measurement = measurement[(n_block)*block_length:(n_block+1)*block_length+tau,:]#/np.sqrt(block_length)
            else:
                this_measurement = measurement[(n_block)*block_length:(n_block+1)*block_length,:]
                
            if detrend:this_measurement = this_measurement - np.mean(this_measurement,axis=0)
            
            refs = (this_measurement[:-tau,ref_channels]).T
            
            current_signals = (this_measurement[tau:, all_channels]).T
            
            this_block = (np.dot(current_signals, refs.T))/current_signals.shape[0]

            corr_memory = corr_matrices_mem[n_block]
            
            corr_mats = np.frombuffer(corr_memory.get_obj()).reshape(corr_mats_shape)
            
            with corr_memory.get_lock():
                corr_mats[:,(tau-1)*num_ref_channels:tau*num_ref_channels] = this_block        
        
    def init_child_process(self, measurement_memory_, corr_matrices_mem_):
        #make the  memory arrays available to the child processes
        
        global measurement_memory
        measurement_memory = measurement_memory_   
        
        global corr_matrices_mem
        corr_matrices_mem = corr_matrices_mem_
        
    def plot_covariances(self):
        tau_max = self.tau_max
        num_ref_channels = self.num_ref_channels     
        num_analised_channels = self.num_analised_channels   
        corr_matrices = self.corr_matrices
        ref_channels = self.ref_channels
#         subspace_matrices = []
#         for n_block in range(self.num_blocks):
#             corr_matrix = self.corr_matrices[n_block]
#             this_subspace_matrix= np.zeros(((num_block_rows+1)*num_analised_channels, num_block_columns*num_ref_channels))
#             for block_column in range(num_block_columns):
#                 this_block_column = corr_matrix[block_column*num_analised_channels:(num_block_rows+1+block_column)*num_analised_channels,:]
#                 this_subspace_matrix[:,block_column*num_ref_channels:(block_column+1)*num_ref_channels]=this_block_column
#             subspace_matrices.append(this_subspace_matrix)
        #self.subspace_matrices = subspace_matrices
        #subspace_matrices = self.subspace_matrices
        
        import matplotlib.pyplot as plot
        #matrices = subspace_matrices+[self.subspace_matrix]
        #matrices = [self.subspace_matrix]
        for corr_matrix in corr_matrices:
            for num_channel,ref_channel in enumerate(ref_channels):
                indices = (np.arange(tau_max)*num_analised_channels + ref_channel,np.repeat([num_channel],tau_max))
                print(indices, corr_matrix.shape)
                plot.plot(corr_matrix[indices])
             
        plot.show()                
    def correct_time_lag(self, channel, lag, sampling_rate):
        '''
        Method does not work very well for small time lags, although
        it might be necessary in certain cases
        '''
        #lag in ms
        #sampling rate in 1/s
        
        def gcd(a, b):
            #Return greatest common divisor using Euclid's Algorithm.
            while b:      
                a, b = b, a % b
            return a
        
        def lcm(a, b):
            #Return lowest common multiple.
            return a * b // gcd(a, b)
        
        delta_t=1/sampling_rate*1000 #ms
        sig_num=2
        factor = lcm(int(delta_t*10**sig_num), int(lag*10**sig_num))/(10**sig_num)
        resampled_col=signal.resample(self.measurement[:,channel], factor*self.measurement.shape[0])
        num_shift = int(sampling_rate*factor*lag/1000)
        shifted_col= resampled_col[num_shift:]
        decimated_col=signal.decimate(shifted_col, factor)
        self.measurement=self.measurement[:decimated_col.shape[1],:]
        self.measurement[:,channel]=decimated_col
        
    def get_rms(self):
        self.correct_offset()
        return np.sqrt(np.mean(np.square(self.measurement),axis=0))
    
    def add_noise(self,amplitude=0,snr=0):
        print('Adding Noise with Amplitude {} and {} percent RMS'.format(amplitude,snr*100))
        assert amplitude !=0 or snr !=0
        
        if snr != 0 and amplitude == 0:
            rms = self.get_rms()
            #print(rms)
            amplitude = rms*snr
        else:
            amplitude = [amplitude for channel in range(self.num_analised_channels)]
            
        for channel in self.ref_channels+self.roving_channels:
            
            self.measurement[:,channel] += np.random.normal(0,amplitude[channel],self.total_time_steps)
        
        
    def get_fft(self):
        
        if self.ft_freq is None or self.sum_ft is None:
            self.ft_freq, self.sum_ft = self.calculate_fft()
        return self.ft_freq, self.sum_ft
    
    def calculate_fft(self):  
        print("Calculating FFT's")
        
            
        ft_freq=None
        sum_ft=None
        for column in range(self.measurement.shape[1]):
            sample_signal = self.measurement[:,column]  
   
            # one-dimensional averaged discrete Fourier Transform for real input
            section_length = 2048
            overlap = 0.5 * section_length
            increment = int(section_length - overlap)
            num_average = (len(sample_signal) - section_length) // increment
            if num_average<1:
                continue
            for iii in range(num_average):
                this_signal = sample_signal[(iii * increment):(iii * increment + section_length)]
                
                ft = np.fft.rfft(this_signal * np.hanning(len(this_signal)))
                ft = abs(ft)
                if iii == 0:
                    average_ft = np.zeros(len(ft))
                average_ft = average_ft + ft
        
            average_ft = average_ft / num_average
            
            if column == 0:
                sum_ft = np.zeros(len(ft))
            
            sum_ft = sum_ft + average_ft
    
            ft_freq = np.fft.rfftfreq(section_length, d = (1/self.sampling_rate))
        

        return ft_freq, sum_ft
    
    def plot_svd_spectrum(self,):
        import matplotlib.pyplot as plot
        plot.figure( tight_layout=1)
        NFFT =  2048
        pxy,freq = plot.csd(self.measurement[:,0],self.measurement[:,0], NFFT, self.sampling_rate)
        psd_matrix = np.zeros((self.num_analised_channels,self.num_analised_channels,len(freq)), dtype=complex)
         
     
        for i in range(self.num_analised_channels):
            for j in range(self.num_analised_channels):
                pxy,freq =plot.csd(self.measurement[:,i],self.measurement[:,j], NFFT, self.sampling_rate)
                psd_matrix[i,j,:]=pxy
        svd_matrix = np.zeros((self.num_analised_channels, len(freq)))
        for i in range(len(freq)):
            u,s,vt = np.linalg.svd(psd_matrix[:,:,i])
            svd_matrix[:,i]=10*np.log(s)
            
        plot.figure( tight_layout=1)
        for i in range(self.num_analised_channels):
            plot.plot(freq,svd_matrix[i,:])
             
        #plot.margins(0,0.1,tight=1)
        plot.xlim((0,self.sampling_rate/2))
        plot.grid(1)
        plot.xlabel('Frequenz [\si{\hertz}]')
        plot.ylabel('Singul\\"arwert Magnitude [\si{\decibel}]')
        plot.yticks([0,-25,-50,-75,-100,-125,-150,-175,-200,-225,-250])
        plot.ylim((-225,0))
        plot.xlim((0.1,5))
        #plot.xticks([0,1,3])
        plot.grid(b=0)
        #plot.gca().xaxis.set_label_coords(0.5, -0.035)
        
        plot.show()
        
# from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QPushButton,\
#     QCheckBox, QButtonGroup, QLabel, QComboBox, \
#     QTextEdit, QGridLayout, QFrame, QVBoxLayout, QAction,\
#     QFileDialog,  QMessageBox, QApplication, QRadioButton,\
#     QLineEdit, QSizePolicy, QDoubleSpinBox
# from PyQt5.QtGui import QIcon, QPalette
# from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot,  QObject, qInstallMessageHandler, QTimer, QEventLoop
# #make qt application not crash on errors
# def my_excepthook(type, value, tback):
#     # log the exception here
# 
#     # then call the default handler
#     sys.__excepthook__(type, value, tback)
# 
# sys.excepthook = my_excepthook
# 
# class PreProcessGUI(QMainWindow):
# 
#     def __init__(self, prep_data):
# 
#         QMainWindow.__init__(self)
#         self.setWindowTitle('Preprocessor: {} - {}'.format(
#             prep_data.setup_name, prep_data.start_time))
# 
#         self.prep_data = prep_data
#         
#         self.create_menu()
#         self.create_main_frame()
# 
#         
# 
#         self.setGeometry(300, 300, 1000, 600)
#         # self.setWindowModality(Qt.ApplicationModal)
#         self.showMaximized()
# 
# '''
#         set up all the widgets and other elements to draw the GUI
#         '''
#         main_frame = QWidget()
# 
#         df_max = self.stabil_calc.df_max * 100
#         dd_max = self.stabil_calc.dd_max * 100
#         dmac_max = self.stabil_calc.dmac_max * 100
#         d_range = self.stabil_calc.d_range
#         mpc_min = self.stabil_calc.mpc_min
#         mpd_max = self.stabil_calc.mpd_max
# 
#         self.fig = self.stabil_plot.fig
#         #self.canvas = self.stabil_plot.fig.canvas
#         # print(self.canvas)
#         self.canvas = FigureCanvasQTAgg(self.fig)
# 
#         # self.stabil_plot.reconnect_cursor()
# 
#         self.canvas.setParent(main_frame)
#         if self.stabil_plot.cursor is None:
#             self.stabil_plot.init_cursor()
#         self.stabil_plot.cursor.show_current_info.connect(
#             self.update_value_view)
#         self.stabil_plot.cursor.mode_selected.connect(self.mode_selector_add)
#         self.stabil_plot.cursor.mode_deselected.connect(
#             self.mode_selector_take)
#         main_layout = QHBoxLayout()
# 
#         left_pane_layout = QVBoxLayout()
#         left_pane_layout.addStretch(1)
#         palette = QPalette()
#         palette.setColor(QPalette.Base, Qt.transparent)
# 
#         self.current_value_view = QTextEdit()
#         self.current_value_view.setFrameShape(QFrame.Box)
#         self.current_value_view.setPalette(palette)
# 
#         self.diag_val_widget = QWidget()
# 
#         fra_1 = QFrame()
#         fra_1.setFrameShape(QFrame.Panel)
#         fra_1.setLayout(self.create_stab_val_widget(df_max=df_max,
#                                                     dd_max=dd_max, d_mac=dmac_max, d_range=d_range, mpc_min=mpc_min,
#                                                     mpd_max=mpd_max))
#         left_pane_layout.addWidget(fra_1)
# 
#         left_pane_layout.addStretch(2)
# 
#         fra_2 = QFrame()
#         fra_2.setFrameShape(QFrame.Panel)
#         fra_2.setLayout(self.create_diag_val_widget())
#         left_pane_layout.addWidget(fra_2)
# 
#         left_pane_layout.addStretch(2)
#         left_pane_layout.addWidget(self.current_value_view)
#         left_pane_layout.addStretch(1)
# 
#         right_pane_layout = QVBoxLayout()
# 
#         self.plot_selector_c = QRadioButton('Mode Shape in Complex Plane')
#         self.plot_selector_c.toggled.connect(self.toggle_cpl_plot)
#         self.plot_selector_msh = QRadioButton('Mode Shape in Spatial Model')
#         self.plot_selector_msh.toggled.connect(self.toggle_msh_plot)
# 
#         self.group = QButtonGroup()
#         self.group.addButton(self.plot_selector_c)
#         self.group.addButton(self.plot_selector_msh)
# 
#         self.mode_selector = QComboBox()
#         self.mode_selector.currentIndexChanged[
#             int].connect(self.update_mode_val_view)
# 
#         self.mode_plot_widget = QWidget()
#         self.cmplx_plot_widget = QWidget()
# 
#         self.cmpl_plot = cmpl_plot
#         fig = self.cmpl_plot.fig
#         canvas1 = FigureCanvasQTAgg(fig)
#         canvas1.setParent(self.cmplx_plot_widget)
# 
#         self.msh_plot = msh_plot
#         if msh_plot is not None:
#             fig = msh_plot.fig
#             #FigureCanvasQTAgg.resizeEvent = resizeEvent_
#             #canvas2 = fig.canvas.switch_backends(FigureCanvasQTAgg)
#             #canvas2.resizeEvent = types.MethodType(resizeEvent_, canvas2)
#             canvas2 = fig.canvas
#             #self.canvas.resize_event = resizeEvent_
#             #self.canvas.resize_event  = funcType(resizeEvent_, self.canvas, FigureCanvasQTAgg)
#             #msh_plot.canvas = canvas2
#             #self.canvas.mpl_connect('button_release_event', self.update_lims)
#             #if fig.get_axes():
#             #    fig.get_axes()[0].mouse_init()
#             canvas2.setParent(self.mode_plot_widget)
#         else:
#             canvas2 = QWidget()
# 
#         lay = QHBoxLayout()
#         lay.addWidget(canvas1)
#         self.cmplx_plot_widget.setLayout(lay)
#         self.cmpl_plot.plot_diagram()
# 
#         lay = QHBoxLayout()
#         lay.addWidget(canvas2)
#         self.mode_plot_widget.setLayout(lay)
# 
#         self.mode_val_view = QTextEdit()
#         self.mode_val_view.setFrameShape(QFrame.Box)
# 
#         self.mode_val_view.setPalette(palette)
#         right_pane_layout.addStretch(1)
#         right_pane_layout.addWidget(self.mode_selector)
# 
#         right_pane_layout.addWidget(self.plot_selector_c)
#         right_pane_layout.addWidget(self.plot_selector_msh)
#         right_pane_layout.addStretch(2)
#         self.mode_plot_layout = QVBoxLayout()
#         self.mode_plot_layout.addWidget(self.cmplx_plot_widget)
#         right_pane_layout.addLayout(self.mode_plot_layout)
#         right_pane_layout.addStretch(2)
#         right_pane_layout.addWidget(self.mode_val_view)
#         right_pane_layout.addStretch(1)
# 
#         main_layout.addLayout(left_pane_layout)
#         main_layout.addWidget(self.canvas)
#         main_layout.setStretchFactor(self.canvas, 1)
#         main_layout.addLayout(right_pane_layout)
#         vbox = QVBoxLayout()
#         vbox.addLayout(main_layout)
#         vbox.addLayout(self.create_buttons())
#         main_frame.setLayout(vbox)
#         self.stabil_plot.fig.set_facecolor('none')
#         self.setCentralWidget(main_frame)
#         self.current_mode = (0, 0)
# 
#         return
# 
#     def create_buttons(self):
#         b0 = QPushButton('Apply')
#         b0.released.connect(self.update_stabil_view)
#         b1 = QPushButton('Save Figure')
#         b1.released.connect(self.save_figure)
# 
#         b2 = QPushButton('Export Results')
#         b2.released.connect(self.save_results)
#         b3 = QPushButton('Save State')
#         b3.released.connect(self.save_state)
#         b4 = QPushButton('OK and Close')
#         b4.released.connect(self.close)
# 
#         lay = QHBoxLayout()
# 
#         lay.addWidget(b0)
#         lay.addWidget(b1)
#         lay.addWidget(b2)
#         lay.addWidget(b3)
#         lay.addWidget(b4)
#         lay.addStretch()
# 
#         return lay
#     def create_menu(self):
#         '''
#         create the menubar and add actions to it
#         '''
#         def add_actions(target, actions):
#             for action in actions:
#                 if action is None:
#                     target.addSeparator()
#                 else:
#                     target.addAction(action)
#                     
# 
#         def create_action(text, slot=None, shortcut=None,
#                           icon=None, tip=None, checkable=False,
#                           signal="triggered()"):
#             action = QAction(text, self)
#             if icon is not None:
#                 action.setIcon(QIcon(":/%s.png" % icon))
#             if shortcut is not None:
#                 action.setShortcut(shortcut)
#             if tip is not None:
#                 action.setToolTip(tip)
#                 action.setStatusTip(tip)
#             if slot is not None:
#                 getattr(action, signal.strip('()')).connect(slot)
#             if checkable:
#                 action.setCheckable(True)
#             return action
# 
#         file_menu = self.menuBar().addMenu("&File")
# 
#         load_file_action = create_action("&Save plot",
#                                          shortcut="Ctrl+S",
#                                          slot=None,
#                                          tip="Save the plot")
#         quit_action = create_action("&Quit",
#                                     slot=self.close,
#                                     shortcut="Ctrl+Q",
#                                     tip="Close the application")
# 
#         add_actions(file_menu,
#                     (load_file_action, None, quit_action))
# 
#         help_menu = self.menuBar().addMenu("&Help")


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
    
    def handler(msg_type, msg_string):
        pass

    if not 'app' in globals().keys():
        global app
        app = QApplication(sys.argv)
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)

    # qInstallMessageHandler(handler) #suppress unimportant error msg
    prep_data = PreprocessData.load_state('/vegas/scratch/womo1998/towerdata/towerdata_results_var/Wind_kontinuierlich__9_2016-10-05_04-00-00_000000/prep_data.npz')
    #prep_data = None
    preprocess_gui = PreProcessGUI(prep_data)
    loop = QEventLoop()
    preprocess_gui.destroyed.connect(loop.quit)
    loop.exec_()
    print('Exiting GUI')

    return

def example():
    path = '/ismhome/staff/womo1998/Projects/2017_Burscheid/Messdaten/2017_10_25_asc_Dateien/Messung_3.asc'
    #measurement = np.loadtxt(path)
    measurement = np.load('/ismhome/staff/womo1998/Projects/2017_Burscheid/Messdaten/2017_10_25_asc_Dateien/Messung_3.npz')
    prep_data = PreprocessData(measurement, sampling_rate=128)
    print('test')
    
if __name__ =='__main__':
    example()
    #main()
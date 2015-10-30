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
import os
import csv
import sys


class GeometryProcessor(object):

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
                 setup_name=None):
        
        super().__init__()
        
        assert isinstance(measurement, np.ndarray)
        assert measurement.shape[0] > measurement.shape[1]
        self.measurement = measurement
        
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
            roving_channels = []
        self.roving_channels = roving_channels
        
        self.num_ref_channels = len(self.ref_channels)
        self.num_roving_channels = len(self.roving_channels)
        self.num_analised_channels = self.num_ref_channels + self.num_roving_channels
        if num_channels is None:
            num_channels = self.num_analised_channels
            
        
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
        
        for channel in self.ref_channels+self.roving_channels:
            assert (channel in accel_channels) + (channel in velo_channels) + (channel in disp_channels) <= 1
            
        self.accel_channels = accel_channels
        self.velo_channels = velo_channels
        self.disp_channels = disp_channels
        
        if setup_name is None:            
            setup_name = ''
        assert isinstance(setup_name, str)
            
        self.setup_name = setup_name
                
    
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
            sampling_rate= int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Reference Channels:'
            ref_channels=f.__next__().strip('\n').split(' ')
            if ref_channels:
                ref_channels=[int(val) for val in ref_channels]
            assert f.__next__().strip('\n').strip(' ')== 'Delete Channels:'
            delete_channels=f.__next__().strip('\n ').split(' ')
            if delete_channels:
                delete_channels=[int(val) for val in delete_channels]
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
        
        measurement = cls.load_measurement_file(meas_file, ftype=kwargs.get('ftype','ascii'), **kwargs)
        
                    
        if chan_dofs_file is not None:
            chan_dofs = cls.load_chan_dofs(chan_dofs_file)

        if delete_channels:
            #delete_channels.sort(reverse=True)
            
            names=['Reference Channels', 'Accel. Channels', 'Velo. Channels', 'Disp. Channels']
            channel_lists=[ref_channels, accel_channels, velo_channels, disp_channels]
            
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
                channel -= 1
                
            measurement=np.delete(measurement, delete_channels, axis=1)
        total_time_steps = measurement.shape[0]
        num_channels = measurement.shape[1]
        roving_channels = [i for i in range(num_channels) if i not in ref_channels]
        if not accel_channels and not velo_channels and not disp_channels:
            accel_channels = [i for i in range(num_channels)]
        prep_data = cls(measurement, sampling_rate, total_time_steps, 
                 num_channels,ref_channels, roving_channels,
                 accel_channels, velo_channels, disp_channels)
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
                
                chan_num, node, az, elev, chan_name = [float(line[i]) if not i in [1,4] else line[i].strip(' ') for i in range(5) ]
                chan_dofs.append([chan_num, node, az, elev, chan_name])
        return chan_dofs
                
    @staticmethod
    def load_measurement_file( fname, ftype='ascii', **kwargs):
        '''
        loads a measurement file
        '''
        assert os.path.exists(fname)
        
        if ftype=='ascii':
            f=open(fname, 'r', encoding='iso-8859-1')
            measurement=np.loadtxt(f, 
                          dtype=kwargs.get('dtype',float), 
                          comments=kwargs.get('comments','#'), 
                          delimiter=kwargs.get('delimiter',None),
                          converters=kwargs.get('converters',None), 
                          skiprows=kwargs.get('skiprows',0), 
                          usecols=kwargs.get('usecols',None), 
                          unpack=kwargs.get('unpack',False),
                          ndmin=kwargs.get('ndmin',0))
            assert measurement.shape [0] > measurement.shape [1]
        else:
            raise RuntimeError('Specified filetype: {} not implemented yet!'.format(ftype))
        
        return measurement
    
    def add_geometry_data(self, geometry_data):
        #print(type(geometry_data))
        if not isinstance(geometry_data, GeometryProcessor):
            raise RuntimeError('Geometry Data has to be loaded and '
                               'Processed by the GeometryProcessor class. '
                               'The provided object is of type {}'.format(type(geometry_data)))
        
        self.geometry_data = geometry_data
    
    def add_chan_dofs(self,chan_dofs):
        self.chan_dofs=chan_dofs
    
    def save_state(self, fname):
        
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
        
        out_dict['self.geometry_data'] = self.geometry_data

        np.savez(fname, **out_dict)
         
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
        
        accel_channels =  list(in_dict['self.accel_channels'])
        velo_channels = list(in_dict['self.velo_channels'])
        disp_channels = list(in_dict['self.disp_channels'])
        preprocessor = cls(measurement, sampling_rate, total_time_steps, 
                 ref_channels=ref_channels, roving_channels=roving_channels,
                 accel_channels=accel_channels, velo_channels=velo_channels, disp_channels=disp_channels, setup_name=setup_name)
        assert preprocessor.num_ref_channels == int(in_dict['self.num_ref_channels'])
        assert preprocessor.num_roving_channels == int(in_dict['self.num_roving_channels'])
        assert preprocessor.num_analised_channels == int(in_dict['self.num_analised_channels'])
        
        preprocessor.add_geometry_data(in_dict['self.geometry_data'].item())  
        return preprocessor
    
    def filter_data(self, filter_parameters):
        '''
        Applies various filters to the data
        '''
        pass
    
    
    def correct_offset(self, x=None):
        '''
        corrects a constant offset from measurement data
        Eliminates displacement of the measurement data originated by initial tension
        by subtracting the average value of the x first measurements from every
        value
        '''
        print('Correcting offset of {} first measurements.'.format(x))
        for ii in range(self.measurement.shape[1]):
            tmp = self.measurement[:,ii]
            if x is not None:
                self.measurement[:,ii] = tmp - tmp[0:x].mean(0)
            else:
                self.measurement[:,ii] = tmp - tmp.mean(0)

    def decimate_data(self, decimate_factor):  
        '''
        decimates the measurement data by the supplied factor
        makes use of scipy's decimate filter
        '''  
        
        num_channels = self.measurement.shape[1]  
        
        for ii in range(self.measurement.shape [1]):
            if ii == 0:
                tmp = signal.decimate(self.measurement[:,ii], decimate_factor, axis = 0)
                meas_decimated = np.zeros((tmp.shape[0],num_channels))       
                meas_decimated[:,ii] = tmp
            else:
                meas_decimated[:,ii] = signal.decimate(self.measurement[:,ii], decimate_factor, axis = 0) 
                
        self.measurement = meas_decimated
    
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
    
    def calculate_fft(self):  
        print("Calculating FFT's")
        
            
        sum_ft=None
        for column in range(self.measurement.shape[1]):
            sample_signal = self.measurement[:,column]  
   
            # one-dimensional averaged discrete Fourier Transform for real input
            section_length = 2048
            overlap = 0.5 * section_length
            increment = int(section_length - overlap)
            num_average = (len(sample_signal) - section_length) // increment
                
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
    
def main():
    working_dir='/home/womo1998/Projects/modal_analysis_tower/'
    conf_file=working_dir+'modal_source_files/meas_1/setup_info'
    meas_file=working_dir+'towerdata/Alle_3h_00_00__1_2015-08-27_00-02-59_690000.dat.txt'
    nodes_file =working_dir+ 'modal_source_files/nodes'
    lines_file =working_dir+ 'modal_source_files/lines'
    master_slaves_file=working_dir+'modal_source_files/master_slaves'
    chan_dofs_file=working_dir+'modal_source_files/meas_1/channel_dofs'
    
    geometry_data = GeometryProcessor.load_geometry(nodes_file, lines_file, master_slaves_file)
    
    preprocessor = PreprocessData.init_from_config(conf_file, meas_file, chan_dofs_file, skiprows=40)
    
    preprocessor.add_geometry_data(geometry_data)
    preprocessor.correct_offset()
    preprocessor.save_state(working_dir+'test.npz')
    
    preprocessor = PreprocessData.load_state(working_dir+'test.npz')

    
if __name__ =='__main__':
    main()
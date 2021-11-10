#
# -*- coding: utf-8 -*-
'''
pyOMA - A toolbox for Operational Modal Analysis
Copyright (C) 2015 - 2021  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


.. TODO::
     * correct linear,.... offsets as well
     * implement loading of different filetypes ascii, lvm, ...
     * currently loading geometry, etc. files will overwrite existing assignments implement "load and append"
     * implement windowing functions
     * Proper documentation
     * add test to tests package
     * Remove multiprocessing routines, since numpy is parallelized already
       and proper vectorization of the code is better than just using multiprocessing to speed up bad code
     * FIX: Unify correlation function definitions welch/b-t some start at 0 some at 1 

'''

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import os
import csv
import sys
import datetime
import time

import multiprocessing as mp
import ctypes as c
import logging
#logging.basicConfig(stream=sys.stdout)  # done at module level
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def nearly_equal(a, b, sig_fig=5):
    return (a == b or
            int(a * 10**sig_fig) == int(b * 10**sig_fig)
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

    .. TODO::
         * change master_slave assignment to skewed coordinate
         * change master_slaves to az, elev
    '''

    def __init__(self, nodes={}, lines=[], master_slaves=[]):
        super().__init__()
        self.nodes = {}
        assert isinstance(nodes, dict)
        self.add_nodes(nodes)
        
        self.lines = []
        assert isinstance(lines, (list, tuple, np.ndarray))
        self.add_lines(lines)
        
        self.master_slaves = []
        assert isinstance(master_slaves, (list, tuple, np.ndarray))
        self.add_master_slaves(master_slaves)

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
                line = []
                for val in line1:
                    if not val:
                        continue
                    line += val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                node, x, y, z = [float(line[i]) if i >= 1 else line[i].strip(
                    ' ') for i in range(4)]  # cut trailing empty columns
                nodes[node] = [x, y, z]
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
                line = []
                for val in line1:
                    if not val:
                        continue
                    line += val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                node_start, node_end = \
                    [line[i] for i in range(2)]  # cut trailing empty columns
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
                line = []
                for val in line1:
                    if not val:
                        continue
                    line += val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl = [
                    float(line[i]) if i not in [0, 4] else line[i].strip(' ') for i in range(8)]
                master_slaves.append(
                    (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl))
        return master_slaves

    @classmethod
    def load_geometry(
            cls,
            nodes_file,
            lines_file=None,
            master_slaves_file=None):
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
            master_slaves = geometry_data.master_slaves_loader(
                master_slaves_file)
            geometry_data.add_master_slaves(master_slaves)

        return geometry_data

    def add_nodes(self, nodes):
        for item in nodes.items():
            try:
                self.add_node(*item)
            except BaseException:
                logger.warning(
                    'Something was wrong while adding node {}. Continuing!'.format(item))
                continue

    def add_node(self, node_name, coordinate_list):
        node_name = str(node_name)
        if node_name in self.nodes.keys():
            logger.warning('Node {} is already defined. Overwriting.'.format(node_name))

        if not isinstance(coordinate_list, (list, tuple)):
            raise RuntimeError(
                'Coordinates must be provided as (x,y,z) tuples/lists.')
        if len(coordinate_list) != 3:
            raise RuntimeError(
                'Coordinates must be provided as (x,y,z) tuples/lists.')

        try:
            node_name = str(node_name)
            coordinate_list = list(coordinate_list)
            for i in range(3):
                coordinate_list[i] = float(coordinate_list[i])
        except ValueError:
            raise RuntimeError(
                'Coordinate {} at position {} could not be converted to float.'.format(
                    coordinate_list[i], i))
        except BaseException:
            raise

        self.nodes[node_name] = tuple(coordinate_list)

    def take_node(self, node_name):
        if node_name not in self.nodes:
            logger.warning('Node not defined. Exiting')
            return

        while True:  # check if any line is connected to this node
            for j in range(len(self.lines)):
                line = self.lines[j]
                if node_name in line:
                    del self.lines[j]
                    break
            else:
                break

        while True:  # check if this node is a master or slave for another node
            for j, master_slave in enumerate(self.master_slaves):
                if node_name == master_slave[0] or node_name == master_slave[4]:
                    m = master_slave
                    del self.master_slaves[j]
                    break
            else:
                break
        del self.nodes[node_name]

        logger.info('Node {} removed.'.format(node_name))

    def add_lines(self, lines):

        for line in lines:
            try:
                self.add_line(line)
            except BaseException:
                logger.warning(
                    'Something was wrong while adding line {}. Continuing!'.format(line))
                continue

    def add_line(self, line):
        if not isinstance(line, (list, tuple)):
            raise RuntimeError(
                'Line has to be provided in format (start_node, end_node).')
        if len(line) != 2:
            raise RuntimeError(
                'Line has to be provided in format (start_node, end_node).')

        line = [str(line[0]), str(line[1])]
        if line[0] not in self.nodes or line[1] not in self.nodes:
            logger.warning('One of the end-nodes of line {} not defined!'.format(line))
        else:
            for line_ in self.lines:
                if line_[0] == line[0] and line_[1] == line[1]:
                    logger.info('Line {} was defined, already.'.format(line))
            else:
                self.lines.append(line)

    def take_line(self, line=None, line_ind=None):
        assert line is None or line_ind is None

        if line is not None:
            for line_ind in range(len(self.lines)):
                line_ = self.lines[line_ind]
                if line[0] == line_[0] and line[1] == line_[1]:
                    break
            else:
                logger.warning('Line {} was not found.'.format(line))
                return
        del self.lines[line_ind]
        logger.info('Line {} at index {} removed.'.format(line, line_ind))

    def add_master_slaves(self, master_slaves):
        for ms in master_slaves:
            try:
                self.add_master_slave(ms)
            except BaseException:
                logger.warning(
                    'Something was wrong while adding master-slave-definition {}. Continuing!'.format(ms))
                continue

    def add_master_slave(self, ms):
        if not isinstance(ms, (list, tuple)):
            raise RuntimeError(
                'master slave definition has to be provided in format (start_node, end_node).')
        if len(ms) != 8:
            raise RuntimeError(
                'master slave definition has to be provided in format (master_node, x_ampli, y_ampli, z_ampli, slave_node, x_ampli, y_ampli, z_ampli).')
        ms = (
            str(
                ms[0]), float(
                ms[1]), float(
                ms[2]), float(
                    ms[3]), str(
                        ms[4]), float(
                            ms[5]), float(
                                ms[6]), float(
                                    ms[7]))
        if ms[0] not in self.nodes or ms[4] not in self.nodes:
            logger.warning(
                'One of the nodes of master slave definition {} not defined!'.format(ms))
        else:
            for ms_ in self.master_slaves:
                b = False
                for i in range(8):
                    b = b and ms_[i] == ms[i]
                if b:
                    logger.info(
                        'master slave definition {} was defined, already.'.format(ms))
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
                logger.warning('master slave definition {} was not found.'.format(ms))
                return

        del self.master_slaves[ms_ind]
        logger.info('master slave definition {} at index {} removed.'.format(ms, ms_ind))

    def rescale_geometry(self, factor):
        pass


class PreProcessSignals(object):
    '''
    A simple pre-processor for signals
    * load ascii datafiles
    * specify sampling rate, reference channels and roving channels
    * specify geometry, channel-dof-assignments
    * specify channel quantities such as acceleration, velocity, etc
    * remove channels, cut time histories
    * remove (constant) offsets from time history data
    * decimate time histories
    TODO :
    * calculate fft, psd, covariance, coherence, frf
    * integrate
    * apply windowing functions
    '''

    def __init__(self, measurement, sampling_rate, total_time_steps=None,
                 # num_channels=None,
                 ref_channels=None,
                 accel_channels=None, velo_channels=None, disp_channels=None,
                 setup_name=None, channel_headers=None, start_time=None,
                 ft_freq=None, sum_ft=None, F=None, **kwargs):

        super().__init__()

        assert isinstance(measurement, np.ndarray)
        assert measurement.shape[0] > measurement.shape[1]
        self.measurement = measurement
        self.measurement_filt = measurement
        
        self.scaling_factors = None
        
        assert isinstance(sampling_rate, (int, float))
        self.sampling_rate = sampling_rate

        # added by anil
        if F is not None:
            assert isinstance(F, np.ndarray)
        self.F = F

        if total_time_steps is None:
            total_time_steps = measurement.shape[0]

        assert measurement.shape[0] >= total_time_steps

        self.total_time_steps = total_time_steps

        if ref_channels is None:
            ref_channels = list(range(measurement.shape[1]))
        self.ref_channels = ref_channels
#         if roving_channels is None:
#             roving_channels = [i for i in range(measurement.shape[1]) if i not in ref_channels]
#         self.roving_channels = roving_channels

        self.num_ref_channels = len(self.ref_channels)
#         self.num_roving_channels = len(self.roving_channels)
        # self.num_ref_channels + self.num_roving_channels
        self.num_analised_channels = measurement.shape[1]

        # if num_channels is None:
        #    num_channels = self.num_analised_channels

        #assert num_channels <= self.measurement.shape[1]

        # if ((self.num_ref_channels + self.num_roving_channels) > num_channels):
        #        sys.exit('The sum of reference and roving channels is greater than the number of all channels!')

        for ref_channel in self.ref_channels:
            if (ref_channel < 0):
                sys.exit('A reference channel number cannot be negative!')
            if (ref_channel > (self.num_analised_channels - 1)):
                sys.exit(
                    'A reference channel number cannot be greater than the number of all channels!')
            # for rov_channel in self.roving_channels:
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
            accel_channels = [
                c for c in range(
                    self.num_analised_channels) if c not in disp_channels and c not in velo_channels]

        for channel in range(self.num_analised_channels):
            if (channel in accel_channels) + (channel in velo_channels) + \
                    (channel in disp_channels) != 1:

                logger.warning(
                    'Measuring quantity of channel {} is not defined.'.format(channel))

        self.accel_channels = accel_channels
        self.velo_channels = velo_channels
        self.disp_channels = disp_channels

        # print(self.accel_channels,self.velo_channels,self.disp_channels)

        if setup_name is None:
            setup_name = ''
        assert isinstance(setup_name, str)

        self.setup_name = setup_name

        if channel_headers is not None:
            assert len(channel_headers) == self.num_analised_channels
        else:
            channel_headers = list(range(self.num_analised_channels))

        self.channel_headers = channel_headers

        if start_time is not None:
            assert isinstance(start_time, datetime.datetime)
        else:
            start_time = datetime.datetime.now()
        self.start_time = start_time
        # print(self.start_time)
        #self.geometry_data = None

        self.channel_factors = [
            1 for channel in range(
                self.measurement.shape[1])]

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
    def init_from_config(
            cls,
            conf_file,
            meas_file,
            chan_dofs_file=None,
            **kwargs):
        '''
        initializes the PreProcessor object with a configuration file

        to remove channels at loading time use 'usecols' keyword argument
        if delete_channels are specified, these will be checked against
        all other channel definitions, which will be adjusted accordingly
        '''
        if not os.path.exists(conf_file):
            raise RuntimeError(
                'Conf File does not exist: {}'.format(conf_file))

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Setup Name:'
            name = f. __next__().strip('\n')
            assert f.__next__().strip('\n').strip(' ') == 'Sampling Rate [Hz]:'
            sampling_rate = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Reference Channels:'
            ref_channels = f.__next__().strip('\n').split(' ')
            if ref_channels:
                ref_channels = [int(val)
                                for val in ref_channels if val.isnumeric()]
            assert f.__next__().strip('\n').strip(' ') == 'Delete Channels:'
            delete_channels = f.__next__().strip('\n ').split(' ')
            if delete_channels:
                delete_channels = [
                    int(val) for val in delete_channels if val.isnumeric()]
            assert f.__next__().strip('\n').strip(' ') == 'Accel. Channels:'
            accel_channels = f.__next__().strip('\n ').split()
            if accel_channels:
                accel_channels = [int(val) for val in accel_channels]
            assert f.__next__().strip('\n').strip(' ') == 'Velo. Channels:'
            velo_channels = f.__next__().strip('\n ').split()
            if velo_channels:
                velo_channels = [int(val) for val in velo_channels]
            assert f.__next__().strip('\n').strip(' ') == 'Disp. Channels:'
            disp_channels = f.__next__().strip('\n ').split()
            if disp_channels:
                disp_channels = [int(val) for val in disp_channels]

        loaded_signals = cls.load_measurement_file(meas_file, **kwargs)

        if not isinstance(loaded_signals, np.ndarray):
            # print(loaded_signals)
            headers, units, start_time, sample_rate, measurement = loaded_signals
        else:
            measurement = loaded_signals
            start_time = datetime.datetime.now()
            sample_rate = sampling_rate
            headers = ['Channel_{}'.format(i)
                       for i in range(measurement.shape[1])]
        if not sample_rate == sampling_rate:
            logger.warning(
                'Sampling Rate from file: {} does not correspond with specified Sampling Rate from configuration {}'.format(
                    sample_rate, sampling_rate))
        # print(headers)

        if chan_dofs_file is not None:
            chan_dofs = cls.load_chan_dofs(chan_dofs_file)
        else:
            chan_dofs = None

        if chan_dofs is not None:
            for chan_dof in chan_dofs:
                if len(chan_dof) == 5:
                    chan = chan_dof[0]
                    chan_name = chan_dof[4]
                    if len(chan_name) == 0:
                        continue
                    elif headers[chan] == 'Channel_{}'.format(chan):
                        headers[chan] = chan_name
                    elif headers[chan] != chan_name:
                        logger.info(
                            'Different headers for channel {} in measurement file ({}) and in channel-DOF-assignment ({}).'.format(
                                chan, headers[chan], chan_name))
                    else:
                        continue

        # print(delete_channels)
        if delete_channels:
            # delete_channels.sort(reverse=True)

            names = [
                'Reference Channels',
                'Accel. Channels',
                'Velo. Channels',
                'Disp. Channels']
            channel_lists = [
                ref_channels,
                accel_channels,
                velo_channels,
                disp_channels]
            # print(chan_dofs)

            num_all_channels = measurement.shape[1]
            #print(chan_dofs, ref_channels, accel_channels, velo_channels,disp_channels, headers)
            new_chan_dofs = []
            new_ref_channels = []
            new_accel_channels = []
            new_velo_channels = []
            new_disp_channels = []
            new_headers = []
            new_channel = 0
            for channel in range(num_all_channels):
                if channel in delete_channels:
                    logger.info(
                        'Now removing Channel {} (no. {})!'.format(
                            headers[channel], channel))
                    continue
                else:
                    for chan_dof in chan_dofs:
                        if chan_dof[0] == channel:
                            node, az, elev = chan_dof[1:4]
                            if len(chan_dof) == 5:
                                cname = chan_dof[4]
                            else:
                                cname = ''
                            break
                    else:
                        logger.warning('Could not find channel in chan_dofs')
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
        # print(measurement)
        prep_signals = cls(measurement, sampling_rate, total_time_steps,
                        # num_channels,
                        ref_channels,  # roving_channels,
                        accel_channels, velo_channels, disp_channels,
                        channel_headers=headers, start_time=start_time,
                        setup_name=name, **kwargs)
        if chan_dofs:
            prep_signals.add_chan_dofs(chan_dofs)

        return prep_signals

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
                line = []
                for val in line1:
                    if not val:
                        continue
                    line += val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                while len(line) <= 5:
                    line.append('')
                chan_num, node, az, elev, chan_name = [
                    line[i].strip(' ') for i in range(5)]
                chan_num, az, elev = int(
                    float(chan_num)), float(az), float(elev)
                #print(chan_num, node, az, elev)
                if node == 'None':
                    node = None
                    # print(None)
                chan_dofs.append([chan_num, node, az, elev, chan_name])
        return chan_dofs

    @staticmethod
    def load_measurement_file(fname, **kwargs):
        '''
        A method for loading a measurement file

        Parameters
        ----------
        fname : str
                The full path of the measurement file

        Returns
        -------
        headers : list of str
                The names of all channels
        units : list of str
                The units of all channels
        start_time : datetime.datetime
                The starting time of the measured signal
        sample_rate : float
                The sample rate, at wich the signal was acquired
        measurement : ndarray
                Array of shape (num_timesteps, num_channels) which contains
                the acquired signal
        '''

        raise NotImplementedError(
            'This method must be provided by the user for each specific analysis task and assigned to the class before instantiating the instance.')
        headers = None
        units = None
        start_time = None
        sample_rate = None
        measurement = None

        return headers, units, start_time, sample_rate, measurement

    def add_chan_dofs(self, chan_dofs):
        '''
        chan_dofs = [ (chan_num, node_name, az, elev, chan_name) ,  ... ]
        This function is not checking if channels or nodes actually exist
        the former should be added
        the latter might only be possible, if the geometry object is known to the class
        
        '''
        for chan_dof in chan_dofs:
            chan_dof[0] = int(chan_dof[0])
            if chan_dof[1] is not None:
                chan_dof[1] = str(chan_dof[1])
            chan_dof[2] = float(chan_dof[2])
            chan_dof[3] = float(chan_dof[3])
            if len(chan_dof) == 4:
                chan_dof.append('')
            self.chan_dofs.append(chan_dof)
        # self.chan_dofs=chan_dofs

    def take_chan_dof(self, chan, node, dof):

        for j in range(len(self.chan_dofs)):
            if self.chan_dofs[j][0] == chan and \
               self.chan_dofs[j][1] == node and \
               nearly_equal(self.chan_dofs[j][2][0], dof[0], 3) and \
               nearly_equal(self.chan_dofs[j][2][1], dof[1], 3) and \
               nearly_equal(self.chan_dofs[j][2][2], dof[2], 3):
                del self.chan_dofs[j]
                break
        else:
            if self.chan_dofs:
                logger.warning('chandof not found')

    def save_state(self, fname):

        #print('fname = ', fname)

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {}
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
        out_dict['self.accel_channels'] = self.accel_channels
        out_dict['self.velo_channels'] = self.velo_channels
        out_dict['self.disp_channels'] = self.disp_channels
        out_dict['self.chan_dofs'] = self.chan_dofs
        out_dict['self.channel_headers'] = self.channel_headers
        out_dict['self.start_time'] = self.start_time
        out_dict['self.ft_freq'] = self.ft_freq
        out_dict['self.sum_ft'] = self.sum_ft
        out_dict['self.tau_max'] = self.tau_max
        out_dict['self.corr_matrix'] = self.corr_matrix
        out_dict['self.psd_mats'] = self.psd_mats
        out_dict['self.s_vals_cf'] = self.s_vals_cf
        out_dict['self.s_vals_psd'] = self.s_vals_psd

        #out_dict['self.geometry_data'] = self.geometry_data

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname):

        logger.info('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        setup_name = str(in_dict['self.setup_name'].item())
        measurement = in_dict['self.measurement']
        sampling_rate = float(in_dict['self.sampling_rate'])
        total_time_steps = int(in_dict['self.total_time_steps'])
        ref_channels = list(in_dict['self.ref_channels'])
        #roving_channels = list(in_dict['self.roving_channels'])
        if in_dict['self.channel_headers'].shape:
            channel_headers = list(in_dict['self.channel_headers'])
        else:
            channel_headers = ['' for _ in range(
                int(in_dict['self.num_analised_channels']))]
        start_time = in_dict['self.start_time'].item()

        accel_channels = list(in_dict['self.accel_channels'])
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
        spectral_values = [None, None, None, None, None]
        for obj_num, name in enumerate(
                ['self.corr_matrix', 'self.psd_mats', 'self.s_vals_cf', 'self.s_vals_psd', 'self.tau_max']):
            try:
                spectral_values[obj_num] = in_dict[name]
            except Exception as e:
                logger.warning(repr(e))

        #sum_ft = in_dict.get( 'self.sum_ft', None)

        preprocessor = cls(measurement, sampling_rate, total_time_steps,
                           ref_channels=ref_channels,  # roving_channels=roving_channels,
                           accel_channels=accel_channels, velo_channels=velo_channels,
                           disp_channels=disp_channels, setup_name=setup_name,
                           channel_headers=channel_headers, start_time=start_time,
                           ft_freq=ft_freq, sum_ft=sum_ft)

        chan_dofs = [[int(float(chan_dof[0])), str(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3]), str(
            chan_dof[4] if 5 == len(chan_dof) else '')] for chan_dof in in_dict['self.chan_dofs']]
        preprocessor.add_chan_dofs(chan_dofs)

        preprocessor.corr_matrix = spectral_values[0]
        preprocessor.pds_mats = spectral_values[1]
        preprocessor.s_vals_cf = spectral_values[2]
        preprocessor.s_vals_psd = spectral_values[3]
        preprocessor.tau_max = int(spectral_values[4])

        assert preprocessor.num_ref_channels == int(
            in_dict['self.num_ref_channels'])
        #assert preprocessor.num_roving_channels == int(in_dict['self.num_roving_channels'])
        assert preprocessor.num_analised_channels == int(
            in_dict['self.num_analised_channels'])

        # preprocessor.add_geometry_data(in_dict['self.geometry_data'].item())
        return preprocessor

    @property
    def duration(self):
        return self.total_time_steps / self.sampling_rate
    
    @property
    def dt(self):
        return 1 / self.sampling_rate
    
    @property
    def signal_power(self):
        ref_channels = self.ref_channels
        all_channels = list(range(self.num_analised_channels))

        measurement = self.measurement

        refs = measurement[:, ref_channels]
        current_signals = measurement[:, all_channels]

        this_block = np.dot(current_signals.T, refs) / current_signals.shape[0]

        return this_block
    
    @property
    def signal_rms(self):
        self.correct_offset()
        return np.sqrt(np.mean(np.square(self.measurement), axis=0))
    
    def filter_signals(self, lowpass=None, highpass=None,
                    overwrite=True,
                    order=4, ftype='butter', RpRs=[3, 3],
                    plot_ax=None):
        logger.info('Filtering signals in the band: {} .. {} with a {} order {} filter.'.format(highpass, lowpass, order, ftype))

        if (highpass is None) and (lowpass is None):
            raise ValueError('Neither a lowpass or a highpass corner frequency was provided.')
        
        ftype_list = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel', 'moving_average', 'brickwall']
        if not (ftype in ftype_list):
            raise ValueError(f'Filter type {ftype} is not any of the available types: {ftype_list}')
        
        if order <= 1:
            raise ValueError('Order must be greater than 1')
        
        nyq = self.sampling_rate / 2
        
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

        freqs[:] = [x / nyq for x in freqs]
        measurement = self.measurement
        
        if ftype in ftype_list[0:5]:  # IIR filter
            #if order % 2:  # odd number
            #    logger.warning(f'Odd filter order {order} will be rounded up to {order+1}, because of forward-backward filtering.')
            #order = int(np.ceil(order / 2))  # reduce by factor 2 because of double filtering
            order = int(order)
            
            sos = scipy.signal.iirfilter(
                order, freqs, rp=RpRs[0], rs=RpRs[1],
                btype=btype, ftype=ftype, output='sos')
            
            measurement_filt = scipy.signal.sosfiltfilt(
                sos, measurement, axis=0)
            if self.F is not None:
                self.F_filt = scipy.signal.sosfiltfilt(sos, self.F, axis=0)
        elif ftype in ftype_list[5:7]:  # FIR filter
            if ftype == 'brickwall':
                fir_irf = scipy.signal.firwin(numtaps=order, cutoff=freqs, fs=np.pi)
            elif ftype == 'moving_average':
                if freqs:
                    logger.warning('For the moving average filter, no cutoff frequencies can be defined.')
                fir_irf = np.ones((order)) / order
            
            measurement_filt = scipy.signal.lfilter(fir_irf, [1.0], measurement, axis=0)
            if self.F is not None:
                self.F_filt = scipy.signal.lfilter(fir_irf, [1.0], self.F, axis=0)
            
        if np.isnan(measurement_filt).any():
            logger.warning('Your filtered signals contain NaNs. Check your filter settings! Continuing...')

        if plot_ax is not None:
            
            N = 2048
            
            dt = 1 / self.sampling_rate
                    
            if isinstance(plot_ax, (list, np.ndarray)):
                freq_ax = plot_ax[1]
                tim_ax = plot_ax[0]
            else:
                freq_ax = plot_ax
                tim_ax = None
                
            if ftype in ftype_list[0:5]:  # IIR Filter
                
                w, h = scipy.signal.sosfreqz(sos, worN=np.fft.rfftfreq(N) * 2 * np.pi)
                
                # convert to decibels
                # the square comes from double filtering and has nothing to do with rms or such
                # db factor 20 due to Root-Mean-Square not Mean-Square-Spectrum quantity
                frf = 20 * np.log10(abs(h)**2)
                freq_ax.plot((nyq / np.pi) * w, frf, color='lightgrey', ls='dashed')
                if tim_ax is not None:
                    irf = np.fft.irfft(h, n=10 * N)
                    
                    logger.debug(f'IRF Integral {np.sum(irf)*dt}')
                    dur = N * dt
                    t = np.linspace(0, dur - dt, 10 * N)
                    #b, a = scipy.signal.sos2tf(sos)
                    #tout, yout = scipy.signal.dimpulse((b, a, dt), n=N)
                    #tim_ax.plot(tout, np.squeeze(yout))
                    tim_ax.plot(t, irf, color='lightgrey')
                    
            else:  # FIR Filter
                
                dt = 1 / self.sampling_rate
                dur = order * dt
                
                # zero-pad the FRF to achieve spectral-interpolated IRF
                frf = np.fft.fft(fir_irf)
                if order % 2:
                    # if numtaps is odd, the maximum frequency is present additionally to the minimum,
                    # which is just a conjugate in the case of real signals
                    neg = frf[order // 2 + 1:order]
                    pos = frf[:order // 2 + 1]
                else:
                    # if numtaps is even, only the mimimum frequency is present
                    pos = frf[:order // 2]
                    neg = frf[order // 2:order]
                    # mirror the conjugate of the minimum frequency to the maximum frequency to ensure symmetry of the spectrum
                    pos = np.hstack([pos, np.conj(neg[0:1])])
                frf_pad = np.hstack([pos, np.zeros((N - order // 2 * 2 - 1,), dtype=complex), neg])
                irf_fine = np.fft.ifft(frf_pad)
                # ensure imaginary part of interpolated IRF is zero
                assert np.max(irf_fine.imag) <= np.finfo(np.float64).eps
                irf_fine = irf_fine.real
                dt_new = dur / N
                irf_fine /= dt_new / dt
                
                logger.debug(f'IRF Integral {np.sum(fir_irf) * dt}, {np.sum(irf_fine) * dt_new}')
                # zero-pad the IRF to achieve high-resolution FRF
                irf_pad = np.zeros((N,))
                irf_pad[:order] = fir_irf
                frf_fine = np.fft.fft(irf_pad)
                
                
                # convert to decibels
                frf_fine = 20 * np.log10(abs(frf_fine))
                # plot FRF and IRF
                freq_ax.plot(np.fft.fftshift(np.fft.fftfreq(N, dt)),
                             np.fft.fftshift(frf_fine), color='lightgrey', ls='dashed')
                if tim_ax is not None:
                    t = np.linspace(-dur / 2, dur / 2 - dt_new, N)
                    tim_ax.plot(t, irf_fine, color='lightgrey', )

        if overwrite:
            self.measurement = measurement_filt
            if self.F is not None:
                self.F = self.F_filt
        self.measurement_filt = measurement_filt
        
        return measurement_filt
    
    
    def plot_signals(self,
            channels=None,
            single_channels=False,
            timescale='time',
            svd_spect=False,
            axest=None,
            axesf=None,
            **kwargs):
        
        '''
        Plot time domain and/or frequency domain signals in various configurations:
         1. time history and spectrum of a single channel in two axes -> set channels = [channel] goto 2
         2. time history of multiple channels (all channels or specified) -> generate axes and arrange them in a list for each channel
             if axes argument not None, check size etc. should be an ndarray of size (num_channels, 2) regardless of the actual figure layout, put flattened axes list into columns
             
            a. time domain overlay in a single axes -> single axes is repeated in the axes list
                i. spectrum overlay in a single axes -> single axes is repeated in the axes list
                ii. svd spectrum in a single axes -> needs an additional argument
            b. in multiple axes' in a grid figure -> axes are generated as subplots
                i. spectrum in multiple axes' -> axes are generated as subplots
                ii. svd spectrum in a single axes -> needs an additional argument
        
        Parameters
        ----------
            channels : int, list, tuple, np.ndarray
                    The channels to plot
            single_channels: bool
                    Whether to plot all channels into a single or multiple axes
            timescale: str ['time', 'samples', 'lags']
                    Whether to display time, sample or lag values on the horizontal axes
                    'lags' implies plotting (auto)-correlations instead of raw time histories  
            svd_spect: bool
                    Whether to plot an SVD spectrum or regular spectra
            axest: ndarray of size num_channels of matplotlib.axes.Axes objects
                    User provided axes objects, into which to plot time domain signals
            axesf: ndarray of size num_channels of matplotlib.axes.Axes objects
                    User provided axes objects, into which to plot spectra
            **kwargs:
                     should contain figure/axes formatting options
            
        '''
        
        f_max = kwargs.pop('f_max', False)
        NFFT = kwargs.pop('NFFT', min(512, int(np.floor(0.5 * self.total_time_steps))))
        window = kwargs.pop('window', 'hamming')
        meth = kwargs.pop('method', "welch")
        
        if timescale == 'samples':
            t = np.linspace(start=0,
                            stop=self.total_time_steps,
                            num=self.total_time_steps)
        elif timescale == 'time':
            t = np.linspace(start=0,
                            stop=self.total_time_steps / self.sampling_rate,
                            num=self.total_time_steps)
        elif timescale == 'lags':
            t = np.linspace(start=0,
                            stop=NFFT / self.sampling_rate,
                            num=NFFT)
        else:
            raise ValueError(f'Type of timescale={timescale} could not be understood.')

        if channels is None:
            channels = list(range(self.num_analised_channels))
        else:
            if isinstance(channels, int):
                channels = [channels]
            assert isinstance(channels, (list, tuple, np.ndarray))
        
        num_channels = len(channels)
        
        if axest is None or axesf is None:
            if single_channels:
                if not svd_spect:
                    # creates a subplot with side by side time and frequency domain plots for each channel
                    _, axes = plt.subplots(nrows=num_channels,
                                           ncols=2,
                                           sharey='col',
                                           sharex='col',
                                           tight_layout=True)
                    if axest is None:
                        axest = axes[:, 0]
                    if axesf is None:
                        axesf = axes[:, 1]
                else:
                    if axest is None:
                        # creates a subplot for time domain plots of each channel
                        nxn = int(np.ceil(np.sqrt(num_channels)))
                        _, axest = plt.subplots(nrows=nxn,
                                                ncols=nxn,
                                                sharey=True,
                                                sharex=True,
                                                tight_layout=True)
                        axest = axest.flatten()
                    if axesf is None:
                        # creates a separate figure for the svd spectrum
                        _, axesf = plt.subplots(nrows=1,
                                                ncols=1,
                                                tight_layout=True)
                        axesf = np.repeat(axesf, num_channels)
            else:
                if axest is None:
                    # create a single figure for overlaying all time domain plots
                    _, axest = plt.subplots(nrows=1,
                                            ncols=1,
                                            tight_layout=True)
                    axest = np.repeat(axest, num_channels)
                    
                if axesf is None:
                    # create a single figure for overlaying all spectra  or svd spectrum
                    _, axesf = plt.subplots(nrows=1,
                                            ncols=1,
                                            tight_layout=True)
                    axesf = np.repeat(axesf, num_channels)
        
        # Check the provided axes objects
        if single_channels:
            if len(axest) < num_channels:
                raise ValueError(f'The number of provided axes objects '
                                 '(time domain) = {len(axest} does not match the '
                                 'number of channels={num_channels}')
        if single_channels and not svd_spect:
            if len(axesf) < num_channels:
                raise ValueError(f'The number of provided axes objects '
                                 '(frequency domain) = {len(axesf} does not match the '
                                 'number of channels={num_channels}')
        if not single_channels:
            if not isinstance(axest, (tuple, list, np.ndarray)):
                axest = np.repeat(axest, num_channels)
            elif len(axest) == 1:
                axest = np.repeat(axest, num_channels)
            elif len(axest) < num_channels:
                raise ValueError(f'The number of provided axes objects '
                                 '(time domain) = {len(axest} does not match the '
                                 'number of channels={num_channels}')
                
        if not single_channels or svd_spect:
            if not isinstance(axesf, (tuple, list, np.ndarray)):
                axesf = np.repeat(axesf, num_channels)
            elif len(axesf) == 1:
                axesf = np.repeat(axesf, num_channels)
            elif len(axesf) < num_channels:
                raise ValueError(f'The number of provided axes objects '
                                 '(frequency domain) = {len(axesf} does not match the '
                                 'number of channels={num_channels}')
        
        if meth=='welch':
            corr_matrix = self.corr_welch(NFFT, False, window)
            #psd_mats, freqs = self.psd_welch(NFFT, False, window)
            psd_mats, freqs = self.psd_mats, self.freqs
        elif meth=="b-t":
            psd_mats, freqs = self.psd_blackman_tukey(NFFT,window)
            corr_matrix = self.corr_matrix
        auto_psd = np.diagonal(psd_mats)
        auto_corr = np.diagonal(corr_matrix)
        
        for axt, axf, channel in zip(axest, axesf, channels):
            alpha = kwargs.pop('alpha', .5)
            if timescale == 'lags':
                axt.plot(t, auto_corr[:, channel], alpha=alpha,
                         label=self.channel_headers[channel], **kwargs)
            else:
                axt.plot(t, self.measurement[:, channel], alpha=alpha,
                         label=self.channel_headers[channel], **kwargs)
            axt.grid(True, axis='y', ls='dotted')
            
            if not svd_spect:
                #normalize psd to match with filter response plots
                this_psd = np.copy(auto_psd[:, channel])
                factor = self.scaling_factors[channel, channel]
                this_psd /= factor
                
                # compute RMS spectrum in decibel (factor 20 for RMS)
                this_psd = 20 * np.log10(np.sqrt(np.abs(this_psd)))
                axf.plot(freqs, this_psd, alpha=alpha,
                         label=self.channel_headers[channel], **kwargs)
                axf.grid(True, axis='x', ls='dotted')
                #axf.set_yscale('log')
        if svd_spect:
            self.plot_svd_spectrum(NFFT, log_scale=True, ax=axesf[0])
            if f_max:
                axesf[0].set_xlim((0, f_max))
        
        
        # label the last axes (which  may be a single axes repeated multiple times)
        axest[-1].set_xlabel('Time [\\si{\\second}]')
        axest[-1].set_ylabel('Amplitude [\\si{\\metre\\per\\second\\squared}]')
        axesf[-1].set_xlabel('Frequency [\\si{\\hertz}]')
        axesf[-1].set_ylabel('Magnitude [\\si{\\decibel}]')
        
        if not single_channels:
            axest[-1].legend()
            axesf[-1].legend()
        else:
            figt = axest[0].get_figure()
            figt.legend()
            figf = axesf[0].get_figure()
            figf.legend()
            
        return axest, axesf

    def correct_offset(self, x=None):
        '''
        corrects a constant offset from measured signals
        by subtracting the average value of the x first measurements from every
        value
        '''
        logger.info('Correcting offset of measured signals')
        # print(self.measurement.mean(axis=0))
        self.measurement -= self.measurement.mean(axis=0)
        # print(self.measurement.mean(axis=0))
        return

        for ii in range(self.measurement.shape[1]):
            tmp = self.measurement[:, ii]
            if x is not None:
                self.measurement[:, ii] = tmp - tmp[0:x].mean(0)
            else:
                self.measurement[:, ii] = tmp - tmp.mean(0)

    def precondition_signals(self, method='iqr'):

        assert method in ['iqr', 'range']

        self.correct_offset()

        for i in range(self.measurement.shape[1]):
            tmp = self.measurement[:, i]
            if method == 'iqr':
                factor = np.subtract(*np.percentile(tmp, [95, 5]))
            elif method == 'range':
                factor = np.max(tmp) - np.min(tmp)
            self.measurement[:, i] /= factor
            self.channel_factors[i] = factor

    def decimate_signals(
            self,
            decimate_factor,
            highpass=None,
            order=8,
            filter_type='cheby1'):
        '''
        decimates measurement data
        filter type and order are choosable (order 8 and type cheby1 are standard for scipy signal.decimate function)
        maximum ripple in the passband (rp) and minimum attenuation in the stop band (rs) are modifiable
        '''

        if highpass:
            logger.info(
                'Decimating signals with factor {} and additional highpass filtering at {}!'.format(
                    decimate_factor, highpass))
        else:
            logger.info('Decimating signals with factor {}!'.format(decimate_factor))

        # input validation
        decimate_factor = abs(decimate_factor)
        order = abs(order)
        
        assert isinstance(decimate_factor, int)
        assert decimate_factor > 1
        assert isinstance(order, int)
        assert order > 1

        RpRs = [None, None]
        if filter_type == 'cheby1' or filter_type == 'cheby2' or filter_type == 'ellip':
            RpRs = [0.05, 0.05]  # standard for signal.decimate

        nyq = self.sampling_rate / 2

        meas_filtered = self.filter_signals(
            lowpass=nyq * 0.8 / decimate_factor,
            highpass=highpass,
            overwrite=False,
            order=order,
            ftype=filter_type,
            RpRs=RpRs,)

        self.sampling_rate /= decimate_factor
        
        N_dec = int(np.floor(self.total_time_steps / decimate_factor))
        # ceil would also work, but breaks indexing for aliasing noise estimation
        # with floor though, care must be taken to shorten the time domain signal to N_dec full blocks before slicing
        #decimate signal
        meas_decimated = np.copy(meas_filtered[0:N_dec * decimate_factor:decimate_factor, :])
        # correct for power loss due to decimation
        # https://en.wikipedia.org/wiki/Downsampling_(signal_processing)#Anti-aliasing_filter
        meas_decimated *= decimate_factor
        
        if self.F is not None:
            F_decimated = self.F_filt[slice(None, None, decimate_factor)]
            self.F = F_decimated
        self.total_time_steps = meas_decimated.shape[0]
        self.measurement = meas_decimated

    def psd_welch(self, n_lines=2048, refs_only=True, window='hamm'):
        '''
        * modify to compute one-sided PSD only, to save computation time
        * make possible to pass arguments to signal.csd
        * compute cross-psd of all  channels only with reference channels (i.e. replace 'numdof' with num_analised_channels or ref_channels, respectively)

        '''
        
        logger.info("Estimating Correlation Function and Power Spectral Density by Welch's method...")

        measurement = self.measurement
        sampling_rate = self.sampling_rate
        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
        else:
            num_ref_channels = num_analised_channels
        
        if 2 * n_lines > self.total_time_steps:
            raise ValueError(f'Number of frequency lines {n_lines} must not be larger than half the number of timesteps {self.total_timesteps}')
        
        psd_mats_shape = (num_analised_channels, num_ref_channels, n_lines + 1)
        psd_mats = np.zeros(psd_mats_shape, dtype=complex)
        #now = time.time()
        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                f, Pxy_den = scipy.signal.csd(measurement[:, channel_1], measurement[:, channel_2], sampling_rate,
                                              nperseg=2 * n_lines, window=window, scaling='spectrum', return_onesided=True)
                print(2*n_lines,2*n_lines//2,(self.total_time_steps-n_lines)/(2*n_lines//2))
                #print(2*n_lines, Pxy_den.shape, measurement.shape)
                Pxy_den *= n_lines
                psd_mats[channel_1, channel_2, :] = Pxy_den
                #then = time.time()
                #print(f'{channel_1}, {channel_2}, ({then-now} s)')
                #now = then

        freqs = np.fft.rfftfreq(2 * n_lines, 1 / sampling_rate)

        if self.scaling_factors is None:
            # obtain the scaling factors for the PSD which remain,
            # even after filtering or any DSP other operation
            self.scaling_factors = psd_mats.max(axis=2)
        
        self.psd_mats = psd_mats
        self.freqs = freqs
        self.n_lines = n_lines
        
        return psd_mats, freqs

    def corr_welch(self, tau_max, refs_only=True, window='hamming'):
        '''
        * compute cross-correlations of all channels only with reference channels (i.e. replace 'numdof' with num_analised_channels or ref_channels, respectively)
        '''
        
        psd_mats, freqs = self.psd_welch(n_lines=tau_max, refs_only=refs_only, window=window)
        
        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
        else:
            num_ref_channels = num_analised_channels
        corr_mats_shape = (num_analised_channels, num_ref_channels, tau_max)

        corr_matrix = np.zeros(corr_mats_shape)

        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                this_psd = psd_mats[channel_1, channel_2, :]
                this_corr = np.fft.irfft(this_psd)
                this_corr = this_corr[:tau_max].real

                corr_matrix[channel_1, channel_2, :] = this_corr

        self.corr_matrix = corr_matrix
        self.tau_max = tau_max
        return corr_matrix

    def psd_blackman_tukey(self, tau_max=256, window='bartlett'):
        '''
        * use rfft
        * why was 2*... removed from the amplitude correction?
        * compare with psd_welch
        * check energy in time domain and frequency domain with parsevals theorem
        * compute cross-psd of all  channels only with reference channels (i.e. replace 'numdof' with num_analised_channels or ref_channels, respectively)
        * compute only one-sided psd (i.e. length only tau_max - 1 or similar)
        * read about the window choices in the reference that is mentioned in the comment and try to implement other windows that ensure non-negative fourier transform
        '''

        logger.info("Estimating Correlation Function and Power Spectral Density by Blackman-Tukey's method...")

        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        ref_channels = self.ref_channels
        corr_matrix = self.compute_correlation_matrices(tau_max)
        
        psd_mats_shape = (num_analised_channels, num_ref_channels, tau_max)
        psd_mats = np.zeros(psd_mats_shape, dtype=complex)

        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                this_correlation_function = corr_matrix[channel_1,
                                                        channel_2, :]
                # create window, use Bartlett for nonnegative Fourier transform
                # otherwise Coherence becomes invalid
                # another option is to convolve another window with itself
                # from : SPECTRAL ANALYSIS OF SIGNALS, Petre Stoica and Randolph Moses, pp. 42 ff
                # window options: bartlett, blackman, hamming, hanning
                if window == 'bartlett':
                    win = np.bartlett(len(this_correlation_function))
                elif window == 'blackman':
                    win = np.blackman(len(this_correlation_function))
                    win = np.convolve(win, win, 'same')
                    win /= np.max(win)
                elif window == 'hamming':
                    win = np.hamming(len(this_correlation_function))
                    win = np.convolve(win, win, 'same')
                    win /= np.max(win)
                elif window == 'hanning':
                    win = np.hanning(len(this_correlation_function))
                    win = np.convolve(win, win, 'same')
                    win /= np.max(win)
                elif window == 'rect':
                    win = np.ones(len(this_correlation_function))
                else:
                    raise RuntimeError('Invalid window.')
                    return
                # test with coherence, should be between 0 and 1
                #coherence = np.abs(G12)**2/G11/G22

                # applies window and calculates fft
                fft = np.fft.rfft(
                    this_correlation_function * win,
                    n=2 * tau_max - 1)
                #print(this_correlation_function.shape, fft.shape)
                # corrections
                fft = fft[:tau_max]
                ampl_correction = (tau_max) / (win).sum()
                fft *= ampl_correction

                if channel_1 == channel_2:
                    fft = np.abs(fft)

                psd_mats[channel_1, channel_2, :] = fft

        freqs = np.fft.rfftfreq(2 * tau_max - 1, 1 / self.sampling_rate)
        
        if self.scaling_factors is None:
            # obtain the scaling factors for the PSD which remain,
            # even after filtering or any DSP other operation
            self.scaling_factors = psd_mats.max(axis=2)
            
        self.psd_mats = psd_mats
        self.freqs = freqs
        self.n_lines = tau_max

        return psd_mats, freqs

    def welch(self, n_lines):
        logger.info("Estimating Correlation Function and Power Spectral Density by Welch's method...")

        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels

        ref_channels = self.ref_channels
        dofs = list(range(self.measurement.shape[1]))

        signal = self.measurement

        #psd_mats_shape = (num_analised_channels, num_analised_channels, 2*n_lines)
        psd_mats_shape = (
            num_analised_channels,
            num_ref_channels,
            2 * n_lines // 2 + 1)

        psd_mats = np.zeros(psd_mats_shape, dtype=complex)

        for channel_1 in range(num_analised_channels):
            for channel_2, ref_channel in enumerate(ref_channels):
                #f, Pxy_den = scipy.signal.csd(signal[:,channel_1],signal[:,channel_2], self.sampling_rate, nperseg=n_lines*2, window='hamm', scaling='spectrum', return_onesided=False)
                f, Pxy_den = scipy.signal.csd(signal[:, channel_1], signal[:, ref_channel], self.sampling_rate,
                                              nperseg=n_lines * 2, window='hamm', scaling='spectrum', return_onesided=True)
                
                if channel_1 == channel_2:
                    assert (Pxy_den.imag == 0).all()

                #Pxy_den *= 2*n_lines-1
                Pxy_den *= n_lines
                psd_mats[channel_1, channel_2, :] = Pxy_den

        corr_mats_shape = (num_analised_channels, num_ref_channels, n_lines)

        corr_matrix = np.zeros(corr_mats_shape)

        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                this_psd = psd_mats[channel_1, channel_2, :]
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

        return corr_matrix, psd_mats  # , s_vals_cf, s_vals_psd

    def get_s_vals_psd(self, n_lines=256, window='hamm'):
        num_analised_channels = self.num_analised_channels
        psd_mats, freqs = self.psd_welch(
            n_lines=n_lines, refs_only=False, window=window)
        s_vals_psd = np.zeros((num_analised_channels, psd_mats.shape[2]))
        for t in range(psd_mats.shape[2]):
            # might use only real part to account for slightly asynchronous data
            # see [Au (2017): OMA, Chapter 7.5]
            s_vals_psd[:, t] = np.linalg.svd(psd_mats[:, :, t], True, False)
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
        logger.info('Computing Correlation Matrices with tau_max {}...'.format(tau_max))
        total_time_steps = self.total_time_steps
        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels
        measurement = self.measurement
        ref_channels = self.ref_channels
        #roving_channels = self.roving_channels

        self.tau_max = tau_max

        # ref_channels + roving_channels
        all_channels = list(range(num_analised_channels))
        # all_channels.sort()

        if not num_blocks:
            num_blocks = 1

        block_length = int(np.floor(total_time_steps / num_blocks))

        if block_length <= tau_max:
            raise RuntimeError(
                'Block length (={}) must be greater or equal to max time lag (={})'.format(
                    block_length, tau_max))

        corr_matrices_mem = []

        corr_mats_shape = (num_analised_channels, num_ref_channels, tau_max)

        for n_block in range(num_blocks):
            # shared memory, can be used by multiple processes
            # @UndefinedVariable
            corr_memory = mp.Array(
                c.c_double, np.zeros(
                    (np.product(corr_mats_shape))))
            corr_matrices_mem.append(corr_memory)

        measurement_shape = measurement.shape
        measurement_memory = mp.Array(
            c.c_double, measurement.reshape(
                measurement.size, 1))  # @UndefinedVariable

        # each process should have at least 10 blocks to compute, to reduce
        # overhead associated with spawning new processes
        n_proc = min(int(tau_max * num_blocks / 10), os.cpu_count())
        # pool=mp.Pool(processes=n_proc, initializer=self.init_child_process,
        # initargs=(measurement_memory, corr_matrices_mem)) #
        # @UndefinedVariable

        iterators = []
        it_len = int(np.ceil(tau_max * num_blocks / n_proc))
        printsteps = np.linspace(0, tau_max * num_blocks, 100, dtype=int)

        curr_it = []
        i = 0
        for n_block in range(num_blocks):
            for tau in range(1, tau_max + 1):
                i += 1
                if i in printsteps:
                    curr_it.append([n_block, tau, True])
                else:
                    curr_it.append((n_block, tau))
                if len(curr_it) > it_len:
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
            self.compute_covariance(
                curr_it,
                tau_max,
                block_length,
                ref_channels,
                all_channels,
                measurement_shape,
                corr_mats_shape)

        # pool.close()
        # pool.join()

        corr_matrices = []
        for corr_mats_mem in corr_matrices_mem:
            corr_mats = np.frombuffer(
                corr_mats_mem.get_obj()).reshape(corr_mats_shape)
            corr_matrices.append(corr_mats)

        self.corr_matrices = corr_matrices

        corr_mats_mean = np.mean(corr_matrices, axis=0)
        #corr_mats_mean = np.sum(corr_matrices, axis=0)
        #corr_mats_mean /= num_blocks - 1
        self.corr_matrix = corr_mats_mean

        #self.corr_mats_std = np.std(corr_matrices, axis=0)

        print('.', end='\n', flush=True)

        return corr_mats_mean

    def compute_covariance(
            self,
            curr_it,
            tau_max,
            block_length,
            ref_channels,
            all_channels,
            measurement_shape,
            corr_mats_shape):

        for this_it in curr_it:
            if len(this_it) > 2:
                print('.', end='', flush=True)
                del this_it[2]
            # standard unbiased estimator
            # R_fg[tau] = 1/(N-tau) /sum_{l=tau+1}^N f[l]g[l+m]
            n_block, tau = this_it

            measurement = np.frombuffer(
                measurement_memory.get_obj()).reshape(measurement_shape)

            this_measurement = measurement[(
                n_block) * block_length:(n_block + 1) * block_length, :]

            refs = this_measurement[:-tau, ref_channels]

            current_signals = this_measurement[tau:, all_channels]

            this_block = np.dot(current_signals.T, refs) / \
                current_signals.shape[0]
#             for i, ref_channel in enumerate(ref_channels):
#                 print(this_block[ref_channel,i])
#                 for chan_dof in self.chan_dofs:
#                     if chan_dof[0]==ref_channel:
#                         print(chan_dof)

            corr_memory = corr_matrices_mem[n_block]

            corr_mats = np.frombuffer(
                corr_memory.get_obj()).reshape(corr_mats_shape)

            with corr_memory.get_lock():
                corr_mats[:, :, tau - 1] = this_block

    def init_child_process(self, measurement_memory_, corr_matrices_mem_):
        # make the  memory arrays available to the child processes
        global measurement_memory
        measurement_memory = measurement_memory_
        
        global corr_matrices_mem
        corr_matrices_mem = corr_matrices_mem_


    def add_noise(self, amplitude=0, snr=0):
        logger.info(
            'Adding Noise with Amplitude {} and {} percent RMS'.format(
                amplitude,
                snr *
                100))
        assert amplitude != 0 or snr != 0

        if snr != 0 and amplitude == 0:
            rms = self.signal_rms()
            amplitude = rms * snr
        else:
            amplitude = [
                amplitude for channel in range(
                    self.num_analised_channels)]

        for channel in range(self.num_analised_channels):
            self.measurement[:,
                             channel] += np.random.normal(0,
                                                          amplitude[channel],
                                                          self.total_time_steps)

    def get_fft(self, svd=True, NFFT=2048):

        if self.ft_freq is None or self.sum_ft is None:
            ft, self.ft_freq = self.psd_welch(n_lines=NFFT, refs_only=False)
            if not svd:
                self.sum_ft = np.abs(np.sum(ft, axis=0))
            else:
                self.sum_ft = np.zeros(
                    (self.num_analised_channels, len(self.ft_freq)))
                for i in range(len(self.ft_freq)):
                    # might use only real parts of psd to account for slightly asynchronous data
                    # see [Au (2017): OMA, Chapter 7.5]
                    u, s, vt = np.linalg.svd(ft[:, :, i])
                    self.sum_ft[:, i] = 10 * np.log(s)
                    # print(10*np.log(s))

        #print(self.ft_freq.shape, self.sum_ft.shape)
        return self.ft_freq, self.sum_ft

    # def get_time_accel(self, channel):
        # time_vec = np.linspace(
            # 0,
            # self.total_time_steps /
            # self.sampling_rate,
            # self.total_time_steps)
        # accel_vel = self.measurement[:, channel]
        # return time_vec, accel_vel

    def plot_svd_spectrum(self, NFFT=512, log_scale=False, ax=None):

        if ax is None:
            ax = plt.subplot(111)

        psd_matrix, freq = self.psd_welch(NFFT, False)
        svd_matrix = np.zeros((self.num_analised_channels, len(freq)))
        # print(freq)
        for i in range(len(freq)):
            # might use only real part to account for slightly asynchronous data
            # see [Au (2017): OMA, Chapter 7.5]
            u, s, vt = np.linalg.svd(psd_matrix[:, :, i])
            if log_scale:
                s = 10 * np.log10(s)
            svd_matrix[:, i] = s  # 10*np.log(s)

        for i in range(self.num_analised_channels):
            ax.plot(freq, svd_matrix[i, :])
            # if i>3: break

        ax.set_xlim((0, self.sampling_rate / 2))
        # plt.grid(1)
        #ax.set_xlabel('Frequency [\si{\hertz}]')
        if log_scale:
            ax.set_ylabel('Singular Value Magnitude [\\si{\\decibel}]')
        else:
            ax.set_ylabel('Singul\\"arwert Magnitude')
        # plt.yticks([0,-25,-50,-75,-100,-125,-150,-175,-200,-225,-250])
        # plt.ylim((-225,0))
        # plt.xlim((0.1,5))

        # plt.grid(b=0)

        # plt.show()
    def plot_correlation(self, tau_max=None, num_blocks=False, ax=None):

        assert tau_max or self.corr_matrix.shape

        if tau_max is not None:
            if not self.corr_matrix:
                self.compute_correlation_matrices(tau_max, num_blocks)
            elif self.corr_matrix.shape[2] <= tau_max:
                self.compute_correlation_matrices(tau_max, num_blocks)
            corr_matrix = self.corr_matrix[:, :, :tau_max]
        else:
            corr_matrix = self.corr_matrix
            tau_max = corr_matrix.shape[2]

        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels

        if ax is None:
            ax = plt.subplot(111)

        for ref_channel in range(num_ref_channels):
            for channel in range(num_analised_channels):
                ax.plot(corr_matrix[channel, ref_channel, :])
        ax.set_xlim((0, tau_max))
        ax.set_xlabel('$\tau_{\text{max}}$')
        ax.set_ylabel(
            '$R_{i,j}(\tau) [\\si{\\milli\\metre\\squared\\per\\second\tothe{4}}]')

    def plot_psd(
            self,
            tau_max=None,
            n_lines=None,
            method='blackman',
            ax=None,
            **kwargs):

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
            elif self.psd_mats.shape[2] <= tau_max:
                if method == 'blackman':
                    self.psd_blackman_tukey(tau_max, **kwargs)
                else:
                    self.psd_welch(tau_max, **kwargs)

            psd_mats = self.psd_mats[:, :, tau_max]
        else:
            psd_mats = self.psd_mats

        freqs = self.freqs

        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels

        if ax is None:
            ax = plt.subplot(111)

        for ref_channel in range(num_ref_channels):
            for channel in range(num_analised_channels):
                ax.plot(freqs, np.abs(psd_mats[channel, ref_channel, :]))
        ax.set_xlim((0, freqs.max()))
        if plt.rc('latex.usetex'):
            ax.set_xlabel('$f [\\si{\\hertz}]$')
            ax.set_ylabel(
                '$S_{i,j}(f) [\\si{\\milli\\metre\\squared\\per\\second\tothe{4}}\\per\\hertz]$')
        else:
            ax.set_xlabel('f [Hz]')
            ax.set_ylabel('S_{i,j}(f) [mm^2/s^4/Hz]')


def load_measurement_file(fname, **kwargs):
    '''
    assign this function to the class before instantiating the object
    PreProcessSignals.load_measurement_file = load_measurement_file
    '''

    # define a function to return the following variables
    headers = ['channel_name', 'channel_name']
    units = ['unit', 'unit', ]
    start_time = datetime.datetime()
    sample_rate = float()
    measurement = np.array()

    # channels im columns
    assert measurement.shape[0] > measurement.shape[1]

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
#     prep_signals = PreProcessSignals.load_state('/vegas/scratch/womo1998/towerdata/towerdata_results_var/Wind_kontinuierlich__9_2016-10-05_04-00-00_000000/prep_signals.npz')
#     #prep_signals = None
#     preprocess_gui = PreProcessGUI(prep_signals)
#     loop = QEventLoop()
#     preprocess_gui.destroyed.connect(loop.quit)
#     loop.exec_()
#     print('Exiting GUI')
#
#     return


def example_filter():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_signals = PreProcessSignals(measurement, sampling_rate=128)

    prep_signals.filter_signals(
        order=4,
        ftype='cheby1',
        lowpass=20,
        highpass=None,
        RpRs=[
            0.1,
            0.1],
        overwrite=False,
        plot_filter=True)


def example_decimate():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_signals = PreProcessSignals(measurement, sampling_rate=128)

    _, f_plot = plt.subplots(2)
    f_plot[0].set_title('Decimate data')
    f_plot[0].plot(np.linspace(
        0, 1, len(prep_signals.measurement[:, 1])), prep_signals.measurement[:, 1])
    print('Original sampling rate: ', prep_signals.sampling_rate, 'Hz')
    print('Original number of time steps: ', prep_signals.total_time_steps)

    prep_signals.decimate_signals(5, order=8, filter_type='cheby1')

    print(prep_signals.measurement[:, 1])
    f_plot[1].plot(np.linspace(
        0, 1, len(prep_signals.measurement[:, 1])), prep_signals.measurement[:, 1])
    print('Decimated sampling rate: ', prep_signals.sampling_rate, 'Hz')
    print('Decimated number of time steps: ', prep_signals.total_time_steps)
    plt.show()


def example_welch():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_signals = PreProcessSignals(
        measurement,
        sampling_rate=128,
        ref_channels=[
            0,
            1])

    startA = time.time()
    corr_matrix, psd_mats = prep_signals.welch(256)
    print('Function A - Time elapsed: ', time.time() - startA)

    startB = time.time()
    corr_matrix_new = prep_signals.corr_welch(256)
    # were certainly generated during function call by corr_welch
    psd_mats_new, freqs = prep_signals.psd_mats, prep_signals.freqs
    print('Function B - Time elapsed: ', time.time() - startB)

    chA = 7
    chB = 1

    _, f_plot = plt.subplots(2)
    f_plot[0].set_title('Correlation Values - different functions')
    f_plot[0].plot(corr_matrix[chA, chB, :])
    f_plot[1].plot(corr_matrix_new[chA, chB, :])
    plt.figure()
    plt.title('Correlation Values - Superimposed graph')
    plt.plot(corr_matrix[chA, chB, :])
    plt.plot(corr_matrix_new[chA, chB, :])
    # plt.show()

    _, f_plot = plt.subplots(2)
    f_plot[0].set_title('PSD Values - different functions')
    f_plot[0].plot(np.abs(psd_mats[chA, chB, :]))
    f_plot[1].plot(np.abs(psd_mats_new[chA, chB, :]))
    plt.figure()
    plt.title('PSD Values - Superimposed graph')
    plt.plot(np.abs(psd_mats[chA, chB, :]))
    plt.plot(np.abs(psd_mats_new[chA, chB, :]))

    plt.show()


def example_blackman_tukey():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_signals = PreProcessSignals(
        measurement,
        sampling_rate=128,
        ref_channels=[
            0,
            1])

    startA = time.time()
    psd_mats, freqs = prep_signals.psd_blackman_tukey(
        tau_max=256, window='bartlett')
    corr_matrix = prep_signals.corr_matrix
    print('Time elapsed: ', time.time() - startA)
    # print(freqs)
    chA = 7
    chB = 1

    plt.figure()
    plt.title('Correlation Values')
    plt.plot(corr_matrix[chA, chB, :])
    plt.figure()
    plt.title('PSD Values')
    plt.plot(freqs, np.abs(psd_mats[chA, chB, :]))
    plt.show()


def compare_PSD_Corr():
    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_signals = PreProcessSignals(
        measurement,
        sampling_rate=128,
        ref_channels=[
            0,
            1])

    prep_signals.filter_signals(lowpass=10, highpass=0.1, overwrite=True)

    startA = time.time()
    psd_mats_b, freqs_b = prep_signals.psd_blackman_tukey(
        tau_max=2048, window='hamming')
    corr_matrix_b = prep_signals.corr_matrix
    print('Blackman-Tukey - Time elapsed: ', time.time() - startA)

    startB = time.time()
    corr_matrix_w = prep_signals.corr_welch(2048, window='hamming')
    psd_mats_w, freqs_w = prep_signals.psd_mats, prep_signals.freqs
    print('Welch - Time elapsed: ', time.time() - startB)

    chA = 7
    chB = 1

    plt.figure()
    plt.title('Correlation Values')
    plt.plot(corr_matrix_b[chA, chB, :])
    plt.plot(corr_matrix_w[chA, chB, :])
    plt.figure()
    plt.title('PSD Values')
    plt.plot(freqs_b, np.abs(psd_mats_b[chA, chB, :]))
    plt.plot(freqs_w, np.abs(psd_mats_w[chA, chB, :]))
    plt.show()


if __name__ == '__main__':
    import os
    path = 'E:/OneDrive/BHU_NHRE/Python/2017_PreProcessGUI/'
    path = '/ismhome/staff/womo1998/Projects/2017_PreProcessGUI/'
    os.chdir(path)

    path = 'Messung_Test.asc'
    measurement = np.loadtxt(path)
    prep_signals = PreProcessSignals(
        measurement,
        sampling_rate=128,
        ref_channels=[
            0,
            1])
    prep_signals.plot_svd_spectrum(8192)
    plt.show()
    # example_filter()
    # example_decimate()
    # example_welch()
    # example_blackman_tukey()
    compare_PSD_Corr()
    main()

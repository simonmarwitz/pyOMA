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
import os
import csv
import datetime

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from core.Helpers import nearly_equal, simplePbar

import logging
#logging.basicConfig(stream=sys.stdout)  # done at module level
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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
                    _ = master_slave
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

    def __init__(self, signals, sampling_rate,
                 ref_channels=None,
                 accel_channels=None, velo_channels=None, disp_channels=None,
                 setup_name=None, channel_headers=None, start_time=None,
                 F=None, **kwargs):

        super().__init__()

        assert isinstance(signals, np.ndarray)
        assert signals.shape[0] > signals.shape[1]
        self.signals = signals
        self.signals_filtered = signals
        
        assert isinstance(sampling_rate, (int, float))
        self.sampling_rate = sampling_rate

        # added by anil
        if F is not None:
            assert isinstance(F, np.ndarray)
        self.F = F

        self._ref_channels = None
        if ref_channels is None:
            ref_channels = list(range(signals.shape[1]))
        self.ref_channels = ref_channels
        
        self._accel_channels = []
        self._velo_channels = []
        self._disp_channels = []
        
        if disp_channels is None:
            disp_channels = []
        if velo_channels is None:
            velo_channels = []
        if accel_channels is None:
            accel_channels = [c for c in range(self.num_analised_channels)
                              if c not in disp_channels and c not in velo_channels]

        for chan in range(self.num_analised_channels):
            if (chan in accel_channels) + (chan in velo_channels) + \
                    (chan in disp_channels) != 1:
                logger.warning(f'Quantity of channel {chan} is not defined.')

        self.accel_channels = accel_channels
        self.velo_channels = velo_channels
        self.disp_channels = disp_channels

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

        self.chan_dofs = []
        
        self.channel_factors = [1 for _ in range(self.num_analised_channels)]
        self.scaling_factors = None
        
        self._last_meth = None
        
        self.corr_matrix_wl = None
        self.psd_matrix_wl = None
        self.n_lines_wl = None
        self.n_segments_wl = None
        
        self.corr_matrix_bt = None
        self.psd_matrix_bt = None
        self.n_lines_bt = None
        self.n_segments_bt = None
        
        self.var_corr_bt = None
        self.var_psd_wl = None
        
        #self.s_vals_cf = None
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
            headers, _, start_time, sample_rate, signals = loaded_signals
        else:
            signals = loaded_signals
            start_time = datetime.datetime.now()
            sample_rate = sampling_rate
            headers = ['Channel_{}'.format(i)
                       for i in range(signals.shape[1])]
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
                            'Different headers for channel {} in signals file ({}) and in channel-DOF-assignment ({}).'.format(
                                chan, headers[chan], chan_name))
                    else:
                        continue

        # print(delete_channels)
        if delete_channels:
            # delete_channels.sort(reverse=True)

            _ = [
                'Reference Channels',
                'Accel. Channels',
                'Velo. Channels',
                'Disp. Channels']
            _ = [
                ref_channels,
                accel_channels,
                velo_channels,
                disp_channels]
            # print(chan_dofs)

            num_all_channels = signals.shape[1]
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

            signals = np.delete(signals, delete_channels, axis=1)

            chan_dofs = new_chan_dofs
            ref_channels = new_ref_channels
            accel_channels = new_accel_channels
            velo_channels = new_velo_channels
            disp_channels = new_disp_channels
            headers = new_headers
            #print(chan_dofs, ref_channels, accel_channels, velo_channels,disp_channels, headers)


#             channel = signals.shape[1]
#             #num_channels = signals.shape[1]
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
#             signals=np.delete(signals, delete_channels, axis=1)
        #total_time_steps = signals.shape[0]
        num_channels = signals.shape[1]
        #roving_channels = [i for i in range(num_channels) if i not in ref_channels]
        if not accel_channels and not velo_channels and not disp_channels:
            accel_channels = [i for i in range(num_channels)]
        #print(signals.shape, ref_channels)
        # print(signals)
        prep_signals = cls(signals, sampling_rate,
                           ref_channels,
                           accel_channels, velo_channels, disp_channels,
                           setup_name=name, channel_headers=headers,
                           start_time=start_time,
                           **kwargs)
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
        A method for loading a signals file

        Parameters
        ----------
        fname : str
                The full path of the signals file

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
        signals : ndarray
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
        
        out_dict['self.signals'] = self.signals
        out_dict['self.sampling_rate'] = self.sampling_rate
        out_dict['self.ref_channels'] = self._ref_channels
        out_dict['self.accel_channels'] = self._accel_channels
        out_dict['self.velo_channels'] = self._velo_channels
        out_dict['self.disp_channels'] = self._disp_channels
        
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.channel_headers'] = self.channel_headers
        out_dict['self.start_time'] = self.start_time
        
        out_dict['self.chan_dofs'] = self.chan_dofs
        out_dict['self.scaling_factors'] = self.scaling_factors
        out_dict['self.channel_factors'] = self.channel_factors
        out_dict['self._last_meth'] = self._last_meth
        
        out_dict['self.corr_matrix_wl'] = self.corr_matrix_wl
        out_dict['self.psd_matrix_wl'] = self.psd_matrix_wl
        out_dict['self.n_lines_wl'] = self.n_lines_wl
        out_dict['self.n_segments_wl'] = self.n_segments_wl
        
        out_dict['self.corr_matrix_bt'] = self.corr_matrix_bt
        out_dict['self.psd_matrix_bt'] = self.psd_matrix_bt
        out_dict['self.n_lines_bt'] = self.n_lines_bt
        out_dict['self.n_segments_bt'] = self.n_segments_bt
        
        out_dict['self.var_corr_bt'] = self.var_corr_bt
        out_dict['self.var_psd_wl'] = self.var_psd_wl

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname):
        
        def validate_array(arr):
            '''
            Determine whether the argument has a numeric datatype and if
            not convert the argument to a scalar object or a list.
        
            Booleans, unsigned integers, signed integers, floats and complex
            numbers are the kinds of numeric datatype.
        
            Parameters
            ----------
            array : array-like
                The array to check.
            
            '''
            _NUMERIC_KINDS = set('buifc')
            if not arr.shape:
                return arr.item()
            elif arr.dtype.kind in _NUMERIC_KINDS:
                return arr
            else:
                return list(arr)
            
        logger.info('Now loading previous results from  {}'.format(fname))
        
        in_dict = np.load(fname, allow_pickle=True)
        
        try:
            signals = in_dict['self.signals']
            sampling_rate = validate_array(in_dict['self.sampling_rate'])
            _ref_channels = validate_array(in_dict['self.ref_channels'])
            _accel_channels = validate_array(in_dict['self.accel_channels'])
            _velo_channels = validate_array(in_dict['self.velo_channels'])
            _disp_channels = validate_array(in_dict['self.disp_channels'])
            channel_headers = validate_array(in_dict['self.channel_headers'])
            start_time = validate_array(in_dict['self.start_time'])
            setup_name = validate_array(in_dict['self.setup_name'])

            preprocessor = cls(signals, sampling_rate,
                               _ref_channels,
                               _accel_channels, _velo_channels, _disp_channels,
                               setup_name, channel_headers, start_time,
                               )
    
            chan_dofs = [[int(float(chan_dof[0])), str(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3]), str(
                chan_dof[4] if 5 == len(chan_dof) else '')] for chan_dof in in_dict['self.chan_dofs']]
            preprocessor.add_chan_dofs(chan_dofs)
            
            preprocessor.scaling_factors = validate_array(in_dict['self.scaling_factors'])
            preprocessor.channel_factors = validate_array(in_dict['self.channel_factors'])
            preprocessor._last_meth = validate_array(in_dict['self._last_meth'])
            
            preprocessor.corr_matrix_wl = validate_array(in_dict['self.corr_matrix_wl'])
            preprocessor.psd_matrix_wl = validate_array(in_dict['self.psd_matrix_wl'])
            preprocessor.n_lines_wl = validate_array(in_dict['self.n_lines_wl'])
            preprocessor.n_segments_wl = validate_array(in_dict['self.n_segments_wl'])
            
            preprocessor.corr_matrix_bt = validate_array(in_dict['self.corr_matrix_bt'])
            preprocessor.psd_matrix_bt = validate_array(in_dict['self.psd_matrix_bt'])
            preprocessor.n_lines_bt = validate_array(in_dict['self.n_lines_bt'])
            preprocessor.n_segments_bt = validate_array(in_dict['self.n_segments_bt'])
            
            preprocessor.var_corr_bt = validate_array(in_dict['self.var_corr_bt'])
            preprocessor.var_psd_wl = validate_array(in_dict['self.var_psd_wl'])
            
        except KeyError:
            preprocessor = cls.load_state_depr(in_dict)
        
        return preprocessor
    
    @classmethod
    def load_state_depr(cls, in_dict):

        logger.warning('You are loading from a deprecated format. Consider re-saving your data once now to ensure future compatibility.')

        setup_name = str(in_dict['self.setup_name'].item())
        measurement = in_dict['self.measurement']
        sampling_rate = float(in_dict['self.sampling_rate'])
        _ = int(in_dict['self.total_time_steps'])
        ref_channels = list(in_dict['self.ref_channels'])
        if in_dict['self.channel_headers'].shape:
            channel_headers = list(in_dict['self.channel_headers'])
        else:
            channel_headers = ['' for _ in range(
                int(in_dict['self.num_analised_channels']))]
        start_time = in_dict['self.start_time'].item()

        accel_channels = list(in_dict['self.accel_channels'])
        velo_channels = list(in_dict['self.velo_channels'])
        disp_channels = list(in_dict['self.disp_channels'])
        
        spectral_values = [None, None, None, None, None]
        for obj_num, name in enumerate(
                ['self.corr_matrix', 'self.psd_mats', 'self.s_vals_psd', 'self.tau_max']):
            try:
                spectral_values[obj_num] = in_dict[name]
            except Exception as e:
                logger.warning(repr(e))

        preprocessor = cls(measurement, sampling_rate,
                           ref_channels,
                           accel_channels, velo_channels, disp_channels,
                           setup_name, channel_headers, start_time,
                           )

        chan_dofs = [[int(float(chan_dof[0])), str(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3]), str(
            chan_dof[4] if 5 == len(chan_dof) else '')] for chan_dof in in_dict['self.chan_dofs']]
        preprocessor.add_chan_dofs(chan_dofs)

        preprocessor._last_meth = 'welch'
        preprocessor.corr_matrix_wl = spectral_values[0]
        preprocessor.pds_matrix_wl = spectral_values[1]
        preprocessor.s_vals_psd = spectral_values[2]
        if spectral_values[3] is None:
            preprocessor.n_lines_wl = None
        else:
            preprocessor.n_lines_wl = (int(spectral_values[3]) - 1) * 2

        return preprocessor
        
    def validate_channels(self, channels, quant_check=False):
        if quant_check:
            accel_channels = self.accel_channels
            velo_channels = self.velo_channels
            disp_channels = self.disp_channels
            
        for channel in channels:
            # channel names
            if channel < 0:
                raise ValueError('A channel number cannot be negative!')
            if channel > self.num_analised_channels - 1:
                raise ValueError('A channel number cannot be greater'
                                 ' than the number of all channels!')
            if quant_check:
                if channel in accel_channels:
                    logger.warning(f'Channel {self.channel_headers[channel]} is already defined'
                                   ' as an acceleration channel. Removing')
                    accel_channels.remove(channel)
                if channel in velo_channels:
                    logger.warning(f'Channel {self.channel_headers[channel]} is already defined'
                                   ' as a velocity channel. Removing')
                    velo_channels.remove(channel)
                if channel in disp_channels:
                    logger.warning(f'Channel {self.channel_headers[channel]} is already defined'
                                   ' as a displacement channel. Removing')
                    disp_channels.remove(channel)
    
    def _channel_numbers(self, channels=None, refs=None):
        """
        Method to return channel numbers
        
        Interpretation of the argument values:
        * None: a list of all channel indices
        * list-of-int: a validated list of given channel indices
        * list-of-str: a list of channel indices for each channel name in the given order
        * int: a single-item list of the given channel index
        * str: a single-item list of the channel index for the given name
        * 'auto' (only refs): a single item list corresponding to the respective channel
        
        Parameters
        ----------
            channels: None, list-of-int, list-of-str, int, str
                The selected channels.
            refs: 'auto', list-of-indices, optional
                The reference channel indices to be contained in the reference channel list
                
        Returns
        -------
            channel_numbers: list
                The generated channel indices
            ref_numbers: list-of-lists
                The corresponding reference channels for each channel in
                channel_numbers, such that it can be looped over in an inner loop.
        """
        if channels is None:
            channel_numbers = list(range(self.num_analised_channels))
        elif isinstance(channels, int):
            channel_numbers = [channels]
        elif isinstance(channels, str):
            try:
                channel_number = int(channels)
            except ValueError:
                channel_number = self.channel_headers.index(channels)
            channel_numbers = [channel_number]
        elif isinstance(channels, (list, tuple, np.ndarray)):
            channel_numbers = []
            for channel in channels:
                if isinstance(channel, (int, np.int, np.int64)):
                    channel_numbers.append(int(channel))
                elif isinstance(channel, str):
                    try:
                        channel_number = int(channel)
                    except ValueError:
                        channel_number = self.channel_headers.index(channel)
                    channel_numbers.append(channel_number)
                else:
                    raise ValueError(f'Channel {channel} in channels is an invalid channel definition.')
        
        if refs is None:
            ref_channels = self.ref_channels
            ref_numbers = [ref_channels for _ in channel_numbers]
        elif refs == 'auto':
            ref_numbers = [[ind] for ind in channel_numbers]
        elif isinstance(refs, int):
            ref_numbers = [[refs] for _ in channel_numbers]
        elif isinstance(refs, str):
            ind = self.channel_headers.index(channels)
            ref_numbers = [[ind] for _ in channel_numbers]
        elif isinstance(refs, (list, tuple, np.ndarray)):
            custom_ref_numbers = []
            for channel in refs:
                if isinstance(channel, int):
                    custom_ref_numbers.append(channel)
                elif isinstance(channel, str):
                    custom_ref_numbers.append(self.channel_headers.index(channel))
                else:
                    raise ValueError(f'Channel {channel} in refs is an invalid channel definition.')
            ref_numbers = [custom_ref_numbers for _ in channel_numbers]
        else:
            raise ValueError(f'{refs} not a valid reference channel specification.')
            
        return channel_numbers, ref_numbers
    
    @property
    def ref_channels(self):
        return self._ref_channels
    
    @ref_channels.setter
    def ref_channels(self, ref_channels):
        ref_channels, _ = self._channel_numbers(ref_channels)
        self.validate_channels(ref_channels)
        self._clear_spectral_values()
        self._ref_channels = ref_channels
    
    @property
    def accel_channels(self):
        return self._accel_channels
    
    @accel_channels.setter
    def accel_channels(self, accel_channels):
        accel_channels, _ = self._channel_numbers(accel_channels)
        self.validate_channels(accel_channels, True)
        self._accel_channels = accel_channels
        
    @property
    def velo_channels(self):
        return self._velo_channels
    
    @velo_channels.setter
    def velo_channels(self, velo_channels):
        velo_channels, _ = self._channel_numbers(velo_channels)
        self.validate_channels(velo_channels, True)
        self._velo_channels = velo_channels
        
    @property
    def disp_channels(self):
        return self._disp_channels
    
    @disp_channels.setter
    def disp_channels(self, disp_channels):
        disp_channels, _ = self._channel_numbers(disp_channels)
        self.validate_channels(disp_channels, True)
        self._disp_channels = disp_channels
    
    @property
    def num_ref_channels(self):
        return len(self.ref_channels)
    
    @property
    def num_analised_channels(self):
        return self.signals.shape[1]
    
    @property
    def total_time_steps(self):
        return self.signals.shape[0]
    
    @property
    def duration(self):
        return self.total_time_steps / self.sampling_rate
    
    @property
    def dt(self):
        return 1 / self.sampling_rate
    
    @property
    def t(self):
        N = self.total_time_steps
        fs = self.sampling_rate
        # t[-1] != self.duration to ensure sample_spacing == self.dt
        return np.linspace(0, N / fs, N, False)
    
    @property
    def n_lines(self):
        if self._last_meth == 'welch':
            return self.n_lines_wl
        elif self._last_meth == 'blackman-tukey':
            return self.n_lines_bt
        else:
            return None
    
    @property
    def freqs(self):
        '''
        Returns
        ----------
            freqs: np.ndarray (n_lines, )
                Array with the frequency lines corresponding to the spectral values
        '''
        if self.n_lines:
            n_lines = self.n_lines
            fs = self.sampling_rate
            return np.fft.rfftfreq(n_lines, 1 / fs)
    
    @property
    def freqs_wl(self):
        '''
        Returns
        ----------
            freqs: np.ndarray (n_lines, )
                Array with the frequency lines corresponding to the spectral values
        '''
        if self.n_lags_wl:
            n_lines = self.n_lines_wl
            fs = self.sampling_rate
            return np.fft.rfftfreq(n_lines, 1 / fs)
    
    @property
    def freqs_bt(self):
        '''
        Returns
        ----------
            freqs: np.ndarray (n_lines, )
                Array with the frequency lines corresponding to the spectral values
        '''
        if self.n_lines_bt:
            n_lines = self.n_lines_bt
            fs = self.sampling_rate
            return np.fft.rfftfreq(n_lines, 1 / fs)
        
    @property
    def lags(self):
        if self.n_lags:
            n_lags = self.n_lags
            fs = self.sampling_rate
            return np.linspace(0, n_lags / fs, n_lags, False)
    
    @property
    def lags_wl(self):
        if self.n_lags_wl:
            n_lags = self.n_lags_wl
            fs = self.sampling_rate
            return np.linspace(0, n_lags / fs, n_lags, False)
    
    @property
    def lags_bt(self):
        if self.n_lags_bt:
            n_lags = self.n_lags_bt
            fs = self.sampling_rate
            return np.linspace(0, n_lags / fs, n_lags, False)
    
    @property
    def n_lags(self):
        if self.n_lines:
            return self.n_lines // 2 + 1
    
    @property
    def n_lags_wl(self):
        if self.n_lines_wl:
            return self.n_lines_wl // 2 + 1
        
    @property
    def n_lags_bt(self):
        if self.n_lines_bt:
            return self.n_lines_bt // 2 + 1
    
    @property
    def corr_matrix(self):
        if self._last_meth == 'welch':
            return self.corr_matrix_wl
        elif self._last_meth == 'blackman-tukey':
            return self.corr_matrix_bt
        else:
            return None
    
    @property
    def psd_matrix(self):
        if self._last_meth == 'welch':
            return self.psd_matrix_wl
        elif self._last_meth == 'blackman-tukey':
            return self.psd_matrix_bt
        else:
            return None
        
    @property
    def signal_power(self):
        if not np.all(np.isclose(np.mean(self.signals, axis=0), 0)):
            logger.warning("Signal has constant offsets. Power values may be errorneous")
        return np.mean(np.square(self.signals), axis=0)
    
    @property
    def signal_rms(self):
        return np.sqrt(self.signal_power)
    
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
            self.signals[:, channel] += np.random.normal(0, amplitude[channel], self.total_time_steps)
        self._clear_spectral_values()
        
    def correct_offset(self):
        '''
        corrects a constant offset from measured signals
        
        ..TODO::
            * remove linear, ... ofsets as well
        '''
        logger.info('Correcting offset of measured signals')
        self.signals -= self.signals.mean(axis=0)
        self._clear_spectral_values()
        
        return

    def precondition_signals(self, method='iqr'):

        assert method in ['iqr', 'range']

        self.correct_offset()

        for i in range(self.signals.shape[1]):
            tmp = self.signals[:, i]
            if method == 'iqr':
                factor = np.subtract(*np.percentile(tmp, [95, 5]))
            elif method == 'range':
                factor = np.max(tmp) - np.min(tmp)
            self.signals[:, i] /= factor
            self.channel_factors[i] = factor
        
        self._clear_spectral_values()
        
    def filter_signals(self, lowpass=None, highpass=None,
                       overwrite=True,
                       order=None, ftype='butter', RpRs=[3, 3],
                       plot_ax=None):
        logger.info('Filtering signals in the band: {} .. {} with a {} order {} filter.'.format(highpass, lowpass, order, ftype))

        if (highpass is None) and (lowpass is None):
            raise ValueError('Neither a lowpass or a highpass corner frequency was provided.')
        
        ftype_list = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel', 'moving_average', 'brickwall']
        if not (ftype in ftype_list):
            raise ValueError(f'Filter type {ftype} is not any of the available types: {ftype_list}')
        
        if order is None:
            if ftype_list.index(ftype) < 5:
                # default FIR filter order
                order = 4
            else:
                # default IIR filter order
                order = 21
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
        measurement = self.signals
        
        if ftype in ftype_list[0:5]:  # IIR filter
            #if order % 2:  # odd number
            #    logger.warning(f'Odd filter order {order} will be rounded up to {order+1}, because of forward-backward filtering.')
            #order = int(np.ceil(order / 2))  # reduce by factor 2 because of double filtering
            order = int(order)
            
            sos = scipy.signal.iirfilter(
                order, freqs, rp=RpRs[0], rs=RpRs[1],
                btype=btype, ftype=ftype, output='sos')
            
            signals_filtered = scipy.signal.sosfiltfilt(
                sos, measurement, axis=0)
            if self.F is not None:
                self.F_filt = scipy.signal.sosfiltfilt(sos, self.F, axis=0)
        elif ftype in ftype_list[5:7]:  # FIR filter
            if ftype == 'brickwall':
                fir_irf = scipy.signal.firwin(numtaps=order, cutoff=freqs, pass_zero=btype, fs=np.pi)
            elif ftype == 'moving_average':
                if freqs:
                    logger.warning('For the moving average filter, no cutoff frequencies can be defined.')
                fir_irf = np.ones((order)) / order
            
            signals_filtered = scipy.signal.lfilter(fir_irf, [1.0], measurement, axis=0)
            if self.F is not None:
                self.F_filt = scipy.signal.lfilter(fir_irf, [1.0], self.F, axis=0)
            
        if np.isnan(signals_filtered).any():
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
            self.signals = signals_filtered
            if self.F is not None:
                self.F = self.F_filt
        self.signals_filtered = signals_filtered
        self._clear_spectral_values()
        
        return signals_filtered
    
    def decimate_signals(self, decimate_factor, nyq_rat=2.5,
                         highpass=None, order=None, filter_type='cheby1'):
        '''
        decimates signals data
        filter type and order are choosable (order 8 and type cheby1 are standard for scipy signal.decimate function)
        maximum ripple in the passband (rp) and minimum attenuation in the stop band (rs) are modifiable
        '''

        if highpass:
            logger.info(f'Decimating signals by factor {decimate_factor}'
                        f' and additional highpass filtering at {highpass}'
                        f' to a sampling rate of {self.sampling_rate/decimate_factor} Hz')
        else:
            logger.info(f'Decimating signals by factor {decimate_factor}'
                        f' to a sampling rate of {self.sampling_rate/decimate_factor} Hz')

        # input validation
        decimate_factor = abs(decimate_factor)
        
        assert isinstance(decimate_factor, int)
        assert decimate_factor >= 1

        if order is None:
            if filter_type in ['brickwall', 'moving_average']:
                order = 21 * decimate_factor - 1  # make it odd to avoid errors when highpass filtering
            else:
                order = 8
        else:
            order = abs(order)
            
        assert isinstance(order, int)
        assert order > 1
        
        RpRs = [None, None]
        if filter_type == 'cheby1' or filter_type == 'cheby2' or filter_type == 'ellip':
            RpRs = [0.05, 0.05]  # standard for signal.decimate
        
        nyq = self.sampling_rate / decimate_factor

        sig_filtered = self.filter_signals(
            lowpass=nyq / nyq_rat,
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
        sig_decimated = np.copy(sig_filtered[0:N_dec * decimate_factor:decimate_factor, :])
        # correct for power loss due to decimation
        # https://en.wikipedia.org/wiki/Downsampling_(signal_processing)#Anti-aliasing_filter
        sig_decimated *= decimate_factor
        
        if self.F is not None:
            F_decimated = self.F_filt[slice(None, None, decimate_factor)]
            self.F = F_decimated
        #self.total_time_steps = sig_decimated.shape[0]
        self.signals = sig_decimated
        self._clear_spectral_values()
    
    def _clear_spectral_values(self):
        """
        Convenience method to clear all previously computed spectral values.
        To be called when, any modifications, such as filtering, decimation,
        etc. are applied to the signals.
        """
        
        self.scaling_factors = None
        self._last_meth = None
        
        self.psd_matrix_bt = None
        self.psd_matrix_wl = None
        
        self.n_lines_wl = None
        self.n_lines_bt = None
        
        self.n_segments_bt = None
        self.n_segments_wl = None
        
        self.corr_matrix_bt = None
        self.corr_matrix_wl = None
        
        self.var_corr_bt = None
        self.var_psd_wl = None
        
    def psd_welch(self, n_lines, n_segments=None, refs_only=True, window='hamming', **kwargs):
        '''
        Estimate the (cross- and auto-) power spectral densities (PSD),
        according to Welch's method. No overlapping is allowed (deliberately
        to ensure statistical independence of blocks for variance estimation).
        Segments are n_lines // 2 long and zero padded to n_lines to allow
        estimation of the full correlation sequence, which is twice as long
        as the input signal. Normalization is applied w.r.t. conservation of
        energy, i.e. magnitudes will change with n_lines but power stays
        constant.
        
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
            n_segments: integer, optional
                Number of segments to perform averaging over
                if given, n_lines is overridden
            refs_only: bool, optional
                Compute cross-PDSs only with reference channels
            window: str or tuple or array_like, optional
                Desired window to use. See scipy.signal.get_window() for more information
            
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to scipy.signals.csd

        Returns
        -------
            psd_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lines // 2 + 1)
                containing the power density values of the respective
                channels and frequencies
                
        .. TODO ::
            * implement variance estimation (self-implement Welch's method)
        '''
        self._last_meth = 'welch'
        
        while True:
            # check, if it is possible to simply return previously computed PSD
            if kwargs:
                break
            if self.psd_matrix_wl is None:
                break
            if self.n_lines_wl != n_lines:
                break
            if n_segments is not None and self.n_segments_wl != n_segments:
                break
            if (self.psd_matrix_wl.shape[1] == self.num_ref_channels) != refs_only:
                break
            
            logger.debug("Using previously computed Power Spectral Density (Welch)...")
            return self.psd_matrix_wl
        
        logger.debug(f'Arguments psd_welch: n_lines={n_lines}, n_segments={n_segments}, refs_only={refs_only}, window={window}, {kwargs}')
        
        if n_lines % 2:
            n_lines += 1
            logger.warning(f"Only even number of frequency lines are supported setting n_lines={n_lines}")
        
        N = self.total_time_steps
        
        if n_segments is not None:
            if n_lines is not None:
                logger.warning('n_segments and n_lines are given. overriding n_lines.')
            n_lines = 2 * N // n_segments
        
        if n_lines > 2 * self.total_time_steps:
            logger.warning(f'Number of frequency lines {n_lines} should not'
                           f'be larger than twice the number of timesteps {self.total_time_steps}')
        
        # it increase variance and does not improve the result in any other sense
        # when using less than the maximally possible number of segments
        n_segments = max(N // (n_lines // 2), 1)
        fs = self.sampling_rate
        
        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
            ref_channels = self.ref_channels
        else:
            num_ref_channels = num_analised_channels
            ref_channels = list(range(num_ref_channels))

        signals = self.signals

        psd_matrix_shape = (num_analised_channels,
                            num_ref_channels,
                            n_lines // 2 + 1)

        psd_matrix = np.empty(psd_matrix_shape, dtype=complex)
        
        logger.info(f"Estimating PSD by Welch's method with {n_lines}"
                    f' frequency lines, {n_segments} non-overlapping'
                    f' segments and a {window} window...')
        
        win = scipy.signal.get_window(window, n_lines // 2, fftbins=True)
        
        pbar = simplePbar(num_analised_channels * num_ref_channels)
        for channel_1 in range(num_analised_channels):
            for channel_2, ref_channel in enumerate(ref_channels):
                next(pbar)
                # compute spectrum according to welch, with automatic application of a window and scaling
                # specrum scaling compensates windowing by dividing by window(n_lines).sum()**2
                # density scaling divides by fs * window(n_lines)**2.sum()
                _, Pxy_den = scipy.signal.csd(signals[:, channel_1],
                                              signals[:, ref_channel],
                                              fs,
                                              window=win,
                                              nperseg=n_lines // 2,
                                              nfft=n_lines,
                                              noverlap=0,
                                              return_onesided=True,
                                              scaling='density',
                                              **kwargs)
                
                if channel_1 == ref_channel:
                    assert np.isclose(Pxy_den.imag, 0).all()
                    Pxy_den.imag = 0
                # compensate averaging over segments (for power equivalence segments should be summed up)
                Pxy_den *= n_segments
                # reverse 1/Hz of scaling="density"
                Pxy_den *= fs
                # compensate onesided
                Pxy_den /= 2
                # compensate zero-padding
                Pxy_den /= 2
                # compensate energy loss through short segments
                Pxy_den *= n_lines
                
                psd_matrix[channel_1, channel_2, :] = Pxy_den
                
        if self.scaling_factors is None:
            # obtain the scaling factors for the PSD which remain,
            # even after filtering or any DSP other operation
            self.scaling_factors = psd_matrix.max(axis=2)
        logger.debug(f'PSD Auto-/Cross-Powers: {np.mean(np.abs(psd_matrix), axis=2)}')
        
        self.psd_matrix_wl = psd_matrix
        self.n_lines_wl = n_lines
        self.n_segments_wl = n_segments
        
        return psd_matrix

    def corr_welch(self, n_lags=None, refs_only=True, **kwargs):
        '''
        Estimate the (cross- and auto-) correlation functions (C/ACF),
        by the inverse Fourier Transform of Power Spectral Densities,
        estimated according to Welch's method. Bias due to windowing of
        the underlying PSD persists.  Normalization is done according to
        the unbiased estimator, i.e. 0-lag correlation value must be
        multiplied by n_lines to get the signals cross-power.
        
        Note that:
            n_lags \= n_lines // 2 + 1
        
            n_lines \= (n_lags - 1) * 2

        Parameters
        ----------
            n_lags: integer, optional
                Total number of lags (positive). Note: this includes the
                0-lag, therefore excludes the n_lags-lag.
            refs_only: bool, optional
                Compute cross-ACFss only with reference channels
            
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to self.psd_welch and further

        Returns
        -------
            corr_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lags)
                containing the correlation values of the respective
                channels and lags
        
        See also
        --------
            psd_welch:
                PSD estimation algorithm used by this method.
                
                        
        .. TODO ::
            * deconvolve window (?)
        '''
        self._last_meth = 'welch'
        
        if n_lags is None:
            n_lags = self.n_lags_wl
        
        if not isinstance(n_lags, int):
            raise ValueError(f"{n_lags} is not a valid number of lags for a correlation sequence")
            
        while True:
            # check, if it is possible to simply return previously computed C/ACF
            if kwargs:
                break
            if self.corr_matrix_wl is None:
                break
            if self.corr_matrix_wl.shape[2] != n_lags:
                break
            if (self.corr_matrix_wl.shape[1] == self.num_ref_channels) != refs_only:
                break
            
            logger.debug("Using previously computed Correlation functions (Welch)...")
            return self.corr_matrix_wl
        
        logger.info("Estimating Correlation Function by Welch's method...")
        
        logger.debug(f'Arguments corr_welch: n_lags={n_lags}, refs_only={refs_only}, {kwargs}')
        
        # F(-omega) = conj(F(omega)) for real f(t)
        #
        # S_fg(omega) = F(omega)conj(G(omega))
        #             = conj(F(-omega)conj(G(-omega))
        #             = conj(F(-omega))G(-omega)
        #             = S_gf(-omega)
        #             = conj(S_gf(omega))
        #
        # onesided, i.e. RFFT suffices for real inputs f and g
        # correlation functions are also real, so IRFFT should suffice
        psd_matrix = self.psd_welch(n_lines=(n_lags - 1) * 2, refs_only=refs_only, **kwargs)
        
        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
        else:
            num_ref_channels = num_analised_channels
        corr_matrix_shape = (num_analised_channels, num_ref_channels, n_lags)

        corr_matrix = np.empty(corr_matrix_shape)

        pbar = simplePbar(num_analised_channels * num_ref_channels)
        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                next(pbar)
                this_psd = psd_matrix[channel_1, channel_2, :]
                this_corr = np.fft.irfft(this_psd)
                this_corr = this_corr[:n_lags].real
                # divide by n_lines [equivalence of r(0) and Var(y)]
                this_corr /= (n_lags - 1) * 2

                corr_matrix[channel_1, channel_2, :] = this_corr
        
        logger.debug(f'0-lag Auto-/Cross-Correlations: {np.abs(corr_matrix[:, :, 0]) * (n_lags - 1) * 2}')
        self.corr_matrix_wl = corr_matrix
        
        return corr_matrix
    
    def corr_blackman_tukey(self, n_lags, num_blocks=1, refs_only=True, **kwargs):
        '''
        Estimate the (cross- and auto-) correlation functions (C/ACF),
        by direct computation of the standard un-biased estimator:
        
        .. math::
        
           \\hat{R}_{fg}[m] = \\frac{1}{N - m}\\sum_{n=0}^{N - m - 1} f[n] g[n + m]
        
        Computes correlation functions of all channels with selected reference
        channels up to, but excluding, a time lag of n_lags. Normalization
        is done according to the unbiased estimator, i.e. 0-lag correlation
        value must be multiplied by n_lines to get the signals cross-power.
        
        Variance estimation for each time lag is performed by dividing
        the signals into num_blocks non-overlapping blocks for individual
        estimation of correlation functions. With increasing numbers of
        non-overlapping blocks the confidence intervals of the correlation
        functions increase ("worsen"), especially at higher lags and
        short block lengths, due to a larger number of time steps being
        discarded.
        
        Note that:
            n_lags \= n_lines // 2 + 1
        
            n_lines \= (n_lags - 1) * 2

        Parameters
        ----------
            n_lags: integer, optional
                Number of lags (positive). Note: this includes the
                0-lag, therefore excludes the "n_lags"-lag.
            num_blocks: integer, optional
                Number of blocks to perform averaging over. If blocks
                are shorter than n_lags it raises a ValueError.
            refs_only: bool, optional
                Compute cross-ACFss only with reference channels
            
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are currently not used
        
        Returns
        -------
            corr_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lags)
                containing the correlation values of the respective
                channels and lags
        
        See also
        --------
            corr_welch:
                Correlation function estimation by Welch's method, possibly
                faster, but distorted for short segments and biased through
                windowing.
        '''
        assert isinstance(n_lags, int)
        
        self._last_meth = 'blackman-tukey'
        
        while True:
            # check, if it is possible to simply return previously computed C/ACF
            if kwargs:
                pass
                #break
            if self.corr_matrix_bt is None:
                break
            if self.n_lines_bt != (n_lags - 1) * 2:
                break
            if num_blocks is not None and self.n_segments_bt != num_blocks:
                break
            if (self.corr_matrix_bt.shape[1] == self.num_ref_channels) != refs_only:
                break
            
            logger.debug("Using previously computed Correlation Functions (BT)...")
            return self.corr_matrix_bt
        
        logger.info(f'Estimating Correlation Functions (BT) with n_lags='
                    f'{n_lags} and num_blocks={num_blocks}...')
        
        logger.debug(f'Arguments corr_blackman_tukey: n_lags={n_lags}, num_blocks={num_blocks}, refs_only={refs_only}, {kwargs}')
        
        total_time_steps = self.total_time_steps
        # increasing block length decreases variance (for non-overlapping blocks)
        # use the maximum possible block length
        N_block = total_time_steps // num_blocks
        logger.debug(f"Blocklength: {N_block}")
        if N_block < n_lags:
            raise ValueError(f'Given block length ({N_block})'
                             f'must be greater or equal to max time lag ({n_lags})')
        
        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
            ref_channels = self.ref_channels
        else:
            num_ref_channels = num_analised_channels
            ref_channels = list(range(num_ref_channels))
        
        signals = self.signals
        
        corr_matrix_shape = (num_analised_channels, num_ref_channels, n_lags)
        corr_matrices = []
        
        pbar = simplePbar(n_lags * num_blocks)
        for block in range(num_blocks):
            this_corr_matrix = np.empty(corr_matrix_shape)
            this_signals_block = signals[block * N_block:(block + 1) * N_block, :]
            
            for lag in range(n_lags):
                next(pbar)
                # theoretically (unbounded, continuous): conj(R_fg) = R_gf
                # for f and g being reference channels, additional
                # performance improvements may be implemented
                # currently, both are computed individually
                y_r = this_signals_block[:N_block - lag, ref_channels]
                y_a = this_signals_block[lag:, :]
    
                # standard un-biased estimator (revert rectangular window)
                this_block = (y_a.T @ y_r) / (N_block - lag)
                
                this_corr_matrix[:, :, lag] = this_block
            
            corr_matrices.append(this_corr_matrix)

        corr_matrix = np.mean(corr_matrices, axis=0)
        
        logger.debug(f'0-lag Auto-/Cross-Correlations: {np.abs(corr_matrix[:, :, 0]) * total_time_steps}')
        assert np.all(corr_matrix.shape == corr_matrix_shape)
        
        self.corr_matrix_bt = corr_matrix
        self.var_corr_bt = np.var(corr_matrices, axis=0)
        self.n_lines_bt = (n_lags - 1) * 2
        self.n_segments_bt = num_blocks

        return corr_matrix
        
    def psd_blackman_tukey(self, n_lines=None, refs_only=True, window='hamming', **kwargs):
        '''
        Estimate the (cross- and auto-) power spectral densities (PSD),
        by Fourier Transform of correlation functions estimated
        according to Blackman-Tukey's method. Non-negativeness of the
        PSD is ensured by using a lag window, i.e. convolving the temporal
        window with itself. Normalization is applied w.r.t. conservation of
        energy, i.e. magnitudes will change with n_lines but power stays
        constant.

        Note that:
            n_lags = n_lines // 2 + 1
            n_lines = (n_lags - 1) * 2
            
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
            refs_only: bool, optional
                Compute cross-PDSs only with reference channels
            window: str or tuple or array_like, optional
                Desired temporal window to be applied to the correlation
                sequence after conversion to a lag window by "self-convolution"
                See scipy.signal.get_window() for more information
            
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to self.corr_blackman_tukey

        Returns
        -------
            psd_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lines // 2 + 1)
                containing the power density values of the respective
                channels and frequencies
                
        '''
        
        self._last_meth = 'blackman-tukey'
        
        while True:
            # check, if it is possible to simply return previously computed PSD
            if kwargs:
                break
            if self.psd_matrix_bt is None:
                break
            if self.psd_matrix_bt.shape[2] != n_lines // 2 + 1:
                break
            if (self.psd_matrix_bt.shape[1] == self.num_ref_channels) != refs_only:
                break
            
            logger.debug("Using previously computed Power Spectral Density (BT)...")
            return self.psd_matrix_bt
        
        logger.info("Estimating Power Spectral Density by Blackman-Tukey's method...")
        
        logger.debug(f'Arguments psd_blackman_tukey: n_lines={n_lines}, refs_only={refs_only}, window={window}, {kwargs}')
        
        if n_lines % 2:
            n_lines += 1
            logger.warning(f"Only even number of frequency lines are supported setting n_lines={n_lines}")
        
        corr_matrix = self.corr_blackman_tukey(n_lines // 2 + 1, refs_only=refs_only, **kwargs)
        
        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
        else:
            num_ref_channels = num_analised_channels
            
        psd_matrix_shape = (num_analised_channels,
                            num_ref_channels,
                            n_lines // 2 + 1)

        psd_matrix = np.empty(psd_matrix_shape, dtype=complex)

        # create a symmetrical window, i.e. lacking the last 0 (for an even number of lines)
        # win = getattr(np, window)(n_lines // 2 + 1)[:n_lines // 2]
        win = scipy.signal.get_window(window, n_lines // 2, fftbins=True)
        # Zero-Pad both sides (= zero pad once and circular convolution)
        # to allow the window to "slide along" the correct number of lags in np.convolve = 3 * n_lines//2 - 1
        # here first (!) zero pad is n_lines//2-1 because it is convolve
        win_pad = np.concatenate((np.zeros(n_lines // 2 - 1), win, np.zeros(n_lines // 2)))
        # Convolve zero-padded and unpadded window
        # resulting shape: M - N + 1 = (3 * n_lines//2 - 1) - (n_lines//2) + 1 = 2 * n_lines//2 = n_lines
        corr_win = np.convolve(win_pad, win, 'valid')
        corr_win /= n_lines // 2  # -np.abs(k_dir) # unbiased not needed here, because it is "windowed"
        
        # normalization factor for power equivalence
        norm_fact = self.total_time_steps
        # equivalent noise bandwidth of the window for density scaling
        eq_noise_bw = np.sum(win**2) / np.sum(win)**2 * (n_lines // 2)
        
        pbar = simplePbar(num_analised_channels * num_ref_channels)
        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                next(pbar)
                # https://en.wikipedia.org/wiki/Cross-correlation#Properties
                # for real f and g: R_fg(tau) = R_gf(tau)
                # for all f and g: R_fg(-tau) = R_gf(tau)
                corr_seq = corr_matrix[channel_1, channel_2, :]
                corr_sequence = np.concatenate((np.flip(corr_seq)[:n_lines // 2], corr_seq[:n_lines // 2]))
                # normalize 0-lag correlation to signal power -> spectral power
                corr_sequence *= norm_fact
                
                # apply window and compute the spectrum by the FFT
                spec_btr = np.fft.fft(corr_sequence * corr_win)
                
                # restrict the spectrum to positive frequencies
                spec_btr = spec_btr[:n_lines // 2 + 1]
                
                # compensate one-sided
                spec_btr *= 2
                
                # compensate window
                spec_btr *= eq_noise_bw
                
                psd_matrix[channel_1, channel_2, :] = spec_btr
        # plt.show()
        logger.debug(f'PSD Auto-/Cross-Powers: {np.mean(np.abs(psd_matrix), axis=2)}')
        
        if self.scaling_factors is None:
            # obtain the scaling factors for the PSD which remain,
            # even after filtering or any DSP other operation
            self.scaling_factors = psd_matrix.max(axis=2)
        
        self.psd_matrix_bt = psd_matrix
        
        self._last_meth = 'blackman-tukey'
        
        return psd_matrix
    
    def welch(self, n_lines, **kwargs):
        logger.warning("DeprecationWarning: method welch() will soon be dropped. Use psd_welch and/or corr_welch instead")
        psd_matrix = self.psd_welch(n_lines, **kwargs)
        corr_matrix = self.corr_welch()
        
        return corr_matrix, psd_matrix

    def correlation(self, n_lags=None, method=None, **kwargs):
        '''
        A convenience method for obtaining the correlation sequence by
        the default or any specified estimation method.
        
        Parameters
        ----------
            n_lags: integer, optional
                Number of lags (positive). Note: this includes the
                0-lag, therefore excludes the n_lags-lag.
            method: str, optional
                The method to use for spectral estimation
        
        Other Parameters
        -----------------
            kwargs:
                Additional parameters are passed to the spectral estimation method
        
        Returns
        -------
            corr_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lags)
                containing the correlation values of the respective
                channels and lags
        
        '''
        logger.debug(f'Arguments correlation: n_lags={n_lags}, method={method}, {kwargs}')
        
        if method is None:
            if self._last_meth is None:
                method = 'blackman-tukey'
            else:
                method = self._last_meth
        if method == 'welch':
            if n_lags is None:
                n_lags = self.n_lags_wl
            if not isinstance(n_lags, int):
                raise ValueError(f"{n_lags} is not a valid number of lags for a correlation sequence")
            return self.corr_welch(n_lags, **kwargs)
        elif method == 'blackman-tukey':
            if n_lags is None:
                n_lags = self.n_lags_bt
            if not isinstance(n_lags, int):
                raise ValueError(f"{n_lags} is not a valid number of lags for a correlation sequence")
            return self.corr_blackman_tukey(n_lags, **kwargs)
        else:
            raise ValueError(f'Unknown method {method}')
        
    def psd(self, n_lines=None, method=None, **kwargs):
        '''
        A convenience method for obtaining the PSD by the default or any
        specified estimation method.
        
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
            method:
                The method to use for spectral estimation

        Other Parameters
        ----------------
            **kwargs:
                Additional parameters are passed to the spectral estimation method
                
        Returns
        -------
            psd_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lines // 2 + 1)
                containing the power density values of the respective
                channels and frequencies
        
        '''
        
        logger.debug(f'Arguments psd: n_lines={n_lines}, method={method}, {kwargs}')
        
        if method is None:
            if self._last_meth is None:
                method = 'welch'
            else:
                method = self._last_meth
        if method == 'welch':
            if n_lines is None:
                n_lines = self.n_lines_wl
            if not isinstance(n_lines, int):
                raise ValueError(f"{n_lines} is not a valid number of frequency lines for a psd sequence")
            return self.psd_welch(n_lines, **kwargs)
        elif method == 'blackman-tukey':
            if n_lines is None:
                n_lines = self.n_lines_bt
            if not isinstance(n_lines, int):
                raise ValueError(f"{n_lines} is not a valid number of frequency lines for a psd sequence")
            return self.psd_blackman_tukey(n_lines, **kwargs)
        else:
            raise ValueError(f'Unknown method {method}')
        
    def sv_psd(self, n_lines=None, **kwargs):
        '''
        Compute the singular values of the power spectral density matrices,
        for which the complete (all cross spectral densities) matrices are used.
        
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
        
        Other Parameters
        ----------------
            kwargs:
                Additional parameters are passed to the spectral estimation method
        '''
        
        num_analised_channels = self.num_analised_channels
        psd_matrix = self.psd(n_lines, refs_only=False, **kwargs)
        
        n_lines = self.n_lines
        s_vals_psd = np.empty((num_analised_channels, n_lines // 2 + 1))
        for k in range(n_lines // 2 + 1):
            # might use only real part to account for slightly asynchronous data
            # see [Au (2017): OMA, Chapter 7.5]
            s_vals_psd[:, k] = np.linalg.svd(psd_matrix[:, :, k], True, False)
            
        return s_vals_psd
        
    def plot_signals(self,
                     channels=None,
                     per_channel_axes=False,
                     timescale='time',
                     psd_scale='db',
                     axest=None,
                     axesf=None,
                     plot_kwarg_dict={},
                     **kwargs):
        
        '''
        Plot time domain and/or frequency domain signals in various configurations:
         1. time history and spectrum of a single channel in two axes -> set channels = [channel] goto 2
         2. time history of multiple channels (all channels or specified)
            * if axes arguments are not None:must be (tuples, lists, ndarrays) of size = (num_channels,) regardless of the actual figure layout
            * else: generate axes for each channel and arrange them in lists
             
            a. time domain overlay in a single axes -> single axes is repeated in the axes list
                i. spectrum overlay in a single axes -> single axes is repeated in the axes list
                ii. svd spectrum in a single axes -> needs an additional argument
            b. in multiple axes' in a grid figure -> axes are generated as subplots
                i. spectrum in multiple axes' -> axes are generated as subplots
                ii. svd spectrum in a single axes -> needs an additional argument
        
        Parameters
        ----------
            channels : None, list-of-int, list-of-str, int, str
                The selected channels (see self._channel_numbers for explanation)
            per_channel_axes: bool
                Whether to plot all channels into a single or multiple axes
            timescale: str ['time', 'samples', 'lags']
                Whether to display time, sample or lag values on the horizontal axis
                'lags' implies plotting (auto)-correlations instead of raw time histories
            psd_scale: str, ['db', 'power', 'rms', 'svd', 'phase']
                Scaling/Output quantity of the ordinate (value axis)
            axest: ndarray of size num_channels of matplotlib.axes.Axes objects
                User provided axes objects, into which to plot time domain signals
            axesf: ndarray of size num_channels of matplotlib.axes.Axes objects
                User provided axes objects, into which to plot spectra
        
        Other Parameters
        ----------------
            plot_kwarg_dict:
                A dictionary to pass arguments to matplotlib.plot
            kwargs:
                Additional kwargs are passed to the spectral estimation method
        '''
        refs = kwargs.pop('refs', None)
        channel_numbers, ref_numbers = self._channel_numbers(channels, refs)
        all_ref_numbers = set(sum(ref_numbers, []))
        # if all requested reference channels are in self.ref_channels,
        # a reduced correlation function may be computed
        refs_only = all_ref_numbers.issubset(self.ref_channels)
        # if not all are needed, but the user requested so, compute full correlation matrix
        if (refs_only and not kwargs.pop('refs_only', True)) or psd_scale == 'svd':
            refs_only = False
        num_channels = len(channel_numbers)
        
        if axest is None or axesf is None:
            if per_channel_axes:
                if psd_scale != 'svd': 
                    # creates a subplot with side by side time and frequency domain plots for each channel
                    _, axes = plt.subplots(nrows=num_channels,
                                           ncols=2,
                                           sharey='col',
                                           sharex='col')
                    if axest is None:
                        axest = axes[:, 0]
                    if axesf is None:
                        axesf = axes[:, 1]
                else:
                    if axest is None:
                        # creates a subplot for time domain plots of each channel
                        nxn = int(np.ceil(np.sqrt(num_channels)))
                        _, axest = plt.subplots(nrows=int(np.ceil(num_channels / nxn)),
                                                ncols=nxn,
                                                sharey=True,
                                                sharex=True)
                        axest = axest.flatten()
                    if axesf is None:
                        # creates a separate figure for the svd spectrum
                        _, axesf = plt.subplots(nrows=1,
                                                ncols=1)
                        axesf = np.repeat(axesf, num_channels)
            else:
                if axest is None:
                    # create a single figure for overlaying all time domain plots
                    _, axest = plt.subplots(nrows=1,
                                            ncols=1)
                    axest = np.repeat(axest, num_channels)
                    
                if axesf is None:
                    # create a single figure for overlaying all spectra  or svd spectrum
                    _, axesf = plt.subplots(nrows=1,
                                            ncols=1)
                    axesf = np.repeat(axesf, num_channels)
        
        # Check the provided axes objects
        if per_channel_axes:
            if len(axest) < num_channels:
                raise ValueError(f'The number of provided axes objects '
                                 f'(time domain) = {len(axest)} does not match the '
                                 f'number of channels={num_channels}')
        if per_channel_axes and psd_scale != 'svd':
            if len(axesf) < num_channels:
                raise ValueError(f'The number of provided axes objects '
                                 f'(frequency domain) = {len(axesf)} does not match the '
                                 f'number of channels={num_channels}')
        if not per_channel_axes:
            if not isinstance(axest, (tuple, list, np.ndarray)):
                axest = np.repeat(axest, num_channels)
            elif len(axest) == 1:
                axest = np.repeat(axest, num_channels)
            elif len(axest) < num_channels:
                raise ValueError(f'The number of provided axes objects '
                                 f'(time domain) = {len(axest)} does not match the '
                                 f'number of channels={num_channels}')
                
        if not per_channel_axes or psd_scale:
            if not isinstance(axesf, (tuple, list, np.ndarray)):
                axesf = np.repeat(axesf, num_channels)
            elif len(axesf) == 1:
                print(axesf, num_channels)
                axesf = np.repeat(axesf, num_channels)
            elif len(axesf) < num_channels:
                raise ValueError(f'The number of provided axes objects '
                                 f'(frequency domain) = {len(axesf)} does not match the '
                                 f'number of channels={num_channels}')
        
        # precompute relevant spectral matrices
        n_lines = kwargs.pop('n_lines', None)
        method = kwargs.pop('method', self._last_meth)
        self.psd(n_lines, method, refs_only=refs_only, **kwargs.copy())
        if timescale == 'lags':
            self.correlation(self.n_lags, method, refs_only=refs_only, **kwargs.copy())
        
        for axt, axf, channel in zip(axest, axesf, channel_numbers):
            
            if timescale == 'lags':
                #omitting **kwargs here to not trigger recomputation, except refs_only
                self.plot_correlation(self.n_lags, [channel], axt, timescale, refs,
                                      plot_kwarg_dict.copy(),
                                      refs_only=refs_only, method=method)
            else:
                self.plot_timeseries(channels=[channel], ax=axt, scale='timescale', **plot_kwarg_dict.copy())
                
            axt.grid(True, axis='y', ls='dotted')
            
            #omitting **kwargs here to not trigger recomputation, except refs_only
            self.plot_psd(n_lines, [channel], axf, psd_scale, refs,
                          plot_kwarg_dict.copy(),
                          refs_only=refs_only, method=method)
        
        if not per_channel_axes:
            axest[-1].legend()
            axesf[-1].legend()
        else:
            figt = axest[0].get_figure()
            figt.legend()
            figf = axesf[0].get_figure()
            figf.legend()
            
        return axest, axesf
    
    def plot_timeseries(self, channels=None, ax=None, scale='time', **kwargs):
        '''
        Plots the time histories of the signals
        
        Parameters
        ----------
            channels : int, list, tuple, np.ndarray
                The channels to plot, may be names, indices, etc.
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object to plot into
            scale: str, ['lags','samples']
                Whether to display time or sample values on the horizontal axis
        
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to the spectral matplotlib.plot
        Returns
        -------
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object containing the graphs
                
        .. TODO::
             * correct labeling of channels and axis (using accel\_, velo\_, and disp\_channels)
        '''
            
        signals = self.signals
        
        t = self.t
        if scale == 'samples':
            t *= self.sampling_rate
            xlabel = '$n$ [-]'
            ylabel = '$y[n]$ [...]'
        else:
            xlabel = '$t$ [s]'
            ylabel = '$y(t)$ [...]'
        
        channel_numbers, _ = self._channel_numbers(channels)

        if ax is None:
            ax = plt.subplot(111)
        
        for channel in channel_numbers:
            channel_name = self.channel_headers[channel]
            
            ax.plot(t, signals[:, channel], label=f'$y_\mathrm{{{channel_name}}}$', **kwargs)
            
        ax.set_xlim((0, self.duration))
        if ax.is_last_row():
            ax.set_xlabel(xlabel)
        if ax.is_first_col():
            ax.set_ylabel(ylabel)
        
        return ax
    
    def plot_correlation(self, n_lags=None, channels=None, ax=None,
                         scale='lags', refs=None, plot_kwarg_dict={}, **kwargs):
        '''
        Plots the Cross- and Auto-Correlation sequences of the signals.
        If correlations have not been estimated yet and no method
        parameter is supplied, Blackman-Tukeys's method is used, else the
        most recently used estimation method is employed.
        
        Parameters
        ----------
            n_lags: integer, optional
                Number of lags (positive). Note: this includes the
                0-lag, therefore excludes the n_lags-lag.
            channels : int, list, tuple, np.ndarray
                The channels to plot, may be names, numbers/indices, etc.
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object to plot into
            scale: str, ['lags','samples']
                Whether to display lag or sample values on the horizontal axis
            refs: 'auto', list-of-indices, optional
                Reference channels to consider for cross-correlations
            
        Other Parameters
        ----------------
            method:
                The method to use for spectral estimation
            plot_kwarg_dict:
                A dictionary to pass arguments to matplotlib.plot
            **kwargs :
                Additional kwargs are passed to the spectral estimation
                method or contain figure/axes formatting options

        Returns
        -------
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object containing the graphs
                
        .. TODO::
            * correct labeling of channels and axis (using accel\_, velo\_, and disp\_channels)
         
        '''
        
        method = kwargs.pop('method', self._last_meth)
        assert method is not None
        # inspect, which reference channels are needed; ref_numbers is a list-of-lists
        channel_numbers, ref_numbers = self._channel_numbers(channels, refs)
        all_ref_numbers = set(sum(ref_numbers, []))
        # if all requested reference channels are in self.ref_channels,
        # a reduced correlation function may be computed
        refs_only = all_ref_numbers.issubset(self.ref_channels)
        # if not all are needed, but the user requested so or full correlation matrix has been computed already -> use that
        if refs_only:
            if method == 'welch' and self.corr_matrix_wl is not None:
                refs_only = self.num_ref_channels == self.corr_matrix_wl.shape[1]
                logger.debug(f'reverting refs_only: False -> Welch precomputed')
            elif method == 'blackman-tukey' and self.corr_matrix_bt is not None:
                refs_only = self.num_ref_channels == self.corr_matrix_bt.shape[1]
                logger.debug(f'reverting refs_only: False -> Blackman-Tukey precomputed')
            if not kwargs.pop('refs_only', True):
                refs_only = False
                logger.debug(f'reverting refs_only: False -> User input')
        
        corr_matrix = self.correlation(n_lags, refs_only=refs_only, method=method, **kwargs)

        assert refs_only is (self.num_ref_channels == corr_matrix.shape[1])
        
        lags = self.lags
        if scale == 'samples':
            lags *= self.sampling_rate
            xlabel = '$m$ [-]'
            ylabel = '$\hat{R}_{i,j}[m]$ [...]'
        else:
            xlabel = '$\\tau$ [s]'
            ylabel = '$\hat{R}_{i,j}(\\tau)$ [...]'
        
        if ax is None:
            ax = plt.subplot(111)
        # print(lags.shape, corr_matrix.shape)
        for channel_number, current_ref_numbers in zip(channel_numbers, ref_numbers):
            channel_name = self.channel_headers[channel_number]
            for ref_index, ref_number in enumerate(current_ref_numbers):
                
                if refs_only:
                    # reduced-channel correlation matrix is indexed by reference channel indices
                    corr = corr_matrix[channel_number, ref_index, :]
                else:
                    # full-channel  correlation matrix is indexed by reference channel numbers
                    corr = corr_matrix[channel_number, ref_number, :]
                    
                if method == 'welch':
                    norm_fact = self.n_lines
                elif method == 'blackman-tukey':
                    norm_fact = self.total_time_steps
                
                if ref_number == channel_number:
                    label = f'$\hat{{R}}_\mathrm{{{channel_name}}}$'
                else:
                    ref_name = self.channel_headers[ref_number]
                    label = f'$\hat{{R}}_\mathrm{{{ref_name},{channel_name}}}$'
                
                ax.plot(lags, corr * norm_fact, label=label, **plot_kwarg_dict)
                
        ax.set_xlim((0, lags.max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        return ax

    def plot_psd(self, n_lines=2048, channels=None, ax=None,
                 scale='db', refs=None, plot_kwarg_dict={}, **kwargs):
        '''
        Plots the Cross- and Auto-Power-Spectral Density of the signals.
        PSD estimation is performed by default using Welch's method.
        
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
            channels : int, list, tuple, np.ndarray
                The channels to plot, may be names, indices, etc.
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object to plot into
            scale: str, ['db', 'power', 'rms', 'svd', 'phase']
               Scaling/Output quantity of the ordinate (value axis)
            refs: 'auto', list-of-indices, optional
                Reference channels to consider for cross-correlations
        
        Other Parameters
        ----------------
            method:
                The method to use for spectral estimation
            plot_kwarg_dict:
                A dictionary to pass arguments to matplotlib.plot
            **kwargs :
                Additional kwargs are passed to the spectral estimation method

        Returns
        -------
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object containing the graphs
                
        .. TODO::
            * correct labeling of channels and axis (using accel\_, velo\_, and disp\_channels)
            * do we need a svd in non-db scale?
            * do we need sample scaling on the abscissa
        '''
        
        assert scale in ['db', 'power', 'rms', 'svd', 'phase']
        
        method = kwargs.pop('method', self._last_meth)
        if scale == 'svd':
            if refs is not None or kwargs.pop('refs_only', False):
                logger.warning("Reference channels are not used in SVD PSD.")
            refs_only = False
            channel_numbers, ref_numbers = self._channel_numbers(channels, [0])
            psd_matrix = self.sv_psd(n_lines, method=method, **kwargs)
        else:
            # inspect, which reference channels are needed; ref_numbers is a list-of-lists
            channel_numbers, ref_numbers = self._channel_numbers(channels, refs)
            all_ref_numbers = set(sum(ref_numbers, []))
            # if all requested reference channels are in self.ref_channels,
            # a reduced correlation function may be computed
            refs_only = all_ref_numbers.issubset(self.ref_channels)
            # if not all are needed, but the user requested so or full psd matrix has been computed already -> use that
            if refs_only:
                if method == 'welch' and self.psd_matrix_wl is not None:
                    refs_only = self.num_ref_channels == self.psd_matrix_wl.shape[1]
                elif method == 'blackman-tukey' and self.psd_matrix_bt is not None:
                    refs_only = self.num_ref_channels == self.psd_matrix_bt.shape[1]
                if not kwargs.pop('refs_only', True):
                    refs_only = False
            
            psd_matrix = self.psd(n_lines, refs_only=refs_only, method=method, **kwargs)
            assert refs_only is (self.num_ref_channels == psd_matrix.shape[1])
        
        # self.freqs refers to the last call of any spectral estimation method
        freqs = self.freqs
        
        if ax is None:
            ax = plt.subplot(111)
            
        for channel_number, current_ref_numbers in zip(channel_numbers, ref_numbers):
            channel_name = self.channel_headers[channel_number]
            for ref_index, ref_number in enumerate(current_ref_numbers):
                
                if scale == 'svd':
                    psd = psd_matrix[channel_number, :]
                    psd = 10 * np.log10(np.abs(psd))
                    ref_name = ''
                elif refs_only:
                    # reduced-size psd matrix is indexed by reference channel indices
                    psd = psd_matrix[channel_number, ref_index, :]
                    ref_name = self.channel_headers[ref_number]
                else:
                    # full-size  psd matrix is indexed by reference channel numbers
                    psd = psd_matrix[channel_number, ref_number, :]
                    ref_name = self.channel_headers[ref_number]
                    
                if scale == 'db':
                    psd = 10 * np.log10(np.abs(psd))
                elif scale == 'power':
                    psd = np.abs(psd)
                elif scale == 'rms':
                    psd = np.sqrt(np.abs(psd))
                elif scale == 'phase':
                    psd = np.angle(psd) / np.pi * 180
                
                if ref_number == channel_number:
                    label = f'$\hat{{S}}_\mathrm{{{channel_name}}}$'
                else:
                    ref_name = self.channel_headers[ref_number]
                    label = f'$\hat{{S}}_\mathrm{{{ref_name},{channel_name}}}$'
                    
                ax.plot(freqs, psd, label=label, **plot_kwarg_dict)
        
        ax.set_xlim((0, freqs.max()))
        ax.set_xlabel('$f$ [Hz]')
        if scale == 'svd':
            ax.set_ylabel('Singular Value Magnitude [dB]')
        elif scale == 'db':
            ax.set_ylabel('PSD [dB]')
        elif scale == 'power':
            ax.set_ylabel('Power Spectral Density [...]')
        elif scale == 'rms':
            ax.set_ylabel('Magnitude Spectral Density [...]')
        elif scale == 'phase':
            ax.set_ylabel('Cross Spectrum Phase[°]')
        
        return ax

    def plot_svd_spectrum(self, NFFT=512, log_scale=True, ax=None):
        logger.warning("DeprecationWarning: method plot_svd_spectrum() will soon be dropped. Use plot_psd(scale='svd')")
        if not log_scale:
            raise NotImplementedError("Log scale for SVD plots cannot be deactivated")
        return self.plot_psd(n_lines=NFFT, ax=ax, scale='svd')
        

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


def spectral_estimation():
    # signal parameters
    N = 2**15
    fs = 128
    dt = 1/fs
    
    
    t, y, omegas, psd, corr = SDOF_ambient(N, fs)
    # spectral estimation parameters
    nperseg_fac = 1
    window = np.hamming
    n_lines = N // nperseg_fac
    
    tau = np.linspace(0, n_lines / fs, n_lines, False)
    omegasr = np.fft.rfftfreq(n_lines, 1 / fs) * 2 * np.pi
    
    do_plot = True
    
    if do_plot:
        fig1, axes = plt.subplots(2,2, sharex='row', sharey='row')
        ax1,ax2,ax3,ax4 = axes.flat
        for ax in axes.flat:
            ax.axhline(0, color='gray', linewidth=0.5)
        handles = []
    
    if do_plot:
        ax1.plot(np.fft.fftshift(omegas)/2/np.pi, np.fft.fftshift(psd), label='analytic', color='black', lw=0.5)
        ax3.plot(t, corr, label='analytic', color='black', lw=0.5)
        ax2.plot(np.fft.fftshift(omegas)/2/np.pi, np.fft.fftshift(psd), label='analytic', color='black', lw=0.5)
        handles.append(ax4.plot(t, corr, label='analytic', color='black', lw=0.5)[0])
        
    print(f'Theoretic powers')
    print(f'PSD: {np.mean(psd)}')
    print(f'0-lag correlation: {correlation[0]}')
    
    prep_signals = PreProcessSignals(y[:,np.newaxis],fs)
    prep_signals.plot_signals(timescale='lags', axest = [ax3], axesf=[ax1], dbscale=False)
    plt.show()
    
    
    
    
    
def SDOF_ambient(N=2**15, fs = 128):
    dt = 1/fs
    
    omegas = np.fft.fftfreq(N, 1 / fs) * 2 * np.pi
    t = np.linspace(0, N / fs, N, False)
    
    # generate sdof system
    zeta = 0.05
    omega = 5 * 2 * np.pi * np.sqrt(1 - zeta**2)  # damped f = 5 Hz
    m = 1
    k = omega**2 * m
    #c = zeta*2*sqrt(m*k)
    H = -omegas**2 / (k * (1 + 2j * zeta * omegas / omega - (omegas / omega)**2))
    
    # generate ambient input forces
    f_scale = 10
    phase = np.random.uniform(-np.pi, np.pi, (N // 2 + 1,))
    ampli = np.exp(1j * np.concatenate((phase[:N // 2 + N % 2], -1 * np.flip(phase[1:]))))
    Pomega = f_scale * np.ones(N, dtype=complex) * ampli
    
    # make the ifft real-valued
    Pomega.imag[0] = 0
    Pomega[N // 2 + N % 2] = np.abs(Pomega[N // 2 + N % 2])
    H.imag[0] = 0
    H[N // 2 + N % 2] = np.abs(H[N // 2 + N % 2])
    
    # generate the  ambient response signal
    y = np.fft.ifft(H * Pomega)
    # add noise
    noise = np.random.normal(0, 0.125, N) # noise adds zero energy due to zero mean?
    
    # discard machine-precision zero imaginary parts
    if np.all(np.isclose(y.imag, 0)): y = y.real
    else: raise RuntimeError()
    
    power = np.sum(y**2)
    power_noise = np.sum(noise**2)
    print(f'Power time-domain: {power}')
    print(f'SNR={10*np.log10(power/power_noise)} dB')
    
    # compute analytical spectrum and correlation functions with correct scaling
    # psd = np.abs(H)**2*f_scale**2
    psd = omegas**4/ (k**2 * (1 + (4 * zeta**2 - 2) * (omegas / omega)**2 + (omegas / omega)**4)) * f_scale**2
    psd /= np.mean(psd)
    # analytical solution for convolution difficult, use numerical inverse of analytical PSD
    corr = np.fft.ifft(psd)
    # discard machine-precision zero imaginary parts
    if np.all(np.isclose(corr.imag,0)): corr = corr.real
    else: raise RuntimeError()
    
    return t, (y + noise)/np.sqrt(power), omegas, psd, corr


def system_frf(N=2**16, fmax=130, L=200, E=2.1e11, rho=7850, A=0.0343, zeta=0.01):

    df = fmax / (N // 2 + 1)
    dt = 1 / df / N
    fs = 1 / dt

    omegas = np.linspace(0, fmax, N // 2 + 1, False) * 2 * np.pi
    assert df * 2 * np.pi == (omegas[-1] - omegas[0]) / (N // 2 + 1 - 1)

    num_modes = int(np.floor((fmax * 2 * np.pi * np.sqrt(rho / E) * L / np.pi * 2 + 1) / 2))
    omegans = (2 * np.arange(1, num_modes + 1) - 1) / 2 * np.pi / L * np.sqrt(E / rho)

    frf = np.zeros((N // 2 + 1,), dtype=complex)
    zetas = np.zeros_like(omegans)
    zetas[:] = zeta
    kappas = omegans**2
#     kappas[:]=E*A/L
    for j, (omegan, zeta) in enumerate(zip(omegans, zetas)):
        frf += -omegan**2 / (kappas[j] * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2))  # Accelerance
    
    return omegas, frf

if __name__ == '__main__':
    fname = '/vegas/users/staff/womo1998/git/pyOMA/tests/files/prepsignals.npz'
    prep_signals_compat = PreProcessSignals.load_state(fname)
    prep_signals_compat.plot_signals(None, True)
    #spectral_estimation()
    #main()

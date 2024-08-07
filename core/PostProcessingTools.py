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

Created on Apr 20, 2017

@author: womo1998

Multi-Setup Merging PoSER

for each setup provide:

prep_data -> PreProcessSignals: chan_dofs, ref_channels, roving_channels
modal_data -> modal_frequencies, modal_damping, mode_shapes
stabil_data -> select_modes

changed/new variables:
    - chan_dofs
    - modal_frequencies
    - modal_damping
    - std_frequencies
    - std_damping
    - mode_shapes
    - select_modes -> actually a dummy
    - ref_channels, roving_channels

in PoGer/PreGer merging
modal_data ->  modal_frequencies, modal_damping, mode_shapes, chan_dofs, ref_channels, roving_channels
stabil_data -> select_modes

PlotMSH (or other postprocessing routines) have to distinguish these three cases:
single-setup (prep_data, modal_data, stabil_data)
poger/preger multi-setup (modal_data, stabil_data)
poser multi-setup (merged_data)
'''

import numpy as np
import datetime
from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase
from .StabilDiagram import StabilCalc
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

class MergePoSER(object):
    '''
    classdocs
    '''

    def __init__(self,):
        '''
        Constructor
        '''
        self.setups = []

        self.merged_chan_dofs = []
        self.merged_num_channels = None

        self.mean_frequencies = None
        self.mean_damping = None
        self.merged_mode_shapes = None

        self.std_frequencies = None
        self.std_damping = None

        self.setup_name = 'merged_poser'
        self.start_time = datetime.datetime.now()
        self.state = [False, False]

    def add_setup(
            self,
            prep_data,
            modal_data,
            stabil_data,
            override_ref_channels=None):
        # does not check, if same method was used for each setup, also anaylsis
        # parameters should be similar
        if override_ref_channels:
            raise RuntimeWarning('This function is not implemented yet!')

        assert isinstance(prep_data, PreProcessSignals)
        assert isinstance(modal_data, ModalBase)
        assert isinstance(stabil_data, StabilCalc)

        # assure objects belong to the same setup
        assert prep_data.setup_name == modal_data.setup_name
        assert modal_data.setup_name == stabil_data.setup_name

        # assure chan_dofs were assigned
        assert prep_data.chan_dofs

        # assure modes were selected
        assert stabil_data.select_modes

        # extract needed information and store them in a dictionary
        self.setups.append({'setup_name': prep_data.setup_name,
                            'chan_dofs': prep_data.chan_dofs,
                            'num_channels': prep_data.num_analised_channels,
                            'ref_channels': prep_data.ref_channels,
                            'modal_frequencies': [modal_data.modal_frequencies[index] for index in stabil_data.select_modes],
                            'modal_damping': [modal_data.modal_damping[index] for index in stabil_data.select_modes],
                            'mode_shapes': [modal_data.mode_shapes[:, index[1], index[0]] for index in stabil_data.select_modes]
                            })
        self.start_time = min(self.start_time, prep_data.start_time)

        print(
            'Added setup "{}" with {} channels and {} selected modes.'.format(
                prep_data.setup_name, prep_data.num_analised_channels, len(
                    stabil_data.select_modes)))

        self.state[0] = True

    def merge(self, base_setup_num=0, mode_pairing=None):
        '''
        generate new_chan_dofs
        assign modes from each setup

        ::
            for each mode:
                for each setup:
                    rescale
                    merge

        .. TODO::
             * rescale w.r.t to the average solution from all setups rather than specifying a base setup
             * compute scaling factors for each setup with each setup and average them for each setup before rescaling
             * corresponding standard deviations can be used to asses the quality of fit
        '''

        def pair_modes(frequencies_1, frequencies_2):
            delta_matrix = np.ma.array(
                np.zeros((len(frequencies_1), len(frequencies_2))))
            for index, frequency in enumerate(frequencies_1):
                delta_matrix[index, :] = np.abs(
                    (frequencies_2 - frequency) / frequency)
            mode_pairs = []
            while True:
                row, col = np.unravel_index(
                    np.argmin(delta_matrix), delta_matrix.shape)
                # TODO:: this code is useless: it always continues to the end and sets del_col to True
                for col_ind in range(delta_matrix.shape[1]):
                    if col_ind == col:
                        continue
                    if np.argmin(delta_matrix[:, col_ind]) == row:
                        del_col = False
                else:
                    del_col = True
                # TODO:: this code is useless: it always continues to the end and sets del_row to True
                for row_ind in range(delta_matrix.shape[0]):
                    if row_ind == row:
                        continue
                    if np.argmin(delta_matrix[row_ind, :]) == col:
                        del_row = False
                else:
                    del_row = True

                if del_col and del_row:
                    delta_matrix[row, :] = np.ma.masked
                    delta_matrix[:, col] = np.ma.masked
                    mode_pairs.append((row, col))
                if len(mode_pairs) == len(frequencies_1):
                    break
                if len(mode_pairs) == len(frequencies_2):
                    break
            return mode_pairs

        setups = self.setups

        # get values from base instance

        chan_dofs_base = setups[base_setup_num]['chan_dofs']
        num_channels_base = setups[base_setup_num]['num_channels']
        mode_shapes_base = setups[base_setup_num]['mode_shapes']
        frequencies_base = setups[base_setup_num]['modal_frequencies']
        damping_base = setups[base_setup_num]['modal_damping']

        del setups[base_setup_num]
        # pair channels and modes of each instance with base instance

        channel_pairing = []

        if mode_pairing is None:
            auto_pairing = True
            mode_pairing = []
        else:
            auto_pairing = False
            print('The provided mode pairs will be applied without any further checks.')

        total_dofs = 0
        total_dofs += num_channels_base
        for setup in setups:
            # calculate the common reference dofs, which may be different channels
            # furthermore reference channels for covariances need not be the reference channels for mode merging
            # channel dof assignments have to be present in each of the
            # instances

            chan_dofs_this = setup['chan_dofs']
            num_channels_this = setup['num_channels']

            these_pairs = []
            for chan_dof_base in chan_dofs_base:
                chan_base, node_base, az_base, elev_base = chan_dof_base[0:4]
                for chan_dof_this in chan_dofs_this:
                    chan_this, node_this, az_this, elev_this = chan_dof_this[0:4]
                    if node_this == node_base and az_this == az_base and elev_this == elev_base:
                        these_pairs.append((chan_base, chan_this))

            channel_pairing.append(these_pairs)

            total_dofs += num_channels_this - len(these_pairs)

            # calculate the mode pairing by minimal frequency difference
            # check that number of modes is equal in all instances (not necessarily)
            # assert len(self.selected_modes_indices) == len(instance.selected_modes_indices)
            if auto_pairing:
                frequencies_this = setup['modal_frequencies']

                mode_pairs = pair_modes(frequencies_base, frequencies_this)
                mode_pairing.append(mode_pairs)

        # delete modes not common to all instance from mode pairing
        for mode_num in range(len(frequencies_base) - 1, -1, -1):
            in_all = True
            for mode_pairs in mode_pairing:
                for mode_pair in mode_pairs:
                    if mode_pair[0] == mode_num:
                        break
                else:
                    in_all = False
                    break
            if in_all:
                continue
            for mode_pairs in mode_pairing:
                while True:
                    for index, mode_pair in enumerate(mode_pairs):
                        if mode_pair[0] == mode_num:
                            del mode_pairs[index]
                            break
                    else:
                        break

        lengths = [len(mode_pairs) for mode_pairs in mode_pairing]

        common_modes = min(lengths)

        new_mode_nums = [mode_num[0] for mode_num in mode_pairing[0]]

        # allocate output objects
        mode_shapes = np.zeros((total_dofs, 1, common_modes), dtype=complex)
        f_list = np.zeros((len(setups) + 1, common_modes))
        d_list = np.zeros((len(setups) + 1, common_modes))
        scale_factors = np.zeros((len(setups), common_modes), dtype=complex)

        start_dof = 0

        # copy modal values from base instance first
        for mode_num_base, mode_num_this in mode_pairing[0]:
            # for mode_num_base in range(common_modes):

            mode_index = new_mode_nums.index(mode_num_base)
            mode_base = mode_shapes_base[mode_num_base]

            mode_shapes[start_dof:start_dof +
                        num_channels_base, 0, mode_index, ] = mode_base
            f_list[0, mode_index] = frequencies_base[mode_num_base]
            d_list[0, mode_index] = damping_base[mode_num_base]

        start_dof += num_channels_base

        # iterate over instances and assemble output objects (mode_shapes,
        # chan_dofs)
        for setup_num, setup in enumerate(setups):

            chan_dofs_this = setup['chan_dofs']
            num_channels_this = setup['num_channels']
            mode_shapes_this = setup['mode_shapes']

            these_pairs = channel_pairing[setup_num]
            num_ref_channels = len(these_pairs)
            num_remain_channels = num_channels_this - num_ref_channels
            ref_channels_base = [pair[0] for pair in these_pairs]
            ref_channels_this = [pair[1] for pair in these_pairs]
            print('Next Instance', ref_channels_base, ref_channels_this)

            # create 0,1 matrices to extract and reorder channels from base
            # instance and this instance
            split_mat_refs_base = np.zeros(
                (num_ref_channels, num_channels_base))
            split_mat_refs_this = np.zeros(
                (num_ref_channels, num_channels_this))
            split_mat_rovs_this = np.zeros(
                (num_remain_channels, num_channels_this))

            row_ref = 0
            for channel in range(num_channels_base):
                if channel in ref_channels_base:
                    split_mat_refs_base[row_ref, channel] = 1
                    row_ref += 1

            row_ref = 0
            row_rov = 0
            # print(instance)
            for channel in range(num_channels_this):
                if channel in ref_channels_this:
                    split_mat_refs_this[row_ref, channel] = 1

                    row_ref += 1
                else:
                    split_mat_rovs_this[row_rov, channel] = 1
                    for chan_dof_this in chan_dofs_this:
                        chan, node, az, elev = chan_dof_this[0:4]
                        if chan == channel:
                            chan = int(start_dof + row_rov)
                            chan_dofs_base.append([chan, node, az, elev])
                            row_rov += 1

            # loop over modes and rescale them and merge with the other
            # instances
            for mode_num_base, mode_num_this in mode_pairing[setup_num]:
                mode_index = new_mode_nums.index(mode_num_base)

                mode_base = mode_shapes_base[mode_num_base]

                mode_refs_base = np.dot(split_mat_refs_base, mode_base)

                mode_this = mode_shapes_this[mode_num_this]

                mode_refs_this = np.dot(split_mat_refs_this, mode_this)
                mode_rovs_this = np.dot(split_mat_rovs_this, mode_this)

                numer = np.dot(
                    np.transpose(
                        np.conjugate(mode_refs_this)),
                    mode_refs_base)
                denom = np.dot(
                    np.transpose(
                        np.conjugate(mode_refs_this)),
                    mode_refs_this)

                scale_fact = numer / denom
                scale_factors[setup_num, mode_index] = (scale_fact)
                mode_shapes[start_dof:start_dof + num_remain_channels,
                            0, mode_index] = scale_fact * mode_rovs_this

                f_list[setup_num + 1,
                       mode_index] = setup['modal_frequencies'][mode_num_this]
                d_list[setup_num + 1,
                       mode_index] = setup['modal_damping'][mode_num_this]

            start_dof += num_remain_channels

        mean_frequencies = np.zeros((common_modes,))
        std_frequencies = np.zeros((common_modes,))
        mean_damping = np.zeros((common_modes,))
        std_damping = np.zeros((common_modes,))

        for mode_num_base, mode_num_this in mode_pairing[0]:
            mode_index = new_mode_nums.index(mode_num_base)

            # rescaling of mode shape
            mode_tmp = mode_shapes[:, 0, mode_index]
            abs_mode_tmp = np.abs(mode_tmp)
            index_max = np.argmax(abs_mode_tmp)
            this_max = mode_tmp[index_max]
            mode_tmp = mode_tmp / this_max
            mode_shapes[:, 0, mode_index] = mode_tmp
            mean_frequencies[mode_index, ] = np.mean(
                f_list[:, mode_index], axis=0)
            std_frequencies[mode_index, ] = np.std(
                f_list[:, mode_index], axis=0)

            mean_damping[mode_index, ] = np.mean(d_list[:, mode_index], axis=0)
            std_damping[mode_index, ] = np.std(d_list[:, mode_index], axis=0)

        self.merged_chan_dofs = chan_dofs_base
        self.merged_num_channels = total_dofs

        self.merged_mode_shapes = mode_shapes
        self.mean_frequencies = np.expand_dims(mean_frequencies, axis=1)
        self.std_frequencies = np.expand_dims(std_frequencies, axis=1)
        self.mean_damping = np.expand_dims(mean_damping, axis=1)
        self.std_damping = np.expand_dims(std_damping, axis=1)

        self.state[1] = True

    def save_state(self, fname):

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {}

        out_dict['self.state'] = self.state

        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time

        if self.state[0]:
            out_dict['self.setups'] = self.setups

        if self.state[1]:
            out_dict['self.merged_chan_dofs'] = self.merged_chan_dofs
            out_dict['self.merged_num_channels'] = self.merged_num_channels
            out_dict['self.merged_mode_shapes'] = self.merged_mode_shapes
            out_dict['self.mean_frequencies'] = self.mean_frequencies
            out_dict['self.std_frequencies'] = self.std_frequencies
            out_dict['self.mean_damping'] = self.mean_damping
            out_dict['self.std_damping'] = self.std_damping

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname):

        print('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['Setups added',
                                                    'Setups merged',
                                                    ]):
            if this_state:
                print(state_string)
        postprocessor = cls()

        setup_name = str(in_dict['self.setup_name'].item())
        start_time = in_dict['self.start_time'].item()

        postprocessor.setup_name = setup_name
        postprocessor.start_time = start_time

        if state[0]:
            postprocessor.setups = list(in_dict['self.setups'])

        if state[1]:
            postprocessor.merged_chan_dofs = [[int(float(chan_dof[0])),
                                               str(chan_dof[1]),
                                               float(chan_dof[2]),
                                               float(chan_dof[3]),
                                               str(chan_dof[-1])] for chan_dof in in_dict['self.merged_chan_dofs']]

            postprocessor.merged_num_channels = in_dict['self.merged_num_channels']
            postprocessor.merged_mode_shapes = in_dict['self.merged_mode_shapes']
            postprocessor.mean_frequencies = in_dict['self.mean_frequencies']
            postprocessor.std_frequencies = in_dict['self.std_frequencies']
            postprocessor.mean_damping = in_dict['self.mean_damping']
            postprocessor.std_damping = in_dict['self.std_damping']

        return postprocessor

    def export_results(self, fname, binary=False):

        selected_freq = self.mean_frequencies
        selected_damp = self.mean_damping

        num_modes = len(selected_freq)

        selected_MPC = StabilCalc.calculateMPC(
            self.merged_mode_shapes[:, 0, :])
        selected_MP, selected_MPD = StabilCalc.calculateMPD(
            self.merged_mode_shapes[:, 0, :])

        selected_stdf = self.std_frequencies
        selected_stdd = self.std_damping

        selected_modes = self.merged_mode_shapes

        freq_str = ''
        damp_str = ''
        ord_str = ''

        msh_str = ''
        mpc_str = ''
        mp_str = ''
        mpd_str = ''
        std_freq_str = ''
        std_damp_str = ''

        for col in range(num_modes):
            freq_str += '{:3.3f} \t\t'.format(selected_freq[col, 0])
            damp_str += '{:3.3f} \t\t'.format(selected_damp[col, 0])

            mpc_str += '{:3.3f}\t \t'.format(selected_MPC[col])
            mp_str += '{:3.2f} \t\t'.format(selected_MP[col])
            mpd_str += '{:3.2f} \t\t'.format(selected_MPD[col])

            std_damp_str += '{:3.3e} \t\t'.format(selected_stdd[col, 0])
            std_freq_str += '{:3.3e} \t\t'.format(selected_stdf[col, 0])

        for row in range(num_modes):
            msh_str += '\n           \t\t'
            for col in range(self.merged_num_channels):
                msh_str += '{:+3.4f} \t'.format(selected_modes[col, 0, row])

        export_modes = 'MANUAL MODAL ANALYSIS\n'
        export_modes += '=======================\n'
        export_modes += 'Frequencies [Hz]:\t' + freq_str + '\n'

        export_modes += 'Standard deviations of the Frequencies [Hz]:\t' + \
            std_freq_str + '\n'
        export_modes += 'Damping [%]:\t\t' + damp_str + '\n'

        export_modes += 'Standard deviations of the Damping [%]:\t' + \
            std_damp_str + '\n'

        export_modes += 'Mode shapes:\t\t' + msh_str + '\n'
        export_modes += 'Model order:\t\t' + ord_str + '\n'

        export_modes += 'MPC [-]:\t\t' + mpc_str + '\n'
        export_modes += 'MP  [\u00b0]:\t\t' + mp_str + '\n'
        export_modes += 'MPD [-]:\t\t' + mpd_str + '\n\n'

        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if binary:
            out_dict = {'selected_freq': selected_freq,
                        'selected_damp': selected_damp}

            out_dict['selected_MPC'] = selected_MPC
            out_dict['selected_MP'] = selected_MP
            out_dict['selected_MPD'] = selected_MPD
            out_dict['selected_modes'] = selected_modes

            out_dict['selected_stdf'] = selected_stdf
            out_dict['selected_stdd'] = selected_stdd

            np.savez_compressed(fname, **out_dict)

        else:
            f = open(fname, 'w')
            f.write(export_modes)
            f.close()


def main():
    from core.PreProcessingTools import PreProcessSignals, GeometryProcessor
    from core.SSICovRef import BRSSICovRef
    from core.StabilDiagram import StabilCalc
    from core.PlotMSH import ModeShapePlot
    from GUI.PlotMSHGUI import start_msh_gui

    working_dir = '/home/womo1998/Projects/2017_modal_merging_test_files/'
    interactive = False

    merger = MergePoSER()
    geometry_data = GeometryProcessor.load_geometry(
        nodes_file=working_dir + 'macec/grid_full.asc',
        lines_file=working_dir + 'macec/beam_full.asc')

    setups = ['meas_1', 'meas_2']
    for setup in setups:
        result_folder = working_dir + setup

        prep_data = PreProcessSignals.load_state(result_folder + 'prep_data.npz')
        modal_data = BRSSICovRef.load_state(
            result_folder + 'modal_data.npz', prep_data)
        stabil_data = StabilCalc.load_state(
            result_folder + 'stabi_data.npz', modal_data, prep_data)

        merger.add_setup(prep_data, modal_data, stabil_data)

    merger.merge()

    if interactive:
        mode_shape_plot = ModeShapePlot(merger, geometry_data)
        start_msh_gui(mode_shape_plot)

    merger.save_state(result_folder + 'merged_setups.npz')
    merger.export_results(result_folder + 'merged_results.txt', binary=False)


if __name__ == '__main__':
    main()

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

Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015-2018
'''

import os
import warnings
import copy

import numpy as np
import scipy.linalg

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase

import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

class BRSSICovRef(ModalBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.
        self.state = [False, False, False]

        self.num_block_columns = None
        self.num_block_rows = None
        self.toeplitz_matrix = None

        # compute_state_matrices
        self.U = None
        self.S = None
        #self.V_T = None
        self.max_model_order = None
        self.modal_contributions = None

    @classmethod
    def init_from_config(cls, conf_file, prep_data):
        assert os.path.exists(conf_file)

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Columns:'
            num_block_columns = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
            max_model_order = int(f. __next__().strip('\n'))

        ssi_object = cls(prep_data)
        ssi_object.build_toeplitz_cov(num_block_columns)
        ssi_object.compute_state_matrices(max_model_order)
        ssi_object.compute_modal_params()

        return ssi_object

    def build_toeplitz_cov(
            self,
            num_block_columns,
            num_block_rows=None,
            shift=0):
        '''
        Builds a Block-Toeplitz Matrix of Covariances with varying time lags
        ::

              <-num_block_columns*num_ref_channels ->  _
            [     R_i      R_i-1      ...      R_1    ]^
            [     R_i+1    R_i        ...      R_2    ]num_block_rows*num_analised_channels
            [     ...      ...        ...      ...    ]v
            [     R_2i-1   ...        ...      R_i    ]_
        '''
        assert isinstance(num_block_columns, int)
        if num_block_rows is None:
            num_block_rows = num_block_columns
        assert isinstance(num_block_rows, int)

        print('Assembling toeplitz matrix using pre-computed correlation functions'
              ' {} block-columns and {} block rows'.format(num_block_columns, num_block_rows + 1))

        self.num_block_columns = num_block_columns
        self.num_block_rows = num_block_rows

        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels = self.prep_data.num_ref_channels

        n_lags = num_block_rows + 1 + num_block_columns - 1
        
        if self.prep_data.n_lags is None:
            self.prep_data.correlation(n_lags, 'blackman-tukey')
            
        if self.prep_data.n_lags < n_lags:
            self.prep_data.correlation(n_lags, 'blackman-tukey')

        corr_matrix = self.prep_data.corr_matrix

        Toeplitz_matrix = np.zeros(
            (num_analised_channels * (num_block_rows + 1), num_ref_channels * num_block_columns))

        for i in range(num_block_rows + 1):
            if i == 0:
                for ii in range(num_block_columns):

                    tau = num_block_columns + i - ii + shift
                    this_block = corr_matrix[:, :, tau - 1]

                    begin_Toeplitz_row = i * num_analised_channels

                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row +
                                                        num_analised_channels), ii *
                                    num_ref_channels:(ii *
                                                      num_ref_channels +
                                                      num_ref_channels)] = this_block
            else:
                previous_Toeplitz_row = (i - 1) * num_analised_channels
                this_block = Toeplitz_matrix[previous_Toeplitz_row:(
                    previous_Toeplitz_row + num_analised_channels), 0:num_ref_channels * (num_block_columns - 1)]
                begin_Toeplitz_row = i * num_analised_channels
                Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row +
                                                    num_analised_channels), num_ref_channels:(num_ref_channels *
                                                                                              num_block_columns)] = this_block
                tau = num_block_columns + i + shift
                this_block = corr_matrix[:, :, tau - 1]

                Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row + num_analised_channels),
                                0:num_ref_channels] = this_block

#         import matplotlib.pyplot as plot
#
#         for num_channel,ref_channel in enumerate(self.prep_data.ref_channels):
#             inds=([],[])
#             for i in range(num_block_columns):
#                 row = ref_channel
#                 col = (num_block_columns-i-1)*num_ref_channels+num_channel
#                 inds[0].append(row)
#                 inds[1].append(col)
#             for ii in range(1,num_block_rows):
#                 row = (ii)*num_analised_channels+ref_channel
#                 col = num_channel
#                 inds[0].append(row)
#                 inds[1].append(col)
#             means = Toeplitz_matrix[inds]
#             #print(means.shape, sigma_r[inds,inds].shape, len(inds))
#             #plt.errorbar(range(num_block_rows+num_block_rows-1), means, yerr=np.sqrt(sigma_r[inds,inds]))
#             #print(np.sqrt(sigma_r[inds,inds]))
#
#             #plt.plot(vec_R[inds,0])
#             #plt.plot(vec_R[inds,1])
#             plt.plot(range(1,num_block_columns+num_block_rows), means)
#         plt.show()

        if shift == 0:
            self.toeplitz_matrix = Toeplitz_matrix
        else:
            return Toeplitz_matrix
        self.state[0] = True

    def compute_state_matrices(self, max_model_order=None):

        if max_model_order is not None:
            assert isinstance(max_model_order, int)

        Toeplitz_matrix = self.toeplitz_matrix

        if max_model_order is None:
            max_model_order = min(Toeplitz_matrix.shape)
        else:
            max_model_order = min(max_model_order, *Toeplitz_matrix.shape)

        print('Computing state matrices with pinv-based method, with maximum model order {}...'.format(max_model_order))

        U, S, V_T = scipy.linalg.svd(Toeplitz_matrix, 1)

        U = U[:, :max_model_order]
        V_T = V_T[:max_model_order, :]

        self.U = U
        self.S = S
        self.V_T = V_T

        self.max_model_order = max_model_order

        self.state[1] = True
        self.state[2] = False  # previous modal params are invalid now

    def compute_modal_params(self, max_modes=None, algo='svd'):
        '''
        computes the modal parameters as indicated in Peeters 1999 and Döhler 2012
        only algorithm svd is optimized for multi-order computation
        max_modes i.e. crystal clear only works with algorithm svd
        '''

        assert algo in ['svd', 'qr', 'opti']

        max_model_order = self.max_model_order
        num_block_rows = self.num_block_rows
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels = self.prep_data.num_ref_channels
        accel_channels = self.prep_data.accel_channels
        velo_channels = self.prep_data.velo_channels
        #merged_num_channels = self.merged_num_channels
        sampling_rate = self.prep_data.sampling_rate

        U = self.U
        V_T = self.V_T
        S = self.S
        S_2 = np.diag(np.power(S[:max_model_order], 0.5))
        S_2_inv = np.diag(np.power(S[:max_model_order], -0.5))

        O = np.dot(U, S_2)
        Z = np.dot(S_2, V_T)

        print('Computing modal parameters...')

        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros(
            (num_analised_channels,
             max_model_order,
             max_model_order),
            dtype=complex)
        eigenvalues = np.zeros(
            (max_model_order, max_model_order), dtype=complex)
        modal_contributions = np.zeros((max_model_order, max_model_order))

        printsteps = list(np.linspace(0, max_model_order, 100, dtype=int))
        for order in range(1, max_model_order):
            while order in printsteps:
                del printsteps[0]
                print('.', end='', flush=True)
            this_modal_frequencies, this_modal_damping, this_mode_shapes, this_eigenvalues, this_modal_contributions = \
                self.single_order_modal(order, algo, max_modes, plot_=False)

            modal_frequencies[order, :order] = this_modal_frequencies
            modal_damping[order, :order] = this_modal_damping
            mode_shapes[:, :order, order] = this_mode_shapes
            eigenvalues[order, :order] = this_eigenvalues
            modal_contributions[order, :order] = this_modal_contributions

        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues
        self.modal_contributions = modal_contributions

        print('.', end='\n', flush=True)

        self.state[2] = True

    def single_order_modal(
            self,
            order,
            algo='svd',
            max_modes=None,
            corr_synth=True,
            plot_=False):

        num_block_rows = self.num_block_rows
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels = self.prep_data.num_ref_channels
        accel_channels = self.prep_data.accel_channels
        velo_channels = self.prep_data.velo_channels
        sampling_rate = self.prep_data.sampling_rate
        n_lags = self.prep_data.n_lags

        U = self.U[:, :order]
        V_T = self.V_T[:order, :]
        S = self.S
        S_2 = np.diag(np.power(S[:order], 0.5))
        S_2_inv = np.diag(np.power(S[:order], -0.5))

        O = np.dot(U, S_2)
        Z = np.dot(S_2, V_T)

        #print('Computing modal parameters...')

        corr_mats_shape = (num_analised_channels, num_ref_channels, n_lags)
        corr_matrix_synth = np.zeros(corr_mats_shape, dtype=np.float64)
        '''
        #From Brincker with participation factor for all channels:

        M = np.zeros((num_modes, n_lags*num_analised_channels), dtype=complex)
        H = np.zeros((num_analised_channels, num_ref_channels*n_lags))

        for tau in range(n_lags):
            M[:,tau*num_analised_channels:(tau+1)*num_analised_channels]=(mu_n**tau).dot(A.T)
            H[:,tau*num_ref_channels:(tau+1)*num_ref_channels]=corr_matrix[:,:,tau]

        Gamma = H.dot(np.linalg.pinv(M))/2/np.pi
        '''

        modal_frequencies = np.zeros((order))
        modal_damping = np.zeros((order))
        mode_shapes = np.zeros((num_analised_channels, order), dtype=complex)
        eigenvalues = np.zeros((order), dtype=complex)

        modal_contributions = np.zeros((order))
        corr_matrix_data = self.prep_data.corr_matrix**2

        Sigma_data = np.zeros((num_analised_channels * num_ref_channels))
        Sigma_synth = np.zeros((num_analised_channels * num_ref_channels))
        Sigma_data_synth = np.zeros(
            (num_analised_channels * num_ref_channels, order))

        On_up = O[:num_analised_channels * num_block_rows, :order]
        On_down = O[num_analised_channels:num_analised_channels *
                    (num_block_rows + 1), :order]

        On_up_i = np.linalg.pinv(On_up)  # , rcond=1e-12)

        if algo == 'svd':
            if max_modes is not None:
                [u, s, v_t] = np.linalg.svd(On_up, 0)
                s = 1. / s[:max_modes]
                On_up_i = np.dot(np.transpose(v_t[:max_modes, :]), np.multiply(
                    s[:, np.newaxis], np.transpose(u[:, :max_modes])))
            else:
                On_up_i = np.linalg.pinv(On_up)  # , rcond=1e-12)
            state_matrix = np.dot(On_up_i, On_down)

        elif algo == 'qr':
            Q, R = np.linalg.qr(On_up)
            S = Q.T.dot(On_down)
            state_matrix = np.linalg.solve(R, S)

        C = O[:num_analised_channels, :order]
        G = Z[:order, -num_ref_channels:]

        eigval, eigvec_r = np.linalg.eig(state_matrix)

        G_m = np.linalg.solve(eigvec_r, G)
        Phi = C.dot(eigvec_r)

        conj_indices = self.remove_conjugates(eigval, eigvec_r, inds_only=True)

        if plot_:

            num_modes = len(conj_indices)
            modelist = list(range(num_modes))
            modelist = [25, 26]
            #num_modes = 5
            #num_plots = int(np.ceil(np.sqrt(num_modes)))
            # num_plots=5
            #num_plots = num_modes
            num_plots = len(modelist) + 1
            fig, axes = plt.subplots(num_plots, 2, 'col', 'none', False)
            #axes= axes.flatten()
        else:
            axes = None
        ip = 0
        for i, ind in enumerate(conj_indices):

            lambda_i = eigval[ind]

            ident = eigval == lambda_i.conj()
            ident[ind] = 1
            # ident=np.diag(ident)

            # this_Lambda=np.diag(eigval).dot(ident)
            this_eigval = eigval[ident][np.newaxis]
            this_Lambda = np.diag(eigval[ident])

            this_Phi = Phi[:, ident]
            this_G_m = G_m[ident, :]

            a_i = np.abs(np.arctan2(np.imag(lambda_i), np.real(lambda_i)))
            b_i = np.log(np.abs(lambda_i))
            freq_i = np.sqrt(a_i**2 + b_i**2) * sampling_rate / 2 / np.pi
            damping_i = 100 * np.abs(b_i) / np.sqrt(a_i**2 + b_i**2)
            mode_shape_i = np.dot(C, eigvec_r[:, ind])
            mode_shape_i = np.array(mode_shape_i, dtype=complex)

            mode_shape_i = self.integrate_quantities(
                mode_shape_i, accel_channels, velo_channels, freq_i * 2 * np.pi)

            k = np.argmax(np.abs(mode_shape_i))
            s_ik = mode_shape_i[k]
            alpha_ik = np.angle(s_ik)
            e_k = np.zeros((num_analised_channels, 1))
            e_k[k, 0] = 1
            #print(f'Scale factor {np.exp(-1j*alpha_ik)}')
            mode_shape_i *= np.exp(-1j * alpha_ik)

            modal_frequencies[i] = freq_i
            modal_damping[i] = damping_i
            mode_shapes[:, i] = mode_shape_i
            eigenvalues[i] = lambda_i

            if plot_:
                if i in modelist:
                    print(ip, i)
                    ip += 1

            if corr_synth:
                ft_freq = np.fft.rfftfreq(n_lags, d=(
                    1 / self.prep_data.sampling_rate))

                this_corr_synth = np.zeros(corr_mats_shape, dtype=np.float64)
                for tau in range(1, n_lags + 1):

                    #this_corr_synth[:,:,tau-1] = this_Phi.dot(this_Lambda**tau).dot(this_G_m).real
                    this_corr_synth[:, :, tau -
                                    1] = this_Phi.dot(this_G_m *
                                                      (this_eigval.T**tau)).real

                this_corr_synth = this_corr_synth**2

                for ref_channel in range(self.prep_data.num_ref_channels):
                    for channel in range(self.prep_data.num_analised_channels):
                        Sigma_data_synth[ref_channel * num_analised_channels + channel,
                                         i] = corr_matrix_data[channel,
                                                               ref_channel,
                                                               :].dot(this_corr_synth[channel,
                                                                                      ref_channel,
                                                                                      :].T)
                        # if
                        # np.mean(Sigma_data_synth[ref_channel*num_analised_channels+channel,
                        # i])<1e-11: continue

                        if plot_:
                            if i in modelist:
                                axes[ip, 0].plot(
                                    this_corr_synth[channel, ref_channel, :])
                                ft_synth = np.fft.rfft(
                                    this_corr_synth[channel, ref_channel, :] * np.hanning(n_lags))
                                axes[ip, 1].plot(ft_freq, np.abs(ft_synth))

                corr_matrix_synth += this_corr_synth

        if plot_:
            for ref_channel in range(self.prep_data.num_ref_channels):
                for channel in range(self.prep_data.num_analised_channels):
                    axes[0, 0].plot(
                        corr_matrix_data[channel, ref_channel, :], alpha=.5)

                    ft_meas = np.fft.rfft(
                        corr_matrix_data[channel, ref_channel, :] * np.hanning(n_lags))
                    axes[0, 1].plot(ft_freq, np.abs(ft_meas), alpha=.5)

        if corr_synth:
            for ref_channel in range(num_ref_channels):
                for channel in range(num_analised_channels):
                    corr_data = corr_matrix_data[channel, ref_channel, :]
                    corr_synth = corr_matrix_synth[channel, ref_channel, :]

                    Sigma_data[ref_channel *
                               num_analised_channels +
                               channel] = corr_data.dot(corr_data.T)
                    Sigma_synth[ref_channel *
                                num_analised_channels +
                                channel] = corr_synth.dot(corr_synth.T)
            ip = 1
            for i, ind in enumerate(conj_indices):
                rho = (Sigma_data_synth[:, i] /
                       np.sqrt(Sigma_data * Sigma_synth)).mean()
                modal_contributions[i] = rho

                if plot_:
                    if i in modelist:
                        axes[ip, 0].set_ylabel('Mode {}, \n MC={:1.2f}'.format(
                            i, rho), rotation=0, labelpad=30)
                        ip += 1
        if plot_:
            axes[-1, 0].set_xlabel('tau [-]')
            axes[-1, 1].set_xlabel('f [Hz]')
#             if plot_:
#                 fig.suptitle(str(np.sum(modal_contributions)))
#                 #plt.show()
            # print(str(np.sum(modal_contributions)))
            for ax in axes.flat:
                ax.set_yticks([])

        return modal_frequencies, modal_damping, mode_shapes, eigenvalues, modal_contributions

    def synthesize_spectrum(self, A, C, G):
        '''

        L = N*dt (duration = number_of_samples*sampling_period)
        P = N*df (maximal frequency = number of samples * frequency inverval)

        dt * df = 1/N
        L * P = N
        '''

        f_max = self.prep_data.sampling_rate / 2
        n_lags = self.prep_data.n_lags
        delta_t = 1 / self.prep_data.sampling_rate

        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels = self.prep_data.num_ref_channels
        order = A.shape[0]
        assert order == A.shape[1]

        psd_mats_shape = (num_analised_channels, num_ref_channels, n_lags)
        psd_matrix = np.zeros(psd_mats_shape, dtype=np.float64)

        I = np.identity(order)

        Lambda_0 = self.prep_data.signal_power()

        for n in range(n_lags):

            z = np.exp(0 + 1j * n * delta_t)
            psd_matrix[:, :, n] = C.dot(np.linalg.solve(
                z * I - A, G)) + Lambda_0 + G.T.dot(np.linalg.solve(1 / z * I - A.T, C.T))

        self.psd_matrix = psd_matrix

        if 0:
            ax = plt.subplot()
            omega_max = psd_matrix.shape[2]

            freqs = np.fft.rfftfreq(2 * omega_max - 1, delta_t)
            print(freqs.max())

            for ref_channel in range(num_ref_channels):
                for channel in range(num_analised_channels):
                    ax.plot(freqs, np.abs(psd_matrix[channel, ref_channel, :]))
            ax.set_xlim((0, freqs.max()))
            plt.show()

    def save_state(self, fname):

        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.]
        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        # out_dict['self.prep_data']=self.prep_data
        if self.state[0]:  # covariances
            out_dict['self.toeplitz_matrix'] = self.toeplitz_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[1]:  # state models
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.U'] = self.U
            out_dict['self.S'] = self.S
            out_dict['self.V_T'] = self.V_T
        if self.state[2]:  # modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.modal_contributions'] = self.modal_contributions

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname)
        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['Covariance Matrices Built',
                                                    'State Matrices Computed',
                                                    'Modal Parameters Computed',
                                                    ]):
            if this_state:
                print(state_string)

        assert isinstance(prep_data, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())
        assert setup_name == prep_data.setup_name
        start_time = prep_data.start_time

        assert start_time == prep_data.start_time
        ssi_object = cls(prep_data)
        ssi_object.state = state
        if state[0]:  # covariances
            ssi_object.toeplitz_matrix = in_dict['self.toeplitz_matrix']
            ssi_object.num_block_columns = int(
                in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
        if state[1]:  # state models
            ssi_object.max_model_order = int(in_dict['self.max_model_order'])
            ssi_object.U = in_dict['self.U']
            ssi_object.S = in_dict['self.S']
            ssi_object.V_T = in_dict['self.V_T']
        if state[2]:  # modal params
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']
            ssi_object.eigenvalues = in_dict['self.eigenvalues']
            ssi_object.modal_contributions = in_dict.get(
                'self.modal_contributions', None)

        return ssi_object


class PogerSSICovRef(ModalBase):
    '''
    "In the PoGER approach, first a nonparametric system model is identified
    for each setup separately. In the time domain, this nonparametric model
    consists of the correlations between all measured outputs.
    In a second step, the output correlations obtained from the different
    setups are stacked on top of each other. Extracting the modal parameters
    from the resulting correlation function yields global values for the
    eigenfrequencies and damping ratios. The identified partial mode shapes
    are stacked on top of each other in a global mode shape. However, due
    to the non-stationary ambient excitation level and the non-stationary
    ambient excitation color, it is necessary to re-scale the partial mode
    shapes in a least-squares sense, for instance to the reference DOFs of the
    first partial mode shape, just as in the PoSER approach"

    from:
    Döhler, M.; Reynders, E.; Magalhaes, F.; Mevel, L.; Roeck, G. D. & Cunha, A.
    Pre-and post-identification merging for multi-setup OMA with covariance-driven SSI
    28th International Modal Analysis Conference, 2010 , 57-70

    Analysis steps:
    * Create your geometry definitions
    * Create configuration files and channel-dof-assignments for each setup
    * Pre-process each setup using PreProcessData
    * Pre-compute correlations functions using PreProcessData.compute_correlation_functions
    (note: n_lags >= num_block_columns + num_block_rows >= 2 * num_block_columns + 1)
    * add the PreProcessData objects of each setup using add_setup
    * call pair_channels(), build_merged_subspace_matrix(),
    compute_state_matrices(), compute_modal_params()

    Notes on the reference channels:
    There are two different uses of reference channels:
    1. Reference channels for reducing the computational effort /
    improving results if noisy channels are present
    2. Reference channels for mode shape rescaling when multiple
    setups should be merged

    In PoGER merging the first group of reference channels are required
    for  joint identification. In this case, reference-based correlation
    functions are "stacked on top of each other" and then assembled into
    a joint Hankel matrix. Here, only the reference channels, that are
    present in all setups can be used.

    Based on each setups' channel-dof-assignments and selected reference
    channels, the PogerSSICovRef class automatically determines the
    reference channels for:
    * joint identification and
    * mode shape rescaling / merging.
    Thus, by changing the reference channel definition in each setup,
    the used reference channels in joint identification can be influenced.
    The reference channels for modeshape rescaling are automatically
    generated, regardless of the the definition in the setup. Rescaling
    is always done with respect to the first setup, so a "good" setup should
    always be added first.
    
    .. TODO::
        * Think about ways, how this class could be instantiated from 
          configuration files
        * Add modal contributions
        * Implement PreGER merging with variance computation in a new class
    '''

    def __init__(self,):
        '''
        Initializes class and all class variables
        channel definition: channels start at 0
        '''
        super().__init__()
        #             0             1                2              3           4
        # self.state= [Setups Added, Channels Paired, Subspace Mat., State
        # Mat., Modal Par.]
        self.state = [False, False, False, False, False]

        # __init__
        self.setup_name = 'merged_'
        #self.start_times = []

        # add_setup
        self.setups = []
        self.sampling_rate = None
        self.num_ref_channels = None
        self.n_lags = None

        # pair_channels
        self.ssi_ref_channels = None
        self.merged_chan_dofs = None
        self.merged_accel_channels = None
        self.merged_velo_channels = None
        self.merged_disp_channels = None
        self.merged_num_channels = None
        self.num_analised_channels = None
        #self.start_time = None

        # build_merged_subspace_matrix
        self.subspace_matrix = None
        self.num_block_columns = None
        self.num_block_rows = None

        # compute_state_matrices
        self.U = None
        self.S = None
        #self.V_T = None
        self.max_model_order = None

        # compute_modal_params
#         self.eigenvalues = None
#         self.modal_damping = None
#         self.modal_frequencies = None
#         self.mode_shapes = None

    def add_setup(self, prep_data):
        '''
        todo:
        check that ref_channels are equal in each setup (by number and by DOF)
        '''
        assert isinstance(prep_data, PreProcessSignals)

        # assure chan_dofs were assigned
        assert prep_data.chan_dofs

        if self.sampling_rate is not None:
            assert prep_data.sampling_rate == self.sampling_rate
        else:
            self.sampling_rate = prep_data.sampling_rate

        if self.num_ref_channels is not None:
            if self.num_ref_channels != prep_data.num_ref_channels:
                warnings.warn(
                    'This setup contains a different number of reference channels ({}), than the previous setups ({})!'.format(
                        prep_data.num_ref_channels, self.num_ref_channels))
                self.num_ref_channels = min(
                    self.num_ref_channels, prep_data.num_ref_channels)
        else:
            self.num_ref_channels = prep_data.num_ref_channels

        if self.n_lags is not None:
            self.n_lags = min(self.n_lags, prep_data.n_lags)
        else:
            self.n_lags = prep_data.n_lags

        self.setup_name += prep_data.setup_name + '_'
        # self.start_times.append(prep_data.start_time)

        # extract needed information and store them in a dictionary
        self.setups.append({'setup_name': prep_data.setup_name,
                            'num_analised_channels': prep_data.num_analised_channels,
                            'chan_dofs': prep_data.chan_dofs,
                            'ref_channels': prep_data.ref_channels,
                            # 'roving_channels': prep_data.roving_channels,
                            'accel_channels': prep_data.accel_channels,
                            'velo_channels': prep_data.velo_channels,
                            'disp_channels': prep_data.disp_channels,
                            'corr_matrix': prep_data.corr_matrix,
                            'start_time': prep_data.start_time,
                            })

        print(
            'Added setup "{}" with {} channels'.format(
                prep_data.setup_name,
                prep_data.num_analised_channels))

        self.state[0] = True
        for i in range(1, 5):
            self.state[i] = False

    def pair_channels(self, ):
        '''
        pairs channels from all given setups for the poger merging methods

        ssi_reference channels are common to all setups
        rescale reference channels are common to at least two setups

        finds common dofs from all setups and their respective channels
        generates new channel_dof_assignments with ascending channel numbers
        rescale reference channels are assumed to be equal to ssi_reference channels
        '''

        print('Pairing channels and dofs...')
        setups = self.setups
        merged_chan_dofs = []
        merged_accel_channels = []
        merged_velo_channels = []
        merged_disp_channels = []

        # extract dofs from each setup and exclude channel numbers
        # merged_chan_dofs will be a list of chan_dof lists
        '''
        merged_chan_dofs = [[dof of setup 0 channel 0,
                             dof of setup 0 channel 1,
                             ...
                             dof of setup 0 channel num_analised_channels]
                            [dof of setup 1 channel 0,
                             dof of setup 1 channel 1,
                             ...
                             dof of setup 1 channel num_analised_channels]
                            ...
                            [dof of setup num_setups channel 0,
                             dof of setup num_setups channel 1,
                             ...
                             dof of setup num_setups channel num_analised_channels]
                            ]
        '''

        for setup in setups:
            chan_dofs = []
            accel_channels = []
            velo_channels = []
            disp_channels = []

            this_chan_dofs = setup['chan_dofs']
            this_num_analised_channels = setup['num_analised_channels']
            #this_ref_channels = setup['ref_channels']
            #this_rov_channels = setup['roving_channels']

            this_accel_channels = setup['accel_channels']
            this_velo_channels = setup['velo_channels']
            this_disp_channels = setup['disp_channels']

            # chan dofs are now sorted by channel number
            this_chan_dofs.sort(key=lambda x: x[0])

            for channel in range(this_num_analised_channels):

                for chan_dof in this_chan_dofs:
                    if channel == chan_dof[0]:
                        node, az, elev = chan_dof[1:4]
                        if len(chan_dof) == 5:
                            name = chan_dof[4]
                        else:
                            name = ''
                        chan_dofs.append([node, az, elev, name])
                        break
                # if channel has not been assigned to a DOF
                else:
                    chan_dofs.append([None, 0, 0, ''])

                accel_channels.append(channel in this_accel_channels)
                velo_channels.append(channel in this_velo_channels)
                disp_channels.append(channel in this_disp_channels)

            merged_chan_dofs.append(chan_dofs)

            merged_accel_channels.append(accel_channels)
            merged_velo_channels.append(velo_channels)
            merged_disp_channels.append(disp_channels)

        # find dofs common to all setups
        # takes the dofs of the first setup as ssi_ref_dofs
        # loops over all setups and only keeps channels that are present in
        # the previous ssi_ref_dofs and the current setup

        # only ssi_ref_dofs can be used in the assembly of the hankel matrix
        # for mode shape rescaling the ref_dofs between the respective combination of two setups could be used
        # but the logic for this is not included here, therefore
        # only ssi_ref_dofs will be used for mode shape rescaling

        ssi_ref_dofs = copy.deepcopy(merged_chan_dofs[0])
        for chan_dofs in merged_chan_dofs[1:]:
            new_ref_dofs = []
            for node, az, elev, name in chan_dofs:
                if node is None:
                    continue
                for rnode, raz, relev, rname in ssi_ref_dofs:
                    if node == rnode and az == raz and elev == relev and name == rname:
                        new_ref_dofs.append((rnode, raz, relev, rname))
                        break
            ssi_ref_dofs = new_ref_dofs
            if len(ssi_ref_dofs) == 0:
                raise RuntimeError(
                    'Could not find any DOF that is common to all setups.')

        # find channels for each common dof
        #     add the channel number to rescale_ref_channels; these will
        #        be used to get the reference modal coordinates in mode shape rescaling
        #     add the index of the channel in setups' ref_channels to
        #        ssi_ref_channels; these will be used to assemble the Hankel
        #        matrix were each column is assembled from a single reference DOF (!)
        #        if reference channel orders have changed between setups, reordering
        #        is needed to achieve constistent columns of the Hankel matrix

        ssi_ref_channels = []
        #base_ssi_ref_channels = ssi_ref_channels[0]
        rescale_ref_channels = []
        for setup, chan_dofs in zip(setups, merged_chan_dofs):
            #prep_data = setup['prep_data']
            this_ssi_ref_channels = []
            this_rescale_ref_channels = []
            for rnode, raz, relev, rname in ssi_ref_dofs:
                index = None
                for channel, node, az, elev, name in setup['chan_dofs']:
                    if node == rnode and az == raz and elev == relev and name == rname:
                        #index = i
                        #channel = setup['chan_dofs'][index][0]

                        this_rescale_ref_channels.append(int(channel))

                        if channel not in setup['ref_channels']:
                            warnings.warn(
                                'Channel {} ({}) is common to multiple setups but not chosen as a reference channel.'.format(
                                    channel, name))
                        else:
                            this_ref_index = setup['ref_channels'].index(
                                channel)
                            this_ssi_ref_channels.append(this_ref_index)

                        break
                else:
                    raise RuntimeError(
                        'Oops! Something went wrong. This should not happen!')
            rescale_ref_channels.append(this_rescale_ref_channels)
            ssi_ref_channels.append(this_ssi_ref_channels)
            # print(this_ssi_ref_channels)

        # reorder chan_dofs, accel_channels, etc. of the first setup
        # s.t. references come first, followed by rovings
        # refs are ordered by ssi_ref_dofs order
        # rovs are ordered by ascending channel number of the underlying setup

        new_chan_dofs, new_accel_channels, new_velo_channels, new_disp_channels = [], [], [], []
        chan_dofs, accel_channels, velo_channels, disp_channels = merged_chan_dofs[
            0], merged_accel_channels[0], merged_velo_channels[0], merged_disp_channels[0]
        #print(chan_dofs,accel_channels, velo_channels, disp_channels )
        for rnode, raz, relev, rname in ssi_ref_dofs:
            for i, (node, az, elev, name) in enumerate(chan_dofs):
                if node == rnode and az == raz and elev == relev and name == rname:
                    new_chan_dofs.append(chan_dofs[i])
                    new_accel_channels.append(accel_channels[i])
                    new_velo_channels.append(velo_channels[i])
                    new_disp_channels.append(disp_channels[i])
                    break
            else:
                raise RuntimeError(
                    'This should not happen, as all ref_dofs were previously checked to be present in each setup.')
            del chan_dofs[i]
            del accel_channels[i]
            del velo_channels[i]
            del disp_channels[i]
        new_chan_dofs += chan_dofs
        new_accel_channels += accel_channels
        new_velo_channels += velo_channels
        new_disp_channels += disp_channels

        merged_chan_dofs[0], merged_accel_channels[0], merged_velo_channels[0], merged_disp_channels[
            0] = new_chan_dofs, new_accel_channels, new_velo_channels, new_disp_channels

        # delete channels of the reference dofs
        for chan_dofs, accel_channels, velo_channels, disp_channels in zip(
                merged_chan_dofs[1:], merged_accel_channels[1:], merged_velo_channels[1:], merged_disp_channels[1:]):
            #prep_data = setup['prep_data']
            for rnode, raz, relev, rname in ssi_ref_dofs:
                index = None
                for i, (node, az, elev, name) in enumerate(chan_dofs):
                    if node == rnode and az == raz and elev == relev and name == rname:
                        index = i
                        break
                else:
                    raise RuntimeError(
                        'This should not happen, as all ref_dofs were previously checked to be present in each setup.')
                # remove the channel_dof_assignment of the reference channels
                # for all setups
                del chan_dofs[index]
                del accel_channels[index]
                del velo_channels[index]
                del disp_channels[index]

        # flatten chan_dofs and add ascending channel numbers
        flattened = []
        channel = 0
        for sublist in merged_chan_dofs:
            for val in sublist:
                val.insert(0, channel)
                flattened.append(val)
                channel += 1
        merged_chan_dofs = flattened

        flattened = []
        channel = 0
        for sublist in merged_accel_channels:
            for val in sublist:
                if val:
                    flattened.append(channel)
                channel += 1
        merged_accel_channels = flattened

        flattened = []
        channel = 0
        for sublist in merged_velo_channels:
            for val in sublist:
                if val:
                    flattened.append(channel)
                channel += 1
        merged_velo_channels = flattened

        flattened = []
        channel = 0
        for sublist in merged_disp_channels:
            for val in sublist:
                if val:
                    flattened.append(channel)
                channel += 1
        merged_disp_channels = flattened

        num_analised_channels = sum(
            [setup['num_analised_channels'] for setup in setups])

        self.merged_accel_channels = merged_accel_channels
        self.merged_velo_channels = merged_velo_channels
        self.merged_disp_channels = merged_disp_channels

        self.ssi_ref_channels = ssi_ref_channels
        self.rescale_ref_channels = rescale_ref_channels
        self.merged_chan_dofs = merged_chan_dofs
        self.merged_num_channels = len(merged_chan_dofs)

        self.num_analised_channels = num_analised_channels
        self.start_time = min([stp['start_time'] for stp in setups])

        self.state[1] = True
        for i in range(2, 5):
            self.state[i] = False

        return ssi_ref_channels, merged_chan_dofs

    def build_merged_subspace_matrix(
            self,
            num_block_columns,
            num_block_rows=None):
        '''
        Builds a Block-HankelMatrix of Covariances with varying time lags

        ::

              <- num_block_columns*num_ref_channels-> _
            [     R_1      R_2      ...      R_i     ]^
            [     R_2      R_3      ...      R_2     ]num_block_rows*(num_num_ref_channels*num_setups)
            [     ...      ...      ...      ...     ]v
            [     R_i      ...      ...      R_2i-1  ]_

            R_1 =   [ R_1^1          ]
                    [ R_1^2          ]
                    [ ...            ]
                    [ R_1^num_setups ]

        '''

        assert isinstance(num_block_columns, int)

        if num_block_rows is None:
            num_block_rows = num_block_columns  # -10
        assert isinstance(num_block_rows, int)

        if not num_block_columns + num_block_columns + 1 <= self.n_lags:
            raise RuntimeError(
                'Correlation functions were pre-computed '
                'up to {} time lags, which is sufficient for assembling '
                'a Hankel-Matrix with up to {} x {} blocks. You requested '
                '{} x {} blocks'.format(
                    self.n_lags,
                    self.n_lags // 2 + 1,
                    self.n_lags // 2,
                    num_block_rows + 1,
                    num_block_columns))

        setups = self.setups

        print(
            'Assembling subspace matrix using pre-computed correlation'
            ' functions from {} setups with {} block-columns and {} '
            'block rows'.format(
                len(setups),
                num_block_columns,
                num_block_rows + 1))

        ssi_ref_channels = self.ssi_ref_channels

        num_analised_channels = self.num_analised_channels
        num_ref_channels = len(ssi_ref_channels[0])
        n_lags = self.n_lags

        subspace_matrix = np.zeros(
            ((num_block_rows + 1) * num_analised_channels,
             num_block_columns * num_ref_channels))
        end_row = None
        for block_row in range(num_block_rows + 1):
            sum_analised_channels = 0
            for this_ssi_ref_channels, setup in zip(ssi_ref_channels, setups):

                this_analised_channels = setup['num_analised_channels']
                this_corr_matrix = setup['corr_matrix']

                this_corr_matrix = this_corr_matrix[:,
                                                    this_ssi_ref_channels, :]
                # for each setup the order of the reference channels must be equal with respect to their DOFs
                #     we need a list of (local) ref_channels that corresponds to the ref_channels' DOFs of the first setup
                #     ssi_ref_channels = [[ref_DOF_1 -> index of ref_channel setup A, ref_DFO_2 -> index of ref_channel setup A, ...],
                #                         [ref_DOF_1 -> index of ref_channel setup B, ref_DFO_2 -> index of ref_channel setup B, ...],
                #                         ...]
                # this_corr_matrix = this_corr_matrix[:,this_ssi_ref_channels]
                # # just reorders the columns corresponding to the ref_channels
                # of current setup

                this_corr_matrix = this_corr_matrix.reshape(
                    (this_analised_channels, num_ref_channels * n_lags), order='F')
                this_block_column = this_corr_matrix[:, block_row * num_ref_channels:(
                    num_block_columns + block_row) * num_ref_channels]

                begin_row = block_row * num_analised_channels + sum_analised_channels
                if end_row is not None:
                    assert begin_row >= end_row
                end_row = begin_row + this_analised_channels

                subspace_matrix[begin_row:end_row, :] = this_block_column

                sum_analised_channels += this_analised_channels

            # block_row    0                                            1
            # setup 0: zeile 0*this_analised_channels ... 1*this_analised_channels,   3*this_analised_channels ... 4*this_analised_channels
            # setup 1: zeile 1*this_analised_channels ... 2*this_analised_channels,   4*this_analised_channels ... 5*this_analised_channels
            # setup 2: zeile 2*this_analised_channels ... 3*this_analised_channels,   5*this_analised_channels ... 6*this_analised_channels
            # (bc*num_setups+setup)*num_ref_channels
        assert (subspace_matrix != 0).all()
        self.subspace_matrix = subspace_matrix
        self.num_block_rows = num_block_rows
        self.num_block_columns = num_block_columns

        self.state[2] = True
        for i in range(3, 5):
            self.state[i] = False

    def compute_state_matrices(self, max_model_order=None):

        if max_model_order is not None:
            assert isinstance(max_model_order, int)

        subspace_matrix = self.subspace_matrix

        if max_model_order is None:
            max_model_order = min(subspace_matrix.shape)
        else:
            max_model_order = min(max_model_order, *subspace_matrix.shape)

        print('Computing state matrices with pinv-based method, with maximum model order {}...'.format(max_model_order))

        U, S, _ = scipy.linalg.svd(subspace_matrix, 1)

        # choose highest possible model order

        U = U[:, :max_model_order]
        #V_T = V_T[:max_model_order,:]

        self.U = U
        self.S = S
        #self.V_T = V_T

        self.max_model_order = max_model_order

        self.state[3] = True
        for i in range(4, 5):
            self.state[i] = False

    def compute_modal_params(self, max_model_order=None):

        if max_model_order is None:
            max_model_order = self.max_model_order

        num_block_rows = self.num_block_rows
        num_analised_channels = self.num_analised_channels
        merged_num_channels = self.merged_num_channels
        sampling_rate = self.sampling_rate

        U = self.U
        S = self.S
        S_2 = np.diag(np.power(S[:max_model_order], -0.5))

        O = np.dot(U, S_2)

        print('Computing modal parameters...')

        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros(
            (merged_num_channels,
             max_model_order,
             max_model_order),
            dtype=complex)
        eigenvalues = np.zeros(
            (max_model_order, max_model_order), dtype=complex)

        printsteps = list(np.linspace(0, max_model_order, 100, dtype=int))
        for order in range(1, max_model_order):
            while order in printsteps:
                del printsteps[0]
                print('.', end='', flush=True)
            # print('\n\n{}\n\n'.format(order))

            On_up = O[:num_analised_channels * num_block_rows, :order]
            On_down = O[num_analised_channels:num_analised_channels *
                        (num_block_rows + 1), :order]

            state_matrix = np.dot(np.linalg.pinv(On_up), On_down)

            C = O[:num_analised_channels, :order]

            eigval, eigvec_r = np.linalg.eig(state_matrix)

            conj_indices = self.remove_conjugates(
                eigval, eigvec_r, inds_only=True)

            for i, ind in enumerate(conj_indices):

                lambda_i = eigval[ind]

                ident = eigval == lambda_i.conj()
                ident[ind] = 1
                ident = np.diag(ident)

                a_i = np.abs(np.arctan2(np.imag(lambda_i), np.real(lambda_i)))
                b_i = np.log(np.abs(lambda_i))
                freq_i = np.sqrt(a_i**2 + b_i**2) * sampling_rate / 2 / np.pi
                damping_i = 100 * np.abs(b_i) / np.sqrt(a_i**2 + b_i**2)
                mode_shape_i = np.dot(C, eigvec_r[:, ind])
                mode_shape_i = np.array(mode_shape_i, dtype=complex)

                mode_shape_i = self.rescale_by_references(mode_shape_i)
                mode_shape_i = self.integrate_quantities(
                    mode_shape_i,
                    self.merged_accel_channels,
                    self.merged_velo_channels,
                    freq_i * 2 * np.pi)

                k = np.argmax(np.abs(mode_shape_i))
                s_ik = mode_shape_i[k]
                alpha_ik = np.angle(s_ik)
                e_k = np.zeros((num_analised_channels, 1))
                e_k[k, 0] = 1
                mode_shape_i *= np.exp(-1j * alpha_ik)

                modal_frequencies[order, i] = freq_i
                modal_damping[order, i] = damping_i
                mode_shapes[:, i, order] = mode_shape_i
                eigenvalues[order, i] = lambda_i

        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues

        print('.', end='\n', flush=True)

        self.state[4] = True

    def rescale_by_references(self, mode_shape):
        '''
        This is PoGer Rescaling

         
         * extracts each setup's reference and roving parts of the modeshape
         * compute rescaling factor from all setup's reference channels using a least-squares approach 
         * rescales each setup's roving channels and assembles final modeshape vector

        reference channel_pairs and final channel-dof-assignments have been determined by function pair_channels
        note: reference channels for SSI need not necessarily be reference channels for rescaling and vice versa

        :math:`S_\\phi \\times \\alpha = [n \\times 1, 0 .. 0]`

        :math:`\\phi^{ref}_i` : Reference-sensor part of modeshape estimated from setup :math:`i = 0 .. n`
        :matH:`j_{max} = \\operatorname{argmax}(\\Pi_i |\\phi^{ref}_i|)` : maximal modal component in all setups → will be approximately scaled to 1, must belong to the same sensor in each setup

        .. math::

            S_\\phi =  \\begin{bmatrix}
            \\phi^{ref}_{0,j_{max}}&  \\phi^{ref}_{1,j_{max}}& ..&            ..&               \\phi^{ref}_{n,j_{max}} \\\\
            \\phi^{ref}_0&            -\\phi^{ref}_1&          0&             ..&               0                      \\\\
            \\phi^{ref}_0&            0&                      -\\phi^{ref}_2& ..&               0                      \\\\
            .                        &.                       &.             & .             & .                      \\\\
            .                        &.                       &.             & .             & .                      \\\\
            \\phi^{ref}_0&            0&                      0&             ..&               -\\phi^{ref}_n          \\\\
            0&                       \\phi^{ref}_1&           -\\phi^{ref}_2& ..&               0                      \\\\
            .                        &.                      & .             & .             & .                      \\\\
            .                        &.                      & .             & .             & .                      \\\\
            0&                       \\phi^{ref}_1&           0&             ..&               -\\phi^{ref}_n          \\\\
            .                        &.                      & .             & .             & .                      \\\\
            .                        &.                      & .             & .             & .                      \\\\
            0&                       0&                      \\phi^{ref}_2&  ..&               -\\phi^{ref}_n          \\\\
            .                        &.                      & .             & .             & .                      \\\\
            .                        &.                      & .             & .            &  .                      \\\\
            0&                       0&                      0&             \\phi^{ref}_{n-1}& -\\phi^{ref}_n
            \\end{bmatrix}

        if references are the same in all setups

        dimensions :math:`= 1 + (n_{setups} ! )* n_{ref_{channels}} \\times n_{setups}`

        not quite exact, since different setups may share different references

        → list based assembly of the :math:`S_\\phi` matrix

        '''

        new_mode_shape = np.zeros((self.merged_num_channels), dtype=complex)

        start_row_scaled = 0
        end_row_scaled = self.setups[0]['num_analised_channels']

        row_unscaled = self.setups[0]['num_analised_channels']

        new_mode_shape[start_row_scaled:end_row_scaled] = mode_shape[start_row_scaled:end_row_scaled]

#         for setup_num,setup in enumerate(self.setups):
#             if setup_num == 0: continue
#             #ssi_ref_channels is ref_channels with respect to setup not to merged mode shape
#
#             base_refs = self.rescale_ref_channels[0]
#
#             this_refs = self.rescale_ref_channels[setup_num]
#             this_all = range(setup['num_analised_channels'])#setup['roving_channels']+setup['ref_channels']
#             this_rovs = list(set(this_all).difference(this_refs))
#
#             this_refs = [int(ref+row_unscaled) for ref in this_refs]
#             this_rovs = [rov+row_unscaled for rov in this_rovs]
#
#             mode_refs_base = mode_shape[base_refs]
#             mode_refs_this = mode_shape[this_refs]
#             mode_refs_this_conj = mode_refs_this.conj()
#             mode_rovs_this = mode_shape[this_rovs]
#
#             numer = np.inner(mode_refs_this_conj, mode_refs_base )
#             denom = np.inner(mode_refs_this_conj, mode_refs_this )
#             scale_fact=numer/denom
#
#             start_row_scaled = end_row_scaled
#             end_row_scaled += len(this_rovs)
#             #print(mode_refs_base, mode_refs_this, scale_fact)
#             new_mode_shape[start_row_scaled:end_row_scaled] = scale_fact * mode_rovs_this
#
#             row_unscaled += setup['num_analised_channels']
#         return new_mode_shape

        num_setups = len(self.setups)

        S_phi = []

        # assemble the first line
        all_ref_modes = []
        for setup_num, setup in enumerate(self.setups):
            this_refs = self.rescale_ref_channels[setup_num]
            mode_refs_this = mode_shape[this_refs]
            all_ref_modes.append(mode_refs_this)
        all_ref_modes = np.array(all_ref_modes).T
        max_ind = np.argmax(np.product(np.abs(all_ref_modes), axis=1))
        S_phi.append(all_ref_modes[max_ind:max_ind + 1, :])

        # assemble S_phi
        row_unscaled_1 = 0
        for setup_num_1, setup_1 in enumerate(self.setups):
            row_unscaled_2 = 0
            for setup_num_2, setup_2 in enumerate(self.setups):
                if setup_num_2 <= setup_num_1:

                    row_unscaled_2 += setup_2['num_analised_channels']
                    continue
                # ssi_ref_channels is ref_channels with respect to setup not to
                # merged mode shape

                base_refs = self.rescale_ref_channels[setup_num_1]

                this_refs = self.rescale_ref_channels[setup_num_2]
                # this_all = range(setup['num_analised_channels'])#setup['roving_channels']+setup['ref_channels']
                #this_rovs = list(set(this_all).difference(this_refs))

                base_refs = [int(ref + row_unscaled_1) for ref in base_refs]
                this_refs = [int(ref + row_unscaled_2) for ref in this_refs]
                #this_rovs = [rov+row_unscaled for rov in this_rovs]

                mode_refs_base = mode_shape[base_refs]
                mode_refs_this = mode_shape[this_refs]

                this_S_phi = np.zeros(
                    (len(base_refs), num_setups), dtype=complex)
                this_S_phi[:, setup_num_1] = mode_refs_base
                this_S_phi[:, setup_num_2] = -1 * mode_refs_this

                S_phi.append(this_S_phi)
                row_unscaled_2 += setup_2['num_analised_channels']
            row_unscaled_1 += setup_1['num_analised_channels']

        S_phi = np.vstack(S_phi)

        # compute scaling factors
        rhs = np.zeros(S_phi.shape[0], dtype=complex)
        rhs[0] = num_setups + 0j
        # rhs[0]=1+0j
        alpha = np.linalg.pinv(S_phi).dot(rhs)

        end_row_scaled = 0  # self.setups[0]['num_analised_channels']
        row_unscaled = 0  # self.setups[0]['num_analised_channels']
        for setup_num, setup in enumerate(self.setups):

            this_refs = self.rescale_ref_channels[setup_num]
            # setup['roving_channels']+setup['ref_channels']
            this_all = range(setup['num_analised_channels'])
            this_rovs = list(set(this_all).difference(this_refs))

            this_rovs = [rov + row_unscaled for rov in this_rovs]

            mode_rovs_this = mode_shape[this_rovs]

            scale_fact = alpha[setup_num]

            if 0:
                base_refs = self.rescale_ref_channels[0]

                this_refs = [int(ref + row_unscaled) for ref in this_refs]
                #this_rovs = [rov+row_unscaled for rov in this_rovs]

                mode_refs_base = mode_shape[base_refs]
                mode_refs_this = mode_shape[this_refs]
                mode_refs_this_conj = mode_refs_this.conj()

                numer = np.inner(mode_refs_this_conj, mode_refs_base)
                denom = np.inner(mode_refs_this_conj, mode_refs_this)
                scale_fact = numer / denom

                #print( scale_fact, scale_fact_old)

            if setup_num == 0:
                mode_refs_this = mode_shape[this_refs]
                start_row_scaled = end_row_scaled
                end_row_scaled += len(this_refs)
                new_mode_shape[start_row_scaled:end_row_scaled] = scale_fact * \
                    mode_refs_this

            start_row_scaled = end_row_scaled
            end_row_scaled += len(this_rovs)

            new_mode_shape[start_row_scaled:end_row_scaled] = scale_fact * \
                mode_rovs_this

            row_unscaled += setup['num_analised_channels']
        return new_mode_shape

    def save_state(self, fname):

        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        #             0             1                2              3           4
        # self.state= [Setups Added, Channels Paired, Subspace Mat., State
        # Mat., Modal Par.]
        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        # out_dict['self.start_times']=self.start_times

        if self.state[0]:  # add_setup
            out_dict['self.setups'] = self.setups
            out_dict['self.sampling_rate'] = self.sampling_rate
            out_dict['self.num_ref_channels'] = self.num_ref_channels
            out_dict['self.n_lags'] = self.n_lags
        if self.state[1]:  # pair_channels
            out_dict['self.ssi_ref_channels'] = self.ssi_ref_channels
            out_dict['self.rescale_ref_channels'] = self.rescale_ref_channels
            out_dict['self.merged_chan_dofs'] = self.merged_chan_dofs
            out_dict['self.merged_accel_channels'] = self.merged_accel_channels
            out_dict['self.merged_velo_channels'] = self.merged_velo_channels
            out_dict['self.merged_disp_channels'] = self.merged_disp_channels
            out_dict['self.merged_num_channels'] = self.merged_num_channels
            out_dict['self.num_analised_channels'] = self.num_analised_channels
            out_dict['self.start_time'] = self.start_time
        if self.state[2]:  # build_merged_subspace_matrix
            out_dict['self.subspace_matrix'] = self.subspace_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[3]:  # compute_state_matrices
            out_dict['self.U'] = self.U
            out_dict['self.S'] = self.S
            out_dict['self.max_model_order'] = self.max_model_order
        if self.state[4]:  # compute_modal_params
            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.mode_shapes'] = self.mode_shapes

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, ):
        print('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        #             0             1                2              3           4
        # self.state= [Setups Added, Channels Paired, Subspace Mat., State
        # Mat., Modal Par.]
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['Setups added',
                                                    'Channels paired, channel-DOF assignments generated',
                                                    'Subspace matrix built',
                                                    'State matrices computed',
                                                    'Modal parameters computed',
                                                    ]):
            if this_state:
                print(state_string)

        #assert isinstance(prep_data, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())

        ssi_object = cls()
        ssi_object.setup_name = setup_name

        ssi_object.state = state
        # debug_here
        if state[0]:  # add_setup
            ssi_object.setups = list(in_dict['self.setups'])
            ssi_object.sampling_rate = in_dict['self.sampling_rate'].item()
            ssi_object.num_ref_channels = in_dict['self.num_ref_channels'].item(
            )
            ssi_object.n_lags = in_dict['self.n_lags'].item()
        if state[1]:  # pair_channels
            ssi_object.ssi_ref_channels = [
                list(l) for l in in_dict['self.ssi_ref_channels']]
            ssi_object.rescale_ref_channels = [
                list(l) for l in in_dict['self.rescale_ref_channels']]
            ssi_object.merged_chan_dofs = [[int(float(cd[0])), str(cd[1]), float(cd[2]), float(
                cd[3]), str(cd[4] if len(cd) == 5 else '')] for cd in in_dict['self.merged_chan_dofs']]
            ssi_object.merged_accel_channels = list(
                in_dict['self.merged_accel_channels'])
            ssi_object.merged_velo_channels = list(
                in_dict['self.merged_velo_channels'])
            ssi_object.merged_disp_channels = list(
                in_dict['self.merged_disp_channels'])
            ssi_object.merged_num_channels = in_dict['self.merged_num_channels'].item(
            )
            ssi_object.num_analised_channels = in_dict['self.num_analised_channels'].item(
            )
            ssi_object.start_time = in_dict['self.start_time'].item()
        if state[2]:  # build_merged_subspace_matrix
            ssi_object.subspace_matrix = in_dict['self.subspace_matrix']
            ssi_object.num_block_columns = in_dict['self.num_block_columns'].item(
            )
            ssi_object.num_block_rows = in_dict['self.num_block_rows'].item()
        if state[3]:  # compute_state_matrices
            ssi_object.U = in_dict['self.U']
            ssi_object.S = in_dict['self.S']
            ssi_object.max_model_order = in_dict['self.max_model_order'].item()
        if state[4]:  # compute_modal_params
            ssi_object.eigenvalues = in_dict['self.eigenvalues']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']

        return ssi_object


if __name__ == '__main__':
    pass

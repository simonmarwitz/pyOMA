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

Written by Volkmar Zabel 2016, refactored by Simon Marwitz 2021

.. TODO::
     * Move the computation of half-spectra to PreProcessData and change this
       class accordingly
     * Proper documentation of the code
     * Test functions should be added to the test package
     * Algorithm seems broken and a lot of overhead is reimplemented, that
       exists in standard libraries

'''

import numpy as np
import os
import scipy.signal
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase


class PLSCF(ModalBase):

    def __init__(self, *args, **kwargs):
        '''
        channel definition: channels start at 0
        '''
        
        logging.warning('This implementation of the PLSCF algorithm seems broken, code must be re-verified and a working example added to the "tests" package.')
        super().__init__(*args, **kwargs)
        #             0             1
        # self.state= [Half_spectra, Modal Par.
        self.state = [False, False]

        self.begin_frequency = None
        self.end_frequency = None
        self.nperseg = None
        self.factor_a = None
        self.selected_omega_vector = None
        self.num_omega = None
        self.spectrum_tensor = None

        self.max_model_order = None

    @classmethod
    def init_from_config(cls, conf_file, prep_data):
        assert os.path.exists(conf_file)
        assert isinstance(prep_data, PreProcessSignals)

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Begin Frequency:'
            begin_frequency = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'End Frequency:'
            end_frequency = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Samples per time segment:'
            nperseg = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
            max_model_order = int(f. __next__().strip('\n'))

        pLSCF_object = cls(prep_data)
        pLSCF_object.build_half_spectra(
            begin_frequency, end_frequency, nperseg)
        pLSCF_object.compute_modal_params(max_model_order)

        return pLSCF_object

    def build_half_spectra(self, begin_frequency, end_frequency, nperseg):
        '''
        Constructs a half spectrum matrix
        Builds a 3D tensor with cross spectral densities:
        Dimensions: number_of_all_channels x number_of_references x number_of_freq_lines

        .. TODO:
         * move this functionality to the PreProcessData class

        '''

        print('Constructing half-spectrum matrix ... ')
        assert isinstance(begin_frequency, float)
        assert isinstance(end_frequency, float)
        assert isinstance(nperseg, int)

        self.begin_frequency = begin_frequency
        self.end_frequency = end_frequency
        self.nperseg = nperseg
        total_time_steps = self.prep_data.total_time_steps
        ref_channels = sorted(self.prep_data.ref_channels)
        roving_channels = [
            i for i in range(
                self.prep_data.num_analised_channels) if i not in ref_channels]

        measurement = self.prep_data.signals
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels = self.prep_data.num_ref_channels

        sampling_rate = self.prep_data.sampling_rate

        # Extract reference time series for half spectra

        all_channels = ref_channels + roving_channels
        all_channels.sort()
        # print(all_channels)

        refs = (measurement[:, ref_channels])

        begin_omega = begin_frequency * 2 * np.pi
        end_omega = end_frequency * 2 * np.pi
        factor_a = 0

        for ii in range(num_ref_channels):

            this_ref = refs[:, ii]

            for jj in range(num_analised_channels):

                this_response = measurement[:, jj]

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

                # spectra from correlation functions of independent sections
                # ####
                '''
                '''

                # num_sections = 200  # number of independent sections for averaging
                #blocklength = int(len(this_ref) * 1/num_sections)
                #blocklength = int(len(this_ref) * 0.025)
                #print('blocklength = ', blocklength)

                tmp_freq, tmp_cross_spec = scipy.signal.csd(this_ref, this_response,
                                                            fs=sampling_rate, window='boxcar', nperseg=self.nperseg,
                                                            noverlap=None, detrend='constant', return_onesided=True,
                                                            scaling='spectrum')

                #rint('tmp_freq = ', tmp_freq)

                R_xy = np.fft.irfft(tmp_cross_spec)
                R_xy = R_xy[0:int(len(R_xy) / 2)]

                # Diagramm mit Korrelationsfunktion

                '''
                #envelope = amplitude * np.exp(exponent * corr_time)

                fig2 = plt.figure(figsize = [15,5])
                ax = fig2.add_subplot(1,1,1)
                ax.plot(corr_time, R_yy, lw=1, visible=True)
                ax.plot(corr_time, envelope, 'g-', lw=2, visible=True)
                ax.grid()
                #ax.scatter(peak_times, peaks, s = 100, c='r')
                '''

                (R_xy, factor_a) = self.Exp_Win(R_xy, 0.001)
                #print('factor_a = ', factor_a)

                frequency_vector, this_half_spec = scipy.signal.welch(
                    R_xy, fs=sampling_rate, window='boxcar', nperseg=len(R_xy), noverlap=None, return_onesided=True)
                omega_vector = frequency_vector * 2 * np.pi

                #fig = plt.figure(figsize = [10,5])

                #ax1 = fig.add_subplot(2,1,1)
                #ax2 = fig.add_subplot(2,1,2)

                #ax1.plot( R_xy, 'b-', label='$R_xy_inv_FFT$')

                #ax2.plot( frequency_vector, abs(this_half_spec), 'b-')
                # plt.show

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

                    selected_omega_vector = omega_vector[index_begin_omega: index_end_omega + 1]
                    num_omega = len(selected_omega_vector)

                    spectrum_tensor = np.zeros(
                        (num_analised_channels,
                         num_ref_channels,
                         len(selected_omega_vector)),
                        dtype=complex)

                spectrum_tensor[jj, ii,
                                :] = this_half_spec[index_begin_omega: index_end_omega + 1]

        self.selected_omega_vector = selected_omega_vector
        self.num_omega = num_omega
        self.spectrum_tensor = spectrum_tensor
        self.factor_a = factor_a

        self.state[0] = True

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
        a = np.log(fin_val) / (n - 1)

        win = np.arange(n)
        win = win * a
        win = np.exp(win)

        x_win = x * win

        return x_win, a

    def compute_modal_params(self, max_model_order):

        if max_model_order is not None:
            assert isinstance(max_model_order, int)

        assert self.state[0]

        self.max_model_order = max_model_order
        factor_a = self.factor_a

        print('Computing modal parameters...')

        #ref_channels = sorted(self.prep_data.ref_channels)
        #roving_channels = self.prep_data.roving_channels
        #signals = self.prep_data.signals
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels = self.prep_data.num_ref_channels
        selected_omega_vector = self.selected_omega_vector
        num_omega = self.num_omega
        spectrum_tensor = self.spectrum_tensor

        sampling_rate = self.prep_data.sampling_rate
        Delta_t = 1 / sampling_rate

        # Compute the modal solutions for all model orders

        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros(
            (num_analised_channels,
             max_model_order,
             max_model_order),
            dtype=complex)
        eigenvalues = np.zeros(
            (max_model_order, max_model_order), dtype=complex)

#         for this_model_order in range(max_model_order+1):
#
#             # minimal model order should be 2 !!!
#
#             if this_model_order >> 1:
#                 print("this_model_order: ", this_model_order)
        printsteps = list(np.linspace(0, max_model_order, 100, dtype=int))
        for this_model_order in range(1, max_model_order + 1):
            while this_model_order in printsteps:
                del printsteps[0]
                print('.', end='', flush=True)
                # Create matrices X_0 and Y_0

            X_o = np.zeros((num_omega, (this_model_order + 1)), dtype=complex)

            for jj in range(this_model_order + 1):
                X_o[:, jj] = selected_omega_vector * jj * Delta_t * 1j

            X_o = np.exp(X_o)

            for n_o in range(num_analised_channels):
                Y_o = np.zeros(
                    (num_omega, ((this_model_order + 1) * num_ref_channels)), dtype=complex)

                for kk in range(num_omega):
                    this_Syy = spectrum_tensor[n_o, :, kk]

                    for ll in range(this_model_order + 1):
                        Y_o[kk, (ll * num_ref_channels):((ll + 1) * \
                                 num_ref_channels)] = X_o[kk, ll] * this_Syy.T

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
                    R_o_inv_tensor = np.zeros(
                        (R_o_rows, R_o_cols, num_analised_channels))
                    R_o_inv_tensor[:, :, n_o] = R_o_inv
                    S_o_rows = S_o.shape[0]
                    S_o_cols = S_o.shape[1]
                    S_o_tensor = np.zeros(
                        (S_o_rows, S_o_cols, num_analised_channels))
                    S_o_tensor[:, :, n_o] = S_o

                else:

                    M = M + 2 * (T_o - (np.dot(np.dot(S_o.T, R_o_inv), S_o)))
                    R_o_inv_tensor[:, :, n_o] = R_o_inv
                    S_o_tensor[:, :, n_o] = S_o

            # Compute alpha, beta

            M_ba = M[num_ref_channels:, : num_ref_channels]
            M_bb = M[num_ref_channels:, num_ref_channels:]
            alpha_b = -np.dot(np.linalg.inv(M_bb), M_ba)
            alpha = np.eye(num_ref_channels)
            alpha = np.concatenate((alpha, alpha_b), axis=0)

            for n_o in range(num_analised_channels):
                R_o_inv = R_o_inv_tensor[:, :, n_o]
                S_o = S_o_tensor[:, :, n_o]
                beta_o = - np.dot(R_o_inv, (np.dot(S_o, alpha)))

                if n_o == 0:
                    beta_o_rows = beta_o.shape[0]
                    beta_o_cols = beta_o.shape[1]
                    beta_o_tensor = np.zeros(
                        (beta_o_rows, beta_o_cols, num_analised_channels))
                    beta_o_tensor[:, :, n_o] = beta_o

                else:
                    beta_o_tensor[:, :, n_o] = beta_o

            # Create matrices A_c and C_c

            A_p = alpha[(this_model_order * num_ref_channels)
                         :((this_model_order + 1) * num_ref_channels), :]
            A_p_inv = np.linalg.inv(A_p)
            B_p = beta_o_tensor[this_model_order, :, :]
            B_p = np.transpose(B_p)
            size_A_c = this_model_order * num_ref_channels
            A_c = np.zeros((size_A_c, size_A_c))
            C_c = np.zeros((B_p.shape[0], size_A_c))

            for p_i in range(this_model_order):
                A_p_i = alpha[((this_model_order - p_i - 1) * num_ref_channels):((this_model_order - p_i) * num_ref_channels), :]
                this_A_c_block = - (np.dot(A_p_inv, A_p_i))
                A_c[0:num_ref_channels, (p_i * num_ref_channels)                    :((p_i + 1) * num_ref_channels)] = this_A_c_block
                B_p_i = beta_o_tensor[(this_model_order - p_i - 1), :, :]
                B_p_i = np.transpose(B_p_i)
                this_C_c_block = B_p_i - (np.dot(B_p, this_A_c_block))
                C_c[:, (p_i * num_ref_channels):((p_i + 1)
                                                 * num_ref_channels)] = this_C_c_block

            A_c_rest = np.eye((this_model_order - 1) * num_ref_channels)
            A_c[num_ref_channels:,
                0:((this_model_order - 1) * num_ref_channels)] = A_c_rest

            # Compute modal parameters from matrices A_c and C_c

            lambda_k = np.array([], dtype=complex)

            eigenvalues_paired, eigenvectors_paired = np.linalg.eig(A_c)
            eigenvectors_single = []
            eigenvalues_single = []

            eigenvalues_single, eigenvectors_single = \
                self.remove_conjugates(eigenvalues_paired, eigenvectors_paired)
            eigenvectors_single = np.array(eigenvectors_single)
            eigenvalues_single = np.array(eigenvalues_single)

            #print('dim. of eigenvectors_paired = ', eigenvectors_paired.shape)
            #print('dim. of eigenvectors_single = ', eigenvectors_single.shape)
            #print('dim. of C_c = ', C_c.shape)

            current_frequencies = np.zeros((1, max_model_order))
            current_damping = np.zeros((1, max_model_order))
            current_mode_shapes = np.zeros(
                (num_analised_channels, max_model_order), dtype=complex)

            for jj in range(len(eigenvalues_single)):
                k = eigenvalues_single[jj]

                lambda_k = np.log(complex(k)) * sampling_rate
                freq_j = np.abs(lambda_k) / (2 * np.pi)

                # damping without correction if no exponential window was
                # applied
                '''
                damping_j = np.real(lambda_k)/np.abs(lambda_k) * (-100)
                '''
                # damping with correction if exponential window was applied to
                # corr. fct.

                damping_j = (np.real(lambda_k) / np.abs(lambda_k) -
                             factor_a * (sampling_rate) / (freq_j * 2 * np.pi)) * (-100)
                #damping_j = (np.real(lambda_k)/np.abs(lambda_k) + factor_a * (freq_j * 2*np.pi)) * (-100)
                #damping_j = (np.real(lambda_k)/np.abs(lambda_k) - factor_a ) * (-100)

                mode_shapes_j = np.dot(C_c[:, :], eigenvectors_single[:, jj])
                mode_shapes_j = mode_shapes_j.reshape(
                    (num_analised_channels, 1))
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                #mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))

                current_frequencies[0:1, jj:jj + 1] = freq_j
                current_damping[0:1, jj:jj + 1] = damping_j
                current_mode_shapes[:, jj:jj + 1] = mode_shapes_j

            modal_frequencies[(this_model_order - 1), :] = current_frequencies
            modal_damping[(this_model_order - 1), :] = current_damping
            eigenvalues[(this_model_order - 1),
                        :len(eigenvalues_single)] = eigenvalues_single
            mode_shapes[:, :, (this_model_order - 1)] = current_mode_shapes
        print('.', end='\n', flush=True)

        self.eigenvalues = eigenvalues
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes

        self.state[1] = True

    def save_state(self, fname):

        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        #             0             1
        # self.state= [Half_spectra, Modal Par.
        self.state = [False, False]
        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time
        # out_dict['self.prep_data']=self.prep_data
        if self.state[0]:  # spectral tensor
            out_dict['self.begin_frequency'] = self.begin_frequency
            out_dict['self.end_frequency'] = self.end_frequency
            out_dict['self.nperseg'] = self.nperseg
            out_dict['self.selected_omega_vector'] = self.selected_omega_vector
            out_dict['self.num_omega'] = self.num_omega
            out_dict['self.spectrum_tensor'] = self.spectrum_tensor
        if self.state[1]:  # modal params
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(
                state, ['Sprectral Tensor Built', 'Modal Parameters Computed', ]):
            if this_state:
                print(state_string)

        assert isinstance(prep_data, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())
        start_time = in_dict['self.start_time'].item()
        assert setup_name == prep_data.setup_name
        start_time = prep_data.start_time

        assert start_time == prep_data.start_time
        #prep_data = in_dict['self.prep_data'].item()
        pLSCF_object = cls(prep_data)
        pLSCF_object.state = state
        if state[0]:  # spectral tensor
            pLSCF_object.begin_frequency = in_dict['self.begin_frequency']
            pLSCF_object.end_frequency = int(in_dict['self.end_frequency'])
            pLSCF_object.nperseg = int(in_dict['self.nperseg'])
            pLSCF_object.selected_omega_vector = in_dict['self.selected_omega_vector']
            pLSCF_object.num_omega = in_dict['self.num_omega']
            pLSCF_object.spectrum_tensor = in_dict['self.spectrum_tensor']
        if state[1]:  # modal params
            pLSCF_object.max_model_order = int(in_dict['self.max_model_order'])
            pLSCF_object.modal_frequencies = in_dict['self.modal_frequencies']
            pLSCF_object.modal_damping = in_dict['self.modal_damping']
            pLSCF_object.mode_shapes = in_dict['self.mode_shapes']

        return pLSCF_object


def main():
    pass


if __name__ == '__main__':
    main()

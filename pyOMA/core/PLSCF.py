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

Written by Volkmar Zabel 2016, 
refactored by Simon Marwitz 2021, 
improved, corrected and refactored by Simon Marwitz 2024

.. TODO::
     * Test functions should be added to the test package

'''

import numpy as np
import os
import scipy.signal
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase
from .Helpers import validate_array


class PLSCF(ModalBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = [False, False]

        self.begin_frequency = None
        self.end_frequency = None
        self.nperseg = None
        self.factor_a = None
        self.selected_omega_vector = None
        self.pos_half_spectra = None


    @classmethod
    def init_from_config(cls, conf_file, prep_signals):
        assert os.path.exists(conf_file)
        assert isinstance(prep_signals, PreProcessSignals)

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Begin Frequency:'
            begin_frequency = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'End Frequency:'
            end_frequency = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Samples per time segment:'
            nperseg = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
            max_model_order = int(f. __next__().strip('\n'))

        pLSCF_object = cls(prep_signals)
        pLSCF_object.build_half_spectra(nperseg, begin_frequency, end_frequency)
        pLSCF_object.compute_modal_params(max_model_order)

        return pLSCF_object

    def build_half_spectra(self, nperseg=None, begin_frequency=None, end_frequency=None, window_decay=0.001):
        '''
        Extracts an array of positive half spectra between begin_frequency 
        and end_frequency from a spectrum of nperseg frequency lines. If 
        begin_frequency > 0.0 or end_frequency<nyquist freqeuncy, the resulting
        array has less than nperseg lines.
        
        Positive power spectra are constructed from positive correlation functions,
        that are windowed by an exponential window and transformed to frequency
        domain by and (R)FFT.
        Correlation functions are computed in prep_signals by either 
        Welch's or Blackman-Tukey's method, though, Welch's method is not 
        recommmended, because the artificial  damping introduced by windowing 
        can not be corrected.
        
        See: Cauberghe-2004-Applied Frequency-Domain System ... : Sections 3.4ff
        
        Note: The previous implementation contained severe mistakes in the computation
        of positive power spectra, e.g. doubled squaring of spectral values, lazy handling
        of array dimensions and therefore effectively only a quarter of nperseg being used,
        numerical inefficiencies
        
        Parameters
        ----------
            nperseg: integer, optional
                Number of frequency lines to consider
            
            begin_frequency, end_frequency: float, optional
                Frequency range to restrict the identified system.
            
            window_decay: float, (0,1)
                Final value of the exponential window, that is applied to the 
                correlation functions.
                
        '''

        logger.info('Constructing half-spectrum matrix ... ')
        if begin_frequency is None:
            begin_frequency = 0.0
        if isinstance(begin_frequency, int): 
            begin_frequency = float(begin_frequency)
        assert isinstance(begin_frequency, float)
        if end_frequency is None:
            end_frequency = self.prep_signals.sampling_rate / 2
        if isinstance(end_frequency, int): 
            end_frequency = float(end_frequency)
        assert isinstance(end_frequency, float)
        if nperseg is None:
            nperseg = self.prep_signals.n_lines
        if nperseg is None:
            raise RuntimeError('Argument nperseg or precomputed spectra must be provided.')
        assert isinstance(nperseg, int)

        self.begin_frequency = begin_frequency
        self.end_frequency = end_frequency
        self.nperseg = nperseg

        sampling_rate = self.prep_signals.sampling_rate
        
        tau = -(nperseg) / np.log(window_decay)
        
        if self.prep_signals._last_meth == 'welch':
            logger.warning("The selected spectral estimation method (Welch) is not recommended (exponential window can not be applied to correlation function).")
        
        correlation_matrix = self.prep_signals.correlation(nperseg, window='boxcar')
        
        win = scipy.signal.windows.get_window(('exponential', 0, tau), nperseg, fftbins=True)
        
        psd_matrix = np.fft.rfft(correlation_matrix * win)

        freqs = np.fft.rfftfreq(nperseg, 1/sampling_rate)
        
        freq_inds = (freqs>begin_frequency) & (freqs<end_frequency)
        
        selected_omega_vector = freqs[freq_inds] * 2 * np.pi
        
        spectrum_tensor = psd_matrix[..., (freqs>begin_frequency) & (freqs<end_frequency)]
        
        factor_a = -1 / tau
            
        self.selected_omega_vector = selected_omega_vector
        self.pos_half_spectra = spectrum_tensor
        self.factor_a = factor_a

        self.state[0] = True
    
    @property
    def num_omega(self):
        return self.selected_omega_vector.shape[0]
    
    def estimate_model(self, order, complex_coefficients=False):
        '''
        Estimate a right matrix-fraction model from positive half-spectra, by the
        constructinga set of reduced normal equations as shown in Peeters 2004. 
        The polynomial is identified following Cauberghe 2004. Sec. 5.2.1 and 
        converted into a state-space model, as outlined in Reynders-2012: Lemma 2.2
        
        Verboven 2002: Sect. 5.3.3 has a discussion on the use of real or complex
        valued coefficients, favoring complex ones. Guillaume 2003, Peeters 2004 
        just assume real coefficients, while later references, e.g.  
        Cauberghe 2004, Reynders 2012 use complex coefficients.
        However, with complex coefficients, stabilization diagrams seem to 
        become corrupted. This implementation uses real coefficients.
        
        Note: The previous implementation was wrong in the estimation of
        alpha coefficients and led to "bad" stabilization. Additionally there 
        was a wrong sign in the assembly of the C_c matrix, which led to corrupted
        mode shapes.
        
        .. TO DO::
            * implement weighting function; c.p. Peeters 2004 Sect. 2.2
            * improve assembly by exploiting the structure of S, R, T; c.p. Cauberghe 2004 Eq. 5.17ff
            * estimate polynomial once at highest order and construct all lower 
            order models from these coefficients; c.p. Peeters 2004 Sect. 2.4
        
        Parameters
        ----------
            order: integer, required
                Model order, at which the RMF model should be estimated
            
            complex_coefficients: bool, optional
                Whether to assume real or complex coefficients
                
        Returns
        -------
            A_c: numpy.ndarray
                Companion matrix: Array of shape (order * n_r, order * n_r)
                
            C_c: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order * n_r)
        
        '''
        
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
        selected_omega_vector = self.selected_omega_vector
        num_omega = self.num_omega
        pos_half_spectra = self.pos_half_spectra

        sampling_rate = self.prep_signals.sampling_rate
        Delta_t = 1 / sampling_rate
        
        # whether to assume real or complex coefficients
        if complex_coefficients:
            dtype=complex
        else:
            dtype=float
        
        RS_solutions = np.zeros((order + 1, (order + 1) * n_r, n_l), dtype=dtype)
        M = np.zeros(((order + 1) * n_r, (order + 1) * n_r), dtype=dtype)
        
        # Create matrices X_0 and Y_0, Peeters 2004: Sect. 2.2ff
        # for channel-dependent weights, this has to move into the loop below
        X_o = np.exp(1j * selected_omega_vector[:, np.newaxis] * 
                     Delta_t * np.arange(order + 1)[np.newaxis, :]) # (num_omega, (order + 1))
        X_o_H = np.conj(X_o.T) # ((order + 1), num_omega)
        R_o = X_o_H @ X_o# ((order + 1),(order + 1))
        if not complex_coefficients: R_o = R_o.real 
        
        Y_o = np.empty((num_omega, ((order + 1) * n_r)), dtype=complex)
        for i_l in range(n_l):
            for kk in range(num_omega):
                Y_o[kk, :] = np.kron(-X_o[kk, :], pos_half_spectra[i_l, :, kk].T)
            
            S_o = X_o_H @ Y_o # ((order + 1),(order + 1) * n_r)
            if not complex_coefficients: S_o = S_o.real
            
            T_o = np.conj(Y_o.T) @ Y_o# ((order + 1) * n_r,(order + 1) * n_r)
            if not complex_coefficients: T_o = T_o.real 
            
            RS_solution = np.linalg.solve(R_o, S_o)
            
            M = M + (T_o - np.conj(S_o).T @ RS_solution)
            M *= 2
            
            RS_solutions[:, :, i_l] = RS_solution
        
        # Compute alpha and beta coefficients: Cauberghe 2004. Sec. 5.2.1
        M_aa = M[:order * n_r, :order * n_r]
        M_ab = M[:order * n_r, -n_r:]
        alpha_b = - np.linalg.solve(M_aa, M_ab)
        alpha = np.concatenate((alpha_b, np.eye(n_r)), axis=0) # ((order + 1) * n_r, n_r)
        
        beta_o_i = np.zeros(((order + 1), n_r, n_l), dtype=dtype)
        for i_l in range(n_l):
            RS_solution = RS_solutions[:, :, i_l]
            beta_o = - RS_solution @ alpha
            
            beta_o_i[:, :, i_l] = beta_o
        
        # Create matrices A_c and C_c; 
        # Reynders-2012-SystemIdentificationMethodsFor(Operational)ModalAnalysisReviewAndComparison: Lemma 2.2
        A_p = alpha[-n_r:, :]
        B_p = beta_o_i[order, :, :].T
        
        A_c = np.zeros((order * n_r, order * n_r), dtype=dtype)
        C_c = np.zeros((n_l, order * n_r), dtype=dtype)
        
        for p_i in range(order):
            A_p_i = alpha[(order - p_i - 1) * n_r:(order - p_i) * n_r, :]
            
            this_A_c_block = - np.linalg.solve(A_p, A_p_i)
            A_c[:n_r, p_i * n_r:(p_i + 1) * n_r] = this_A_c_block
            
            B_p_i = beta_o_i[order - p_i - 1, :, :].T
            
            this_C_c_block = B_p_i + (B_p @ this_A_c_block)
            C_c[:, p_i * n_r:(p_i + 1) * n_r] = this_C_c_block
        
        A_c_rest = np.eye((order - 1) * n_r)
        A_c[n_r:, :(order - 1) * n_r] = A_c_rest
        
        return A_c, C_c
    
    def modal_analysis(self, A_c, C_c):
        '''
        Perform a modal decomposition of the identified companion matrix A_c. 
        Mode shapes are scaled to unit modal displacements. Complex conjugate 
        and real modes are removed prior to further processing. Damping values
        are corrected, if half-spectra were constructed with an exponential window.
        
        Parameters
        -------
            A_c: numpy.ndarray
                Companion matrix: Array of shape (order * n_r, order * n_r)
                
            C_c: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order * n_r)
         
        Returns
        -------
            modal_frequencies: (order * n_r,) numpy.ndarray 
                Array holding the modal frequencies for each mode
            modal_damping: (order * n_r,) numpy.ndarray 
                Array holding the modal damping ratios (0,100) for each mode
            mode_shapes: (n_l, order * n_r,) numpy.ndarray 
                Complex array holding the mode shapes 
            eigenvalues: (order * n_r,) numpy.ndarray
                Complex array holding the eigenvalues for each mode
        '''
        accel_channels = self.prep_signals.accel_channels
        velo_channels = self.prep_signals.velo_channels
        
        n_l = self.prep_signals.num_analised_channels
        factor_a = self.factor_a
        sampling_rate = self.prep_signals.sampling_rate
        
        eigvals, eigvecs_r = np.linalg.eig(A_c)
        
        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)
        n_modes = len(conj_indices)
        
        modal_frequencies = np.zeros((n_modes,))
        modal_damping = np.zeros((n_modes, ))
        mode_shapes = np.zeros((n_l, n_modes), dtype=complex)
        eigenvalues = np.zeros((n_modes), dtype=complex)
        
        Phi = C_c @ eigvecs_r
        
        for i, ind in enumerate(reversed(conj_indices)):
            
            lambda_i = np.log(eigvals[ind]) * sampling_rate
            freq_i = np.abs(lambda_i) / (2 * np.pi)
            
            # damping without correction if no exponential window was applied
            # damping_i = np.real(lambda_i)/np.abs(lambda_i) * (-100)
            # damping with correction if exponential window was applied to
            damping_i = (np.real(lambda_i) / np.abs(lambda_i) - factor_a * (sampling_rate) / (freq_i * 2 * np.pi)) * (-100)

            mode_shape_i = Phi[:, ind]
            
            # scale modeshapes to modal displacements
            mode_shape_i = self.integrate_quantities(
                mode_shape_i, accel_channels, velo_channels, freq_i * 2 * np.pi)
            
            # rotate mode shape in complex plane
            mode_shape_i = self.rescale_mode_shape(mode_shape_i)
            
            modal_frequencies[i] = freq_i
            modal_damping[i] = damping_i
            mode_shapes[:, i] = mode_shape_i
            eigenvalues[i] = lambda_i
        
        return modal_frequencies, modal_damping, eigenvalues, mode_shapes
        
    def compute_modal_params(self, max_model_order, complex_coefficients=False):
        '''
        Perform a multi-order computation of modal parameters. Successively
        calls 
        
         * estimate_model(order, complex_coefficients)
         * modal_analysis(A_r, C_r)
        
        At ascending model orders, up to max_model_order. 
        See the explanations in the the respective methods, for a detailed 
        explanation of parameters.
        
        Parameters
        ----------
            max_model_order: integer, optional
                Maximum model order, where to interrupt the algorithm.
        '''
        
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
            
        assert self.state[0]

        logger.info('Computing modal parameters...')
        
        # Peeters 2004,p 400: "a pth order right matrix-fraction model yield pm poles"
        modal_frequencies = np.zeros((max_model_order, max_model_order*n_r))
        modal_damping = np.zeros((max_model_order, max_model_order*n_r))
        mode_shapes = np.zeros((n_l, max_model_order*n_r, max_model_order), dtype=complex)
        eigenvalues = np.zeros((max_model_order, max_model_order*n_r), dtype=complex)

        printsteps = list(np.linspace(0, max_model_order, 100, dtype=int))
        for order in range(1,max_model_order):
            while order in printsteps:
                del printsteps[0]
                print('.', end='', flush=True)
            
            A_c, C_c = self.estimate_model(order)

            f, d, lamda, phi = self.modal_analysis(A_c, C_c)
            n_modes = len(f)

            modal_frequencies[order, :n_modes] = f
            modal_damping[order, :n_modes] = d
            eigenvalues[order,:n_modes] = lamda
            mode_shapes[:, :n_modes, order] = phi
            
        print('.', end='\n', flush=True)

        self.max_model_order = max_model_order
        
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
        # out_dict['self.prep_signals']=self.prep_signals
        if self.state[0]:  # half spectra
            out_dict['self.begin_frequency'] = self.begin_frequency
            out_dict['self.end_frequency'] = self.end_frequency
            out_dict['self.nperseg'] = self.nperseg
            out_dict['self.selected_omega_vector'] = self.selected_omega_vector
            out_dict['self.pos_half_spectra'] = self.pos_half_spectra
            out_dict['self.factor_a'] = self.factor_a
        if self.state[1]:  # modal params
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.eigenvalues'] = self.eigenvalues

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_signals):
        logger.info('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        assert isinstance(prep_signals, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())
        assert setup_name == prep_signals.setup_name
        start_time = prep_signals.start_time

        assert start_time == prep_signals.start_time
        
        pLSCF_object = cls(prep_signals)
        pLSCF_object.state = state
        if state[0]:  # positive half spectra
            pLSCF_object.begin_frequency = validate_array(in_dict['self.begin_frequency'])
            pLSCF_object.end_frequency = validate_array(in_dict['self.end_frequency'])
            pLSCF_object.nperseg = validate_array(in_dict['self.nperseg'])
            pLSCF_object.selected_omega_vector = validate_array(in_dict['self.selected_omega_vector'])
            pLSCF_object.pos_half_spectra = validate_array(in_dict['self.pos_half_spectra'])
            pLSCF_object.factor_a = validate_array(in_dict['self.factor_a'])
        if state[1]:  # modal params
            pLSCF_object.max_model_order = int(in_dict['self.max_model_order'])
            pLSCF_object.modal_frequencies = in_dict['self.modal_frequencies']
            pLSCF_object.modal_damping = in_dict['self.modal_damping']
            pLSCF_object.mode_shapes = in_dict['self.mode_shapes']
            pLSCF_object.eigenvalues = in_dict['self.eigenvalues']

        return pLSCF_object


def main():
    pass


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
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

Created on Thu Oct 16 14:41:56 2014

@author: volkmar
"""

import numpy as np
import scipy.linalg

import os

from .Helpers import lq_decomp, validate_array

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

'''
..todo::
     * define unit tests to check functionality after changes
     * update SSIData to follow the generalized subspace algorithm approach by doehler et.al.
     * parallel state-estimation for SSI-DataMC (different starting points and overlapping states)
     * add switch to keep synthesized time-histories
'''


# class SSIData(ModalBase):
#
#     def __init__(self, *args, **kwargs):
#         '''
#         channel definition: channels start at 0
#         '''
#         logger.warning("This implementation of SSIData is outdated. Use SSIDataMC or VarSSIRef (with projection method) instead")
#         super().__init__(*args, **kwargs)
#         #             0         1           2             3
#         # self.state= [Hankel, QR_decomp.,  State matr.,  Modal Par.]
#         self.state = [False, False, False, False]
#
#         #self.num_block_columns = None
#         self.num_block_rows = None
#         self.Hankel_matrix_T = None
#
#         self.max_model_order = None
#         self.P_i_ref = None
#         self.state_matrix = None
#         self.output_matrix = None
#
#     @classmethod
#     def init_from_config(cls, conf_file, prep_signals):
#         assert os.path.exists(conf_file)
#         assert isinstance(prep_signals, PreProcessSignals)
#
#         with open(conf_file, 'r') as f:
#
#             assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Columns:'
#             num_block_rows = int(f. __next__().strip('\n'))
#             assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
#             max_model_order = int(f. __next__().strip('\n'))
#
#         ssi_object = cls(prep_signals)
#         ssi_object.build_block_hankel(num_block_rows)
#         ssi_object.compute_projection_matrix(num_block_rows)
#         ssi_object.compute_state_matrices(num_block_rows, max_model_order)
#         ssi_object.compute_modal_params(max_model_order)
#
#         return ssi_object
#
#     def build_block_hankel(self, num_block_rows=None, ):
#         '''
#         Builds a Block-Hankel Matrix of the measured time series with varying time lags
#
#         ::
#
#              <- num_time samples - num_block_rows->       _
#             [     y_0      y_1      ...      y_(j-1)     ]^
#             [     y_1      y_2      ...      y_j         ]num_block_rows (=i)*num_analised_channels
#             [     ...      ...      ...      ...         ]v
#             [     y_(2i-1)   y_(2i)  ...     y_(2i+j-2)  ]_
#         '''
#
#         # print(multiprocess)
#         assert isinstance(num_block_rows, int)
#
#         # self.num_block_columns=num_block_columns
#         self.num_block_rows = num_block_rows
#         total_time_steps = self.prep_signals.total_time_steps
#         ref_channels = sorted(self.prep_signals.ref_channels)
#         #roving_channels = self.prep_signals.roving_channels
#         measurement = self.prep_signals.signals
#         num_analised_channels = self.prep_signals.num_analised_channels
#         num_ref_channels = self.prep_signals.num_ref_channels
#
#         # Reduce maximal size of Hankel matrix to a fixed value
#         flexlimit = total_time_steps - (num_block_rows) + 1
#         fixlimit = 10000  # 14000
#         extract_length = int(min(flexlimit, (fixlimit - (num_block_rows) + 1)))
#         logger.debug('extract_length = ', extract_length)
#
#         if fixlimit < total_time_steps:
#             measurement = measurement[0:(fixlimit + 1), :]
#             total_time_steps = fixlimit
#
#         # Extract reference time series
#         #all_channels = ref_channels + roving_channels
#         # all_channels.sort()
#
#         if (num_ref_channels < num_analised_channels):
#
#             refs = (measurement[0:extract_length, ref_channels])
#
#         else:
#             refs = measurement[0:extract_length, :]
#
#         logger.info('Creating block Hankel matrix...')
#
#         i = num_block_rows
#         j = total_time_steps - 2 * i
#         doehler_style = True
#         if doehler_style:
#             q, p = i, i
#
#             Y_minus = np.zeros((q * num_ref_channels, j))
#             Y_plus = np.zeros((p * num_analised_channels, j))
#
#             for ii in range(i):
#                 Y_minus[(q - ii - 1) * num_ref_channels:(q - ii) *
#                         num_ref_channels, :] = refs[ii:(ii + j)].T
#                 Y_plus[ii *
#                        num_analised_channels:(ii +
#                                               1) *
#                        num_analised_channels, :] = measurement[(i +
#                                                                 ii):(i +
#                                                                      ii +
#                                                                      j)].T
#
#             Hankel_matrix = np.vstack((Y_minus, Y_plus))
#             Hankel_matrix /= np.sqrt(j)
#             self.Hankel_matrix = Hankel_matrix
#
#         else:
#             Hankel_matrix_T = np.zeros(
#                 (j, (num_ref_channels * i + num_analised_channels * i)))
#
#             for ii in range(i):
#
#                 Hankel_matrix_T[:, ii *
#                                 num_ref_channels:(ii +
#                                                   1) *
#                                 num_ref_channels] = refs[ii:(ii +
#                                                              j)]
#
#             for ii in range(i):
#
#                 Hankel_matrix_T[:, (i *
#                                     num_ref_channels +
#                                     ii *
#                                     num_analised_channels): (i *
#                                                              num_ref_channels +
#                                                              (ii +
#                                                               1) *
#                                                              num_analised_channels)] = measurement[(i +
#                                                                                                     ii):(i +
#                                                                                                          ii +
#                                                                                                          j)]
#
#             Hankel_matrix_T = Hankel_matrix_T / np.sqrt(j)
#
#             self.Hankel_matrix_T = Hankel_matrix_T
#         self.state[0] = True
#
#     def compute_projection_matrix(
#             self,
#             num_block_rows=None):
#         '''
#         QR decomposition of [Y_(0|2i-1)]^T
#         '''
#
#         logger.info('Computing QR decomposition of block Hankel matrix...')
#
#         Hankel_matrix_T = self.Hankel_matrix_T
#         num_analised_channels = self.prep_signals.num_analised_channels
#         num_ref_channels = self.prep_signals.num_ref_channels
#         i = num_block_rows
#
#         doehler_style = True
#         if doehler_style:
#             l, q = lq_decomp(self.Hankel_matrix, mode='reduced')
#             R21 = l[(num_ref_channels * i):((num_ref_channels + \
#                      num_analised_channels) * i), 0:(num_ref_channels * i)]
#             P_i_ref = R21
#
#         else:
#             shape = Hankel_matrix_T.shape
#             logger.debug('Hankel shape = ', shape)
#
#             Q, R = scipy.linalg.qr(Hankel_matrix_T, mode='economic')
#             # Reduce Q (MxK) to Q (MXN) and R (KxN) to R (NxN), where N =
#             # total_time_steps - 2*num_block_rows
#             Q = (Q[:, 0:((num_ref_channels + num_analised_channels) * i)]).T
#             R = (R[0:((num_ref_channels + num_analised_channels) * i), :]).T
#
#             #check_I = np.dot(Q,Q.T)
#             #new_Hankel = np.dot(R,Q)
#             #Hankel_diff = Hankel_matrix_T - new_Hankel.T
#
#             #R_21 = R[(num_ref_channels*i):(num_ref_channels*(i+1)),0:(num_ref_channels*i)]
#             #R_22 = R[(num_ref_channels*i):(num_ref_channels*(i+1)),(num_ref_channels*i):(num_ref_channels*(i+1))]
#             #R_31 = R[(num_ref_channels*(i+1)):(num_ref_channels*(i+1)+num_roving_channels),0:(num_ref_channels*i)]
#             #R_32 = R[(num_ref_channels*(i+1)):(num_ref_channels*(i+1)+num_roving_channels),(num_ref_channels*i):(num_ref_channels*(i+1))]
#             #R_33 = R[(num_ref_channels*(i+1)):(num_ref_channels*(i+1)+num_roving_channels),(num_ref_channels*(i+1)):(num_ref_channels*(i+1)+num_roving_channels)]
#             #R_41 = R[(num_ref_channels*(i+1)+num_roving_channels):((num_ref_channels + num_analised_channels) * i),0:(num_ref_channels*i)]
#             #R_42 = R[(num_ref_channels*(i+1)+num_roving_channels):((num_ref_channels + num_analised_channels) * i),(num_ref_channels*i):(num_ref_channels*(i+1))]
#
#             #Q_1 = Q[0:(num_ref_channels*i),:]
#             #Q_12 = Q[0:(num_ref_channels*(i+1)),:]
#             #Q_123 = Q[0:(num_ref_channels*(i+1)+num_roving_channels),:]
#             P_i_ref = R[(num_ref_channels * i):((num_ref_channels + \
#                          num_analised_channels) * i), 0:(num_ref_channels * i)]
#             #P_i_ref = np.dot(P_i_ref, Q_1)
#
#         self.P_i_ref = P_i_ref
#         self.state[1] = True
#         self.state[2] = False  # previous state matrices are invalid now
#
#     def compute_state_matrices(
#             self,
#             num_block_rows=None,
#             max_model_order=None):
#         '''
#         computes the state and output matrices A and C, resp., of the state-space-model
#         by applying a singular value decomposition to the projection matrix P_i_ref
#         the state space model matrices are obtained by appropriate truncation
#         of the svd matrices at max_model_order
#         '''
#         if max_model_order is not None:
#             assert isinstance(max_model_order, int)
#             self.max_model_order = max_model_order
#
#         assert self.state[1]
#
#         P_i_ref = self.P_i_ref
#         num_analised_channels = self.prep_signals.num_analised_channels
#
#         logger.info('Computing state matrices A and C...')
#
#         [U, S, V_T] = np.linalg.svd(P_i_ref)
#         S_2 = np.diag(np.sqrt(S))
#
#         # choose highest possible model order
#         if max_model_order is None:
#             max_model_order = len(S)
#         else:
#             max_model_order = min(max_model_order, len(S))
#
#         S_2 = S_2[:max_model_order, :max_model_order]
#         U = U[:, :max_model_order]
#         Oi_full = np.dot(U, S_2)
#         C_full = Oi_full[:num_analised_channels, :]
#         A_full = np.dot(np.linalg.pinv(Oi_full[:(num_analised_channels * (num_block_rows - 1)), :]),
#                         Oi_full[num_analised_channels:(num_analised_channels * num_block_rows), :])
#
#         #O_i1_full = Oi_full[:((num_block_rows-1)* num_analised_channels),:]
#
#         self.state_matrix = A_full
#         self.output_matrix = C_full
#         self.max_model_order = max_model_order
#
#         self.state[2] = True
#         self.state[3] = False  # previous modal params are invalid now
#
#     def compute_modal_params(self, max_model_order=None):
#
#         if max_model_order is not None:
#             assert isinstance(max_model_order, int)
#             self.max_model_order = max_model_order
#
#         assert self.state[2]
#
#         max_model_order = self.max_model_order
#         A_full = self.state_matrix
#         C_full = self.output_matrix
#         num_analised_channels = self.prep_signals.num_analised_channels
#         sampling_rate = self.prep_signals.sampling_rate
#
#         logger.info('Computing modal parameters...')
#
#         lambda_k = np.array([], dtype=complex)
#         modal_frequencies = np.zeros((max_model_order, max_model_order))
#         modal_damping = np.zeros((max_model_order, max_model_order))
#         mode_shapes = np.zeros(
#             (num_analised_channels,
#              max_model_order,
#              max_model_order),
#             dtype=complex)
#
#         '''
#         S_2 = np.diag(np.sqrt(S))
#         for index in range(max_model_order):
#
#             if index > 1:
#
#                 this_S = S_2[0:index,0:index]
#                 this_U=U[:,0:index]
#                 Oi = Oi_full[:,0:index]
#                 X_i = np.dot(np.linalg.pinv(Oi), P_i_ref)
#                 O_i1 = O_i1_full[:,0:index]
#                 X_i1 = np.dot(np.linalg.pinv(O_i1), P_i_minus_1_ref)
#
#
#                 Kalman_matrix = np.zeros(((num_analised_channels + index),dim_Y_i_i[1]))
#                 Kalman_matrix[0:index,:] = X_i1
#                 Kalman_matrix[index:(num_analised_channels + index),:] = Y_i_i
#                 AC_matrix = np.dot(Kalman_matrix, np.linalg.pinv(X_i))
#                 this_A = AC_matrix[0:index, :]
#                 this_C = AC_matrix[index:(num_analised_channels + index), :]
#
#                 print('INDEX = ', index)
#
#         '''
#
#
#         for order in range(0, max_model_order, 1):
#
#
#             eigenvalues_paired, eigenvectors_paired = np.linalg.eig(
#                 A_full[0:order + 1, 0:order + 1])
#             eigenvalues_single, eigenvectors_single = self.remove_conjugates(
#                 eigenvalues_paired, eigenvectors_paired)
# #                 ax1.plot(eigenvalues_single.real,eigenvalues_single.imag, ls='', marker='o')
#
#             lambdas = []
#             for index, k in enumerate(eigenvalues_single):
#                 lambda_k = np.log(complex(k)) * sampling_rate
#                 lambdas.append(lambda_k)
#                 freq_j = np.abs(lambda_k) / (2 * np.pi)
#                 damping_j = np.real(lambda_k) / np.abs(lambda_k) * (-100)
#                 mode_shapes_j = np.dot(
#                     C_full[:, 0:order + 1], eigenvectors_single[:, index])
#
#                 # integrate acceleration and velocity channels to level out all channels in phase and amplitude
#                 #mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))
#
#                 modal_frequencies[order, index] = freq_j
#                 modal_damping[order, index] = damping_j
#                 mode_shapes[:, index, order] = mode_shapes_j
#             lambdas = np.array(lambdas)
#
#         self.modal_frequencies = modal_frequencies
#         self.modal_damping = modal_damping
#         self.mode_shapes = mode_shapes
#
#         self.state[3] = True
#
#     def save_state(self, fname):
#
#         dirname, filename = os.path.split(fname)
#         if not os.path.isdir(dirname):
#             os.makedirs(dirname)
#
#         #             0         1           2             3
#         # self.state= [Hankel, QR_decomp.,  State matr.,  Modal Par.]
#         out_dict = {'self.state': self.state}
#         out_dict['self.setup_name'] = self.setup_name
#         out_dict['self.start_time'] = self.start_time
#         # out_dict['self.prep_signals']=self.prep_signals
#         if self.state[0]:  # Block Hankel matrix
#             out_dict['self.Hankel_matrix_T'] = self.Hankel_matrix_T
#             out_dict['self.num_block_rows'] = self.num_block_rows
#         if self.state[1]:  # QR decomposition, Projection matrix
#             out_dict['self.P_i_ref'] = self.P_i_ref
#         if self.state[2]:  # state models
#             out_dict['self.max_model_order'] = self.max_model_order
#             out_dict['self.state_matrix'] = self.state_matrix
#             out_dict['self.output_matrix'] = self.output_matrix
#         if self.state[3]:  # modal params
#             out_dict['self.modal_frequencies'] = self.modal_frequencies
#             out_dict['self.modal_damping'] = self.modal_damping
#             out_dict['self.mode_shapes'] = self.mode_shapes
#
#         np.savez_compressed(fname, **out_dict)
#
#     @classmethod
#     def load_state(cls, fname, prep_signals):
#         logger.info('Now loading previous results from  {}'.format(fname))
#
#         in_dict = np.load(fname, allow_pickle=True)
#         #             0         1           2             3
#         # self.state= [Hankel, QR_decomp.,  State matr.,  Modal Par.]
#         if 'self.state' in in_dict:
#             state = list(in_dict['self.state'])
#         else:
#             return
#
#         for this_state, state_string in zip(state, ['Block Hankel Matrix Built',
#                                                     'QR Decomposition Finished',
#                                                     'State Matrices Computed',
#                                                     'Modal Parameters Computed',
#                                                     ]):
#             if this_state:
#                 logger.debug(state_string)
#
#         assert isinstance(prep_signals, PreProcessSignals)
#         setup_name = str(in_dict['self.setup_name'].item())
#         start_time = in_dict['self.start_time'].item()
#         assert setup_name == prep_signals.setup_name
#         start_time = prep_signals.start_time
#
#         assert start_time == prep_signals.start_time
#         #prep_signals = in_dict['self.prep_signals'].item()
#         ssi_object = cls(prep_signals)
#         ssi_object.state = state
#         if state[0]:  # Block Hankel matrix
#             ssi_object.Hankel_matrix_T = in_dict['self.Hankel_matrix_T']
#             ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
#         if state[1]:  # QR decomposition, Projection matrix
#             ssi_object.P_i_ref = in_dict['self.P_i_ref']
#         if state[2]:  # state models
#             ssi_object.max_model_order = int(in_dict['self.max_model_order'])
#             ssi_object.state_matrix = in_dict['self.state_matrix']
#             ssi_object.output_matrix = in_dict['self.output_matrix']
#         if state[3]:  # modal params
#             ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
#             ssi_object.modal_damping = in_dict['self.modal_damping']
#             ssi_object.mode_shapes = in_dict['self.mode_shapes']
#
#         return ssi_object



class SSIDataMC(ModalBase):

    def __init__(self, *args, **kwargs):
        '''
        channel definition: channels start at 0
        '''
        super().__init__(*args, **kwargs)
        
        self.state = [False, False, False, False]

        self.num_block_rows = None

        self.P_i_ref = None
        self.P_i_1 = None
        self.Y_i_i = None
        self.S = None
        self.U = None
        self.V_T = None
        
        self.max_model_order = None

        self.modal_contributions = None

    @classmethod
    def init_from_config(cls, conf_file, prep_signals):
        assert os.path.exists(conf_file)
        assert isinstance(prep_signals, PreProcessSignals)

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Columns:'
            num_block_rows = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
            max_model_order = int(f. __next__().strip('\n'))

        ssi_object = cls(prep_signals)
        ssi_object.build_block_hankel(num_block_rows)
        ssi_object.compute_modal_params(max_model_order)

        return ssi_object

    def build_block_hankel(self, num_block_rows=None):
        '''
        Builds a Block-Hankel Matrix of the measured time series with varying
        time lags and estimates the subspace matrix from its LQ decomposition. 

        ::

              <- num_time samples - num_block_rows->      _
            [     y_0      y_1      ...      y_(j-1)     ]^
            [     y_1      y_2      ...      y_j         ]num_block_rows (=i)*n_l
            [     ...      ...      ...      ...         ]v
            [     y_(2i-1)   y_(2i)  ...     y_(2i+j-2)  ]_
            
            
        The notation mostly follows Peeters 1999.
        
        Parameters
        -------
            num_block_rows: integer, required
                The number of block rows of the Subspace matrix
        '''

        if num_block_rows is None:
            num_block_rows = self.num_block_rows

        assert isinstance(num_block_rows, int)

        self.num_block_rows = num_block_rows

        signals = self.prep_signals.signals
        total_time_steps = self.prep_signals.total_time_steps

        ref_channels = sorted(self.prep_signals.ref_channels)
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels

        logger.info('Building Block-Hankel matrix...')

        q = num_block_rows
        p = num_block_rows
        N = int(total_time_steps - 2 * p)

        Y_minus = np.zeros((q * n_r, N))
        Y_plus = np.zeros(((p + 1) * n_l, N))

        for ii in range(q):
            Y_minus[(q - ii - 1) * n_r:(q - ii) * n_r, :] = signals[(ii):(ii + N), ref_channels].T
        for ii in range(p + 1):
            Y_plus[ii * n_l:(ii + 1) * n_l, :] = signals[(q + ii):(q + ii + N)].T

        Hankel_matrix = np.vstack((Y_minus, Y_plus))
        Hankel_matrix /= np.sqrt(N)
        
        # self.Hankel_matrix = Hankel_matrix
        
        logger.info('Estimating subspace matrix...')
        
        l, q = lq_decomp(Hankel_matrix, mode='full')

        a = n_r * p
        b = n_r
        c = n_l - n_r
        d = n_l * p

        P_i_ref = l[a:a + b + c + d, : a] @ q[ :a, :]
        
        [U, S, V_T] = np.linalg.svd(P_i_ref, full_matrices=False)
        
        
        P_i_1 = l[a + b + c:a + b + c + d, : a + b] @ q[ : a + b, : ]
        Y_i_i = l[a : a + b + c, : a + b + c] @ q[ : a + b + c, : ]
        
        
        self.P_i_1 = P_i_1
        self.P_i_ref = P_i_ref
        self.Y_i_i = Y_i_i
        
        self.S = S
        self.U = U
        self.V_T = V_T

        self.max_model_order = self.S.shape[0]

        self.state[0] = True

    def compute_modal_params(self, max_model_order=None, j=None, synth_sig=True):
        '''
        Perform a multi-order computation of modal parameters. Successively
        calls
        
         * estimate_state(order,)
         * modal_analysis(A,C)
         * synthesize_signals(A, C, Q, R, S, j) 
        
        at ascending model orders, up to max_model_order. 
        See the explanations in the the respective methods, for a detailed 
        explanation of parameters.
        
        Parameters
        ----------
            max_model_order: integer, optional
                Maximum model order, where to interrupt the algorithm. If not given,
                it is determined from the previously computed subspace matrix.
        '''
        assert self.state[0]
        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
        else:
            max_model_order = self.max_model_order
            
        assert max_model_order <= self.S.shape[0]

        # num_block_rows = self.num_block_rows
        num_analised_channels = self.prep_signals.num_analised_channels
        # num_ref_channels = self.prep_signals.num_ref_channels
        # sampling_rate = self.prep_signals.sampling_rate
        
        if j is None:
            j = self.prep_signals.total_time_steps
        
        assert j <= self.prep_signals.signals.shape[0]

        logger.info('Computing modal parameters...')
        eigenvalues = np.zeros(
            (max_model_order, max_model_order), dtype=np.complex128)
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros(
            (num_analised_channels,
             max_model_order,
             max_model_order),
            dtype=complex)
        modal_contributions = np.zeros((max_model_order, max_model_order))
        

        
        printsteps = list(np.linspace(0, max_model_order, 100, dtype=int))
        for order in range(1, max_model_order):
            while order in printsteps:
                del printsteps[0]
                print('.', end='', flush=True)

            A,C,Q,R,S = self.estimate_state(order)
            
            f, d, phi, lamda, = self.modal_analysis(A, C)
            modal_frequencies[order, :order] = f
            modal_damping[order, :order] = d
            mode_shapes[:phi.shape[0], :order, order] = phi
            eigenvalues[order, :order] = lamda
            
            if synth_sig:
                _, delta = self.synthesize_signals(A, C, Q, R, S, j)
                modal_contributions[order, :order] = delta

        self.max_model_order = max_model_order
        
        self.modal_contributions = modal_contributions
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues

        self.state[2] = True
        print('.', end='\n', flush=True)
        
        
    def estimate_state(self, order,):
        '''
        Estimate the state matrices A, C and noise covariances Q, R and S from
        the subspace / projection matrix. Several methods exist, e.g. 
        
         * Peeters 1999 Reference Based Stochastic Subspace Identification for OMA
         * DeCock 2007 Subspace Identification Methods
         * the algorithm used in BRSSICovRef.
        
        Here, the first algorithm, a residual-based computation of Q, R  and S, 
        is implemented.
        
        
        Parameters
        ----------
            order: integer, required
                The model order, at which to truncate the singular values of the
                projection Matrix P_i_ref
                
        Returns
        -------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
                
            Q: numpy.ndarray
                state noise covariance matrix: Symmetric array of shape (order, order)
            
            R: numpy.ndarray
                signal noise covariance matrix: Array of shape (num_analised_channels, num_analised_channels)
            
            S: numpy.ndarray
                system noise - signal noise covariance matrix: Array of shape (order, num_analised_channels)
        '''
        num_block_rows = self.num_block_rows
        num_analised_channels = self.prep_signals.num_analised_channels
        # num_ref_channels = self.prep_signals.num_ref_channels
        
        U = self.U[:, :order]
        S = self.S[:order]
        # V_T = self.V_T[:order, :]
        
        P_i_1 = self.P_i_1
        P_i_ref = self.P_i_ref
        Y_i_i = self.Y_i_i
        
        # compute state-space model
        S_2 = np.power(S, 0.5)
        O = U * S_2[np.newaxis, :]
        
        O_i_1 = O[:num_analised_channels * num_block_rows, :order]
        O_i = O[:, :order]

        X_i = np.linalg.pinv(O_i) @ P_i_ref
        X_i_1 = np.linalg.pinv(O_i_1) @ P_i_1

        X_i_1_Y_i = np.vstack((X_i_1, Y_i_i))

        AC = X_i_1_Y_i @ np.linalg.pinv(X_i)
        A = AC[:order, :]
        C = AC[order:, :]

        roh_w_v = X_i_1_Y_i - AC @ X_i

        QSR = roh_w_v @ roh_w_v.T

        Q = QSR[:order, :order]
        S = QSR[:order, order:order + num_analised_channels]
        R = QSR[order:order + num_analised_channels,
                order:order + num_analised_channels]
            
        return A, C, Q, R, S
    
    def modal_analysis(self, A, C, rescale_fun=None):
        '''
        Computes the modal parameters from a given state space model as described 
        by Peeters 1999 and Döhler 2012. Mode shapes are scaled to unit modal 
        displacements. Complex conjugate and real modes are removed prior to 
        further processing.
                
        Parameters
        ----------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
         
        Returns
        -------
            modal_frequencies: numpy.ndarray, shape (order,)
                Array holding the modal frequencies for each mode
            modal_damping: numpy.ndarray, shape (order,)
                Array holding the modal damping ratios (0,100) for each mode
            mode_shapes: numpy.ndarray, shape (n_l, order,)
                Complex array holding the mode shapes 
            eigenvalues: numpy.ndarray, shape (order,)
                Complex array holding the eigenvalues for each mode
        '''
        # collect variables
        accel_channels = self.prep_signals.accel_channels
        velo_channels = self.prep_signals.velo_channels
        sampling_rate = self.prep_signals.sampling_rate
        
        n_l = self.num_analised_channels
        
        order = A.shape[0]
        assert order == A.shape[1]
        
        # allocate output arrays
        modal_frequencies = np.zeros((order))
        modal_damping = np.zeros((order))
        mode_shapes = np.zeros((n_l, order), dtype=complex)
        eigenvalues = np.zeros((order), dtype=complex)
        
        # compute modal model
        eigvals, eigvecs_r = np.linalg.eig(A)
        Phi = C.dot(eigvecs_r)
        
        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)
        for i, ind in enumerate(conj_indices):

            lambda_i = eigvals[ind]
            mode_shape_i  = Phi[:,ind]

            a_i = np.abs(np.arctan2(np.imag(lambda_i), np.real(lambda_i)))
            b_i = np.log(np.abs(lambda_i))
            freq_i = np.sqrt(a_i**2 + b_i**2) * sampling_rate / 2 / np.pi
            damping_i = 100 * np.abs(b_i) / np.sqrt(a_i**2 + b_i**2)
            
            if rescale_fun is not None:
                mode_shape_i = rescale_fun(mode_shape_i)
            
            # scale modeshapes to modal displacements
            mode_shape_i = self.integrate_quantities(
                mode_shape_i, accel_channels, velo_channels, freq_i * 2 * np.pi)
            
            # rotate mode shape in complex plane
            mode_shape_i = self.rescale_mode_shape(mode_shape_i)

            modal_frequencies[i] = freq_i
            modal_damping[i] = damping_i
            mode_shapes[:mode_shape_i.shape[0], i] = mode_shape_i
            eigenvalues[i] = lambda_i

        return modal_frequencies, modal_damping, mode_shapes, eigenvalues, 
    
    def synthesize_signals(self, A, C, Q, R, S, j):
        '''
        Computes the modal response signals and the contribution of each mode.
        The algorithm follows Peeters 1999 and the Lyapunov equation is solved
        as a discrete-time algebraic Riccati equation (DARE). For long signals,
        the computation may become time-consuming, thus only time steps up to j
        may be used to synthesize the signal.
        
        
        Parameters
        ----------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
                
            Q: numpy.ndarray
                state noise covariance matrix: Symmetric array of shape (order, order)
            
            R: numpy.ndarray
                signal noise covariance matrix: Array of shape (num_analised_channels, num_analised_channels)
            
            S: numpy.ndarray
                system noise - signal noise covariance matrix: Array of shape (order, num_analised_channels)
         
        Returns
        -------
            sig_synth: numpy.ndarray, shape (num_analised_channels, j, order // 2)
                Array holding the modally decomposed input signals for
                each channel n_l and all modes
                
            modal_contributions: numpy.ndarray, shape (order, )
                Array holding the contributions of each mode to the input
                signals.
        '''
        
        order = A.shape[0]
        assert order == A.shape[1]
        
        signals = self.prep_signals.signals[:j, :]
        n_l = self.prep_signals.num_analised_channels
        
        modal_contributions = np.zeros((order))
        sig_synth = np.zeros((n_l, j, order // 2))
        
        try:
            P = scipy.linalg.solve_discrete_are(
                a=A.T, b=C.T, q=Q, r=R, s=S, balanced=True)
        except:
            logger.warning('Correlations of residuals are not symmetric. Skiping Modal Contributions')
            return sig_synth, modal_contributions

        APCS = A @ P @ C.T + S
        CPCR = C @ P @ C.T + R
        K = np.linalg.solve(CPCR.T, APCS.T,).T
        
        eigvals, eigvecs_r = np.linalg.eig(A)
        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)
        
        A_0 = np.diag(eigvals)
        C_0 = C @ eigvecs_r
        K_0 = np.linalg.solve(eigvecs_r, K)
        
        states = np.zeros((order, j), dtype=complex)

        AKC = A_0 - K_0 @ C_0
        K_0m = K_0 @ signals.T

        for k in range(j - 1):
            states[:, k + 1] = K_0m[:, k] + AKC @ states[:, k]

        Y = signals.T
        norm = 1 / np.einsum('ji,ji->j', Y, Y)
        
        for i, ind in enumerate(conj_indices):

            lambda_i = eigvals[ind]

            ident = eigvals == lambda_i.conj()
            ident[ind] = 1

            C_0I = C_0[:, ident]

            this_sig_synth = C_0I @ states[ident, :]
            if not np.all(np.isclose(this_sig_synth.imag,0)):
                logger.warning(f'Synthetized signals are complex at mode index {order}:{ind}.')
            
            sig_synth[:,:,i] = this_sig_synth.real

            mYT = np.einsum('ji,ji->j', sig_synth[:,:,i], Y)

            modal_contributions[i] = np.mean(norm * mYT)
        
        return sig_synth, modal_contributions
    
    def save_state(self, fname):

        logger.info('Saving results to  {}...'.format(fname))

        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time
        
        if self.state[0]:  # subspace matrix
            out_dict['self.num_block_rows'] = self.num_block_rows
            out_dict['self.P_i_1'] = self.P_i_1
            out_dict['self.P_i_ref'] = self.P_i_ref
            out_dict['self.Y_i_i'] = self.Y_i_i
            out_dict['self.S'] = self.S
            out_dict['self.U'] = self.U
            out_dict['self.V_T'] = self.V_T
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.S'] = self.S
            out_dict['self.U'] = self.U
            out_dict['self.V_T'] = self.V_T
        if self.state[2]:  # modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.modal_contributions'] = self.modal_contributions

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_signals):
        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        assert isinstance(prep_signals, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())
        
        assert setup_name == prep_signals.setup_name
        start_time = prep_signals.start_time

        assert start_time == prep_signals.start_time
        #prep_signals = in_dict['self.prep_signals'].item()
        ssi_object = cls(prep_signals)

        if state[0]:  # subspace matrix
            ssi_object.num_block_rows = validate_array(in_dict['self.num_block_rows'])
            ssi_object.P_i_1 = validate_array(in_dict['self.P_i_1'])
            ssi_object.P_i_ref = validate_array(in_dict['self.P_i_ref'])
            ssi_object.Y_i_i = validate_array(in_dict['self.Y_i_i'])
            ssi_object.S = validate_array(in_dict['self.S'])
            ssi_object.U = validate_array(in_dict['self.U'])
            ssi_object.V_T = validate_array(in_dict['self.V_T'])
            ssi_object.max_model_order = validate_array(in_dict['self.max_model_order'])
            ssi_object.S = validate_array(in_dict['self.S'])
            ssi_object.U = validate_array(in_dict['self.U'])
            ssi_object.V_T = validate_array(in_dict['self.V_T'])
        if state[2]:  # modal params
            ssi_object.modal_frequencies = validate_array(in_dict['self.modal_frequencies'])
            ssi_object.modal_damping = validate_array(in_dict['self.modal_damping'])
            ssi_object.eigenvalues = validate_array(in_dict['self.eigenvalues'])
            ssi_object.mode_shapes = validate_array(in_dict['self.mode_shapes'])
            ssi_object.modal_contributions = validate_array(in_dict['self.modal_contributions'])

        ssi_object.state = state
        return ssi_object

class SSIData(SSIDataMC):
    
    def compute_modal_params(self, max_model_order):
        
        '''
        Perform a multi-order computation of modal parameters. Successively
        calls
        
         * estimate_state(order,)
         * modal_analysis(A,C)
        
        at ascending model orders, up to max_model_order. 
        See the explanations in the the respective methods, for a detailed 
        explanation of parameters.
        
        Parameters
        ----------
            max_model_order: integer, optional
                Maximum model order, where to interrupt the algorithm. If not given,
                it is determined from the previously computed subspace matrix.
        '''
        super().compute_modal_params(max_model_order, synth_sig=False)
    
    def estimate_state(self, order, max_modes=None, algo='svd'):
        '''
        
        Compute the state matrix A and output matrix C  from the singular values 
        and vectors of the projection matrix, truncated at the requested order. Estimation of the
        state matrix can be performed by QR decomposition or Singular Value decomposition
        of the shifted observability matrix. If max_modes is specified, the singular
        value decomposition is truncated additionally, also known as Crystal Clear SSI.
        
        Parameters
        ----------
            order: integer, required
                Model order, at which the state matrices should be estimated
            
            max_modes: integer, optional
                Maximum number of modes, that are known to be present in the signal,
                to suppress noise modes
            
            algo: str, optional
                Algorithm to use for estimation of A. Either 'svd' or 'qr'.
                
        Returns
        -------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
                
        '''
        if order>self.max_model_order:
            raise RuntimeError(f'Order cannot be higher than {self.max_model_order}. Consider using more block_rows/block_columns.')
        
        assert algo in ['svd', 'qr']
        
        n_l = self.num_analised_channels
        
        num_block_rows = self.num_block_rows
        
        U = self.U[:, :order]
        S = self.S[:order]
        
        # compute state-space model
        S_2 = np.power(S, 0.5)
        O = U * S_2[np.newaxis, :]
        
        On_up = O[:n_l * num_block_rows, :order]
        On_down = O[n_l:n_l * (num_block_rows + 1), :order]
        
        if algo == 'svd':
            if max_modes is not None:
                [u, s, v_t] = np.linalg.svd(On_up, 0)
                s = 1. / s[:max_modes]
                # On_up_i = np.dot(np.transpose(v_t[:max_modes, :]), np.multiply(
                #     s[:, np.newaxis], np.transpose(u[:, :max_modes])))
            
                On_up_i = v_t[:max_modes, :].T @ (s[:, np.newaxis] *  u[:, :max_modes].T)
            else:
                On_up_i = np.linalg.pinv(On_up)  # , rcond=1e-12)
            A = On_up_i @ On_down

        elif algo == 'qr':
            Q, R = np.linalg.qr(On_up)
            S = Q.T.dot(On_down)
            A = np.linalg.solve(R, S)

        C = O[:n_l, :order] # output matrix
        
        return A, C, None, None, None
        
def main():
    pass


if __name__ == '__main__':
    main()

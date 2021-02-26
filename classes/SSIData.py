# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:41:56 2014

@author: volkmar
"""

import numpy as np
import scipy.linalg
#import sys
import os
#import json

import multiprocessing as mp
#import ctypes as c
from collections import deque
#import datetime
#from copy import deepcopy

from PreprocessingTools import PreprocessData
#import pyximport 
#pyximport.install()


#import cython_code.cython_helpers

'''
TODO:
- define unit tests to check functionality after changes
- update SSIData to follow the generalized subspace algorithm approach by doehler et.al.
- parallel state-estimation for SSI-DataMEC (different starting points and overlapping states)
- add switch to keep synthesized time-histories
- try to 
'''

#from decomp_qr_VZ import qr
#####################################################################################################################################
def rq_decomp(a, mode='full'):
    q,r = np.linalg.qr(np.flipud(a).T,mode=mode)
    return np.flipud(r.T), q.T

def ql_decomp(a, mode='full'):
    q,r = np.linalg.qr(np.fliplr(a),mode)
    return q, np.fliplr(r)

def lq_decomp(a, mode='full', unique=False):
    '''
    a: array_like, shape (M,N)
    l: (M,K)
    q: (K,N)
    '''
    if mode == 'r':
        r = np.linalg.qr(a.T,mode)
    else:
        q,r = np.linalg.qr(a.T,mode)
        
    if unique:
        fact = np.sign(np.diag(r))
        r*= np.repeat(np.reshape(fact,(r.shape[0],1)),r.shape[1],axis=1)
        if mode != 'r':
            q*= fact
            #print(np.allclose(a.T,q.dot(r)))
            
    if mode == 'r':
        return r.T
    else:
        return r.T, q.T
    
class SSIData(object):
    
    def __init__(self,prep_data):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_data, PreprocessData)
        self.prep_data =prep_data
        self.setup_name = prep_data.setup_name
        self.start_time = prep_data.start_time
        #             0         1           2             3
        #self.state= [Hankel, QR_decomp.,  State matr.,  Modal Par.]
        self.state  =[False,    False,     False,        False]
        
        #self.num_block_columns = None
        self.num_block_rows = None
        self.Hankel_matrix_T = None
        
        self.max_model_order = None
        self.P_i_ref = None
        self.state_matrix = None
        self.output_matrix = None
        
        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None
            
    @classmethod
    def init_from_config(cls,conf_file, prep_data):
        assert os.path.exists(conf_file)
        assert isinstance(prep_data, PreprocessData)
        
        with open(conf_file, 'r') as f:
            
            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Rows:'
            num_block_rows = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Maximum Model Order:'
            max_model_order= int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Use Multiprocessing:'
            multiprocessing= f.__next__().strip('\n').strip(' ')=='yes'
        
            
        ssi_object = cls(prep_data)
        ssi_object.build_block_hankel(num_block_rows, multiprocess=multiprocessing)
        ssi_object.compute_projection_matrix(num_block_rows, multiprocess=multiprocessing)
        ssi_object.compute_state_matrices(num_block_rows, max_model_order)
        ssi_object.compute_modal_params(max_model_order, multiprocessing)
        print('max_model_order = ', max_model_order)
        print('ssi_object.max_model_order = ', ssi_object.max_model_order)
        
        return ssi_object

        
    def build_block_hankel(self, num_block_rows=None, multiprocess=True):
        '''
        Builds a Block-Hankel Matrix of the measured time series with varying time lags
            | <- num_time samples - num_block_rows->     |_
            |     y_0      y_1      ...      y_(j-1)     |^
            |     y_1      y_2      ...      y_j         |num_block_rows (=i)*num_analised_channels
            |     ...      ...      ...      ...         |v
            |     y_(2i-1)   y_(2i)  ...     y_(2i+j-2)  |_
        '''
        #print(multiprocess)
        assert isinstance(num_block_rows, int)
        
        #self.num_block_columns=num_block_columns
        self.num_block_rows=num_block_rows
        total_time_steps = self.prep_data.total_time_steps
        ref_channels = sorted(self.prep_data.ref_channels)
        roving_channels = self.prep_data.roving_channels
        measurement = self.prep_data.measurement
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels 
        
        # Reduce maximal size of Hankel matrix to a fixed value 
        flexlimit = total_time_steps - (num_block_rows) + 1
        fixlimit = 10000   #14000
        extract_length = int(min(flexlimit, (fixlimit - (num_block_rows) + 1)))
        print('extract_length = ', extract_length)
        
        if fixlimit < total_time_steps:
            measurement = measurement[0:(fixlimit+1),:]
            total_time_steps = fixlimit
                       
        # Extract reference time series 
        all_channels = ref_channels + roving_channels
        all_channels.sort()
                              
        if (num_ref_channels < num_analised_channels):
            
            refs = (measurement[0:extract_length,ref_channels])
                     
        else:
            refs = measurement[0:extract_length,:]

       
           
        ###############################################################################
        ######## Create transpose of the block Hankel matrix [Y_(0|2i-1)]^T ###########
        ###############################################################################
        
        print('Creating block Hankel matrix...')
        
        i = num_block_rows
        j = total_time_steps - 2*i
        doehler_style = True
        if doehler_style:
            q,p=i,i
            
            Y_minus = np.zeros((q*num_ref_channels,j))
            Y_plus = np.zeros((p*num_analised_channels,j))
            
            
            for ii in range(i):
                Y_minus[(q-ii-1)*num_ref_channels:(q-ii)*num_ref_channels,:] = refs[ii:(ii+j)].T
                Y_plus[ii*num_analised_channels:(ii+1)*num_analised_channels,:] = measurement[(i+ii):(i+ii+j)].T
                
            Hankel_matrix = np.vstack((Y_minus,Y_plus))
            Hankel_matrix /=np.sqrt(j)
            self.Hankel_matrix = Hankel_matrix
        
        else:
            Hankel_matrix_T = np.zeros((j,(num_ref_channels*i + num_analised_channels*i)))
            
            for ii in range(i):
                
                Hankel_matrix_T[:, ii*num_ref_channels:(ii+1)*num_ref_channels] = \
                    refs[ii:(ii+j)]
            
            for ii in range(i):
                
                Hankel_matrix_T[:, (i*num_ref_channels + ii*num_analised_channels): \
                    (i*num_ref_channels + (ii+1)*num_analised_channels)] = \
                    measurement[(i+ii):(i+ii+j)]
            
            Hankel_matrix_T = Hankel_matrix_T / np.sqrt(j)
    
            self.Hankel_matrix_T = Hankel_matrix_T              
        self.state[0]=True
          
          
          
          
    def compute_projection_matrix(self, num_block_rows=None, multiprocess=True):
            
        ###############################################################################
        ####################### QR decomposition of [Y_(0|2i-1)]^T ####################
        ###############################################################################
        
        print('Computing QR decomposition of block Hankel matrix...')
        
        Hankel_matrix_T = self.Hankel_matrix_T              
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels 
        i = num_block_rows
        

        doehler_style = True
        if doehler_style:
            l,q = lq_decomp(self.Hankel_matrix, mode='reduced')
            R21 = l[(num_ref_channels*i):((num_ref_channels + num_analised_channels) * i),0:(num_ref_channels*i)] 
            P_i_ref = R21
            
        else:
            shape = Hankel_matrix_T.shape
            print('Hankel shape = ', shape)
    
            Q, R = scipy.linalg.qr(Hankel_matrix_T, mode='economic')
            # Reduce Q (MxK) to Q (MXN) and R (KxN) to R (NxN), where N = total_time_steps - 2*num_block_rows
            Q = (Q[:,0:((num_ref_channels + num_analised_channels) * i)]).T
            R = (R[0:((num_ref_channels + num_analised_channels) * i),:]).T
            
            #check_I = np.dot(Q,Q.T)
            #new_Hankel = np.dot(R,Q)
            #Hankel_diff = Hankel_matrix_T - new_Hankel.T
            
            #R_21 = R[(num_ref_channels*i):(num_ref_channels*(i+1)),0:(num_ref_channels*i)]
            #R_22 = R[(num_ref_channels*i):(num_ref_channels*(i+1)),(num_ref_channels*i):(num_ref_channels*(i+1))]
            #R_31 = R[(num_ref_channels*(i+1)):(num_ref_channels*(i+1)+num_roving_channels),0:(num_ref_channels*i)]
            #R_32 = R[(num_ref_channels*(i+1)):(num_ref_channels*(i+1)+num_roving_channels),(num_ref_channels*i):(num_ref_channels*(i+1))]
            #R_33 = R[(num_ref_channels*(i+1)):(num_ref_channels*(i+1)+num_roving_channels),(num_ref_channels*(i+1)):(num_ref_channels*(i+1)+num_roving_channels)]
            #R_41 = R[(num_ref_channels*(i+1)+num_roving_channels):((num_ref_channels + num_analised_channels) * i),0:(num_ref_channels*i)]
            #R_42 = R[(num_ref_channels*(i+1)+num_roving_channels):((num_ref_channels + num_analised_channels) * i),(num_ref_channels*i):(num_ref_channels*(i+1))]
            
            #Q_1 = Q[0:(num_ref_channels*i),:]
            #Q_12 = Q[0:(num_ref_channels*(i+1)),:]
            #Q_123 = Q[0:(num_ref_channels*(i+1)+num_roving_channels),:]
            P_i_ref = R[(num_ref_channels*i):((num_ref_channels + num_analised_channels) * i),0:(num_ref_channels*i)] 
            #P_i_ref = np.dot(P_i_ref, Q_1)
         
        self.P_i_ref = P_i_ref              
        self.state[1]=True
        self.state[2] = False # previous state matrices are invalid now
      
                       
    '''    
    def init_child_process(self, refs_memory_, measurement_memory_, toeplitz_memory_):
        #make the  memory arrays available to the child processes
        global refs_memory
        refs_memory = refs_memory_
        
        global measurement_memory
        measurement_memory = measurement_memory_   
        
        global toeplitz_memory
        toeplitz_memory = toeplitz_memory_
    '''
        
        
    def compute_state_matrices(self, num_block_rows=None, max_model_order=None):
        
        '''
        computes the state and output matrices A and C, resp., of the state-space-model
        by applying a singular value decomposition to the projection matrix P_i_ref
        the state space model matrices are obtained by appropriate truncation 
        of the svd matrices at max_model_order
        '''        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
            self.max_model_order = max_model_order 
        
        assert self.state[1]
        
        P_i_ref = self.P_i_ref              
        num_analised_channels = self.prep_data.num_analised_channels
           
        ###############################################################################
        ############# Computation of state matrices A and C ###########################
        ###############################################################################
       
        
        print('Computing state matrices A and C...')
        
        
        [U,S,V_T] = np.linalg.svd(P_i_ref);
        S_2 = np.diag(np.sqrt(S)) 
     
        # choose highest possible model order
        if max_model_order is None:
            max_model_order=len(S)
        else:
            max_model_order = min(max_model_order,len(S))       
        
        S_2=S_2[:max_model_order,:max_model_order]
        U=U[:,:max_model_order]
        Oi_full = np.dot(U, S_2)
        C_full=Oi_full[:num_analised_channels,:]        
        A_full = np.dot(np.linalg.pinv(Oi_full[:(num_analised_channels * (num_block_rows - 1)),:]),
                   Oi_full[num_analised_channels:(num_analised_channels * num_block_rows),:])
        
        #O_i1_full = Oi_full[:((num_block_rows-1)* num_analised_channels),:]
       
        self.state_matrix = A_full
        self.output_matrix = C_full
        self.max_model_order = max_model_order
        
        self.state[2] = True
        self.state[3] = False # previous modal params are invalid now
        

    def compute_modal_params(self, max_model_order=None, multiprocessing=True): 
        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
            self.max_model_order = max_model_order 
            
        assert self.state[2]
        
        max_model_order = self.max_model_order
        A_full = self.state_matrix
        C_full = self.output_matrix
        num_analised_channels = self.prep_data.num_analised_channels
        sampling_rate = self.prep_data.sampling_rate
          
        ###############################################################################
        ############# Computation of modal parameters #################################
        ###############################################################################
        
         
        print('Computing modal parameters...')
          
        lambda_k = np.array([], dtype=np.complex)        
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros((num_analised_channels, max_model_order, max_model_order),dtype=complex)
                
        '''                 
        S_2 = np.diag(np.sqrt(S)) 
        for index in range(max_model_order):    
            
            if index > 1:
                
                this_S = S_2[0:index,0:index]
                this_U=U[:,0:index]
                Oi = Oi_full[:,0:index]
                X_i = np.dot(np.linalg.pinv(Oi), P_i_ref)
                O_i1 = O_i1_full[:,0:index]
                X_i1 = np.dot(np.linalg.pinv(O_i1), P_i_minus_1_ref)
                
                
                Kalman_matrix = np.zeros(((num_analised_channels + index),dim_Y_i_i[1]))
                Kalman_matrix[0:index,:] = X_i1
                Kalman_matrix[index:(num_analised_channels + index),:] = Y_i_i
                AC_matrix = np.dot(Kalman_matrix, np.linalg.pinv(X_i))
                this_A = AC_matrix[0:index, :]
                this_C = AC_matrix[index:(num_analised_channels + index), :]
                
                print('INDEX = ', index)
        
        '''  
        
                    
        if multiprocessing:
            manager=mp.Manager()      #@UndefinedVariable  
            pool = mp.Pool()       #@UndefinedVariable 
            return_dict=manager.dict()
            
            #balanced allocation of work (numpy.linalg.eig takes approx. n^3 operations)
            work_slice_size = sum([n**3 for n in range(max_model_order)])/max_model_order
            current_size = 0
            current_orders = []
            for order in range(0,max_model_order,1):
                current_orders.append(order)
                current_size += order**3
                if current_size >= work_slice_size:
                    pool.apply_async(self.multiprocess_evd , args=(A_full, current_orders, return_dict))
                    current_orders = []
                    current_size = 0
            pool.close()
            pool.join()


        for order in range(0,max_model_order,1):    
            
            if multiprocessing:
                eigenvalues_single, eigenvectors_single = return_dict[order]
            else:
                eigenvalues_paired, eigenvectors_paired = np.linalg.eig(A_full[0:order+1, 0:order+1])                
                eigenvectors_single,eigenvalues_single = \
                    self.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
#                 ax1.plot(eigenvalues_single.real,eigenvalues_single.imag, ls='', marker='o')
                
            lambdas=[]
            for index,k in enumerate(eigenvalues_single): 
                lambda_k = np.log(complex(k)) * sampling_rate
                lambdas.append(lambda_k)
                freq_j = np.abs(lambda_k) / (2*np.pi)        
                damping_j = np.real(lambda_k)/np.abs(lambda_k) * (-100)  
                mode_shapes_j = np.dot(C_full[:, 0:order + 1], eigenvectors_single[:,index])
            
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                #mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))                
                        
                modal_frequencies[order,index]=freq_j
                modal_damping[order,index]=damping_j
                mode_shapes[:,index,order]=mode_shapes_j
            lambdas = np.array(lambdas)
        
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
            
        self.state[3]=True
        
    
    def multiprocess_evd(self, a, truncation_orders, return_dict):
        
        for truncation_order in truncation_orders:
            eigenvalues_paired, eigenvectors_paired = np.linalg.eig(a[0:truncation_order+1, 0:truncation_order+1])
    
            eigenvectors_single,eigenvalues_single = \
                    self.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            return_dict[truncation_order] = (eigenvalues_single, eigenvectors_single)
        
        return
                    
    @staticmethod
    def remove_conjugates_new (vectors, values):
        '''
        removes conjugates and marks the vectors which appear in pairs
        
        vectors.shape = [order+1, order+1]
        values.shape = [order+1,1]
        '''
        num_val=vectors.shape[1]
        conj_indices=deque()
        
        for i in range(num_val):
            this_vec=vectors[:,i]
            this_conj_vec = np.conj(this_vec)
            this_val=values[i]
            this_conj_val = np.conj(this_val)
            if this_val == this_conj_val: #remove real eigenvalues
                continue
            for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
                if vectors[0,j] == this_conj_vec[0] and \
                   vectors[-1,j] == this_conj_vec[-1] and \
                   values[j] == this_conj_val:
                    # saves computation time this function gets called many times and 
                    #numpy's np.all() function causes a lot of computation time
                    conj_indices.append(i)
                    break
        conj_indices=list(conj_indices)
        vector = vectors[:,conj_indices]
        value = values[conj_indices]

        return vector,value

    
    @staticmethod
    def integrate_quantities(vector, accel_channels, velo_channels, omega):
        # input quantities = [a, v, d]
        # output quantities = [d, d, d]
        # converts amplitude and phase
        #                     phase + 180; magn / omega^2
        
        vector[accel_channels] *= -1       / (omega ** 2)
        #                    phase + 90; magn / omega
        vector[velo_channels] *=  1j        / omega
        
        return vector   
    
    def save_state(self, fname):
        
        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        #             0         1           2             3
        #self.state= [Hankel, QR_decomp.,  State matr.,  Modal Par.]
        out_dict={'self.state':self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time']=self.start_time
        #out_dict['self.prep_data']=self.prep_data
        if self.state[0]:# Block Hankel matrix
            out_dict['self.Hankel_matrix_T'] = self.Hankel_matrix_T
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[1]:# QR decomposition, Projection matrix
            out_dict['self.P_i_ref'] = self.P_i_ref                          
        if self.state[2]:# state models
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.state_matrix'] = self.state_matrix
            out_dict['self.output_matrix'] = self.output_matrix
        if self.state[3]:# modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            
        np.savez_compressed(fname, **out_dict)

    @classmethod 
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname)    
        #             0         1           2             3
        #self.state= [Hankel, QR_decomp.,  State matr.,  Modal Par.]
        if 'self.state' in in_dict:
            state= list(in_dict['self.state'])
        else:
            return
        
        for this_state, state_string in zip(state, ['Block Hankel Matrix Built',
                                                    'QR Decomposition Finished',
                                                    'State Matrices Computed',
                                                    'Modal Parameters Computed',
                                                    ]):
            if this_state: print(state_string)
        
        assert isinstance(prep_data, PreprocessData)
        setup_name= str(in_dict['self.setup_name'].item())
        start_time=in_dict['self.start_time'].item()
        assert setup_name == prep_data.setup_name
        start_time = prep_data.start_time
        
        assert start_time == prep_data.start_time
        #prep_data = in_dict['self.prep_data'].item()
        ssi_object = cls(prep_data)
        ssi_object.state = state
        if state[0]:# Block Hankel matrix
            ssi_object.Hankel_matrix_T = in_dict['self.Hankel_matrix_T']
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
        if state[1]:# QR decomposition, Projection matrix
            ssi_object.P_i_ref = in_dict['self.P_i_ref']
        if state[2]:# state models
            ssi_object.max_model_order = int(in_dict['self.max_model_order'])
            ssi_object.state_matrix= in_dict['self.state_matrix']
            ssi_object.output_matrix = in_dict['self.output_matrix']
        if state[3]:# modal params
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']
         
        return ssi_object
    
    @staticmethod
    def rescale_mode_shape(modeshape):
        #scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape
    
class SSIDataMC(object):
    
    def __init__(self,prep_data):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_data, PreprocessData)
        self.prep_data =prep_data
        self.setup_name = prep_data.setup_name
        self.start_time = prep_data.start_time
        #             0         1           2             3
        #self.state= [Hankel, QR_decomp.,  State matr.,  Modal Par.]
        self.state  =[False,    False,     False,        False]
        
        #self.num_block_columns = None
        self.num_block_rows = None
        self.Hankel_matrix_T = None
        
        self.max_model_order = None
        self.P_i_ref = None
        self.state_matrix = None
        self.output_matrix = None
        
        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None
        self.modal_contributions = None
            
    @classmethod
    def init_from_config(cls,conf_file, prep_data):
        assert os.path.exists(conf_file)
        assert isinstance(prep_data, PreprocessData)
        
        with open(conf_file, 'r') as f:
            
            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Rows:'
            num_block_rows = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Maximum Model Order:'
            max_model_order= int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Use Multiprocessing:'
            multiprocessing= f.__next__().strip('\n').strip(' ')=='yes'
        
        
        ssi_object = cls(prep_data)
        ssi_object.build_block_hankel(num_block_rows)
        ssi_object.compute_state_matrices(max_model_order)
        ssi_object.compute_modal_params(plot_=False)
        #print('max_model_order = ', max_model_order)
        #print('ssi_object.max_model_order = ', ssi_object.max_model_order)
        
        return ssi_object

        
    def build_block_hankel(self, num_block_rows=None):
        '''
        Builds a Block-Hankel Matrix of the measured time series with varying time lags
            | <- num_time samples - num_block_rows->     |_
            |     y_0      y_1      ...      y_(j-1)     |^
            |     y_1      y_2      ...      y_j         |num_block_rows (=i)*num_analised_channels
            |     ...      ...      ...      ...         |v
            |     y_(2i-1)   y_(2i)  ...     y_(2i+j-2)  |_
        '''
        if num_block_rows is None:
            num_block_rows = self.num_block_rows
            
        assert isinstance(num_block_rows, int)
        
        self.num_block_rows=num_block_rows
        
        measurement = self.prep_data.measurement
        total_time_steps = self.prep_data.total_time_steps
        
        ref_channels = sorted(self.prep_data.ref_channels)
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels 
        
        print('Building subspace matrix...')
        
        q = num_block_rows
        p = num_block_rows
        N = int(total_time_steps - 2*p)
        
        Y_minus = np.zeros((q*num_ref_channels,N))
        Y_plus = np.zeros(((p+1)*num_analised_channels,N))
         
        for ii in range(q):
            Y_minus[(q-ii-1)*num_ref_channels:(q-ii)*num_ref_channels,:] = measurement[(ii):(ii+N),ref_channels].T
         
        for ii in range(p+1):
            Y_plus[ii*num_analised_channels:(ii+1)*num_analised_channels,:] = measurement[(q+ii):(q+ii+N)].T
             
        Hankel_matrix = np.vstack((Y_minus,Y_plus))
         
        Hankel_matrix /=np.sqrt(N)
        self.Hankel_matrix = Hankel_matrix
        
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels 

        
        l = lq_decomp(self.Hankel_matrix, mode='r')
        q_=True
        if q_:
            l,q = lq_decomp(self.Hankel_matrix, mode='full')
               
        
        a = num_ref_channels*p
        b = num_ref_channels
        c = num_analised_channels-num_ref_channels
        d = num_analised_channels*(p)
        
        R_k1 = l[:,:a]
        
        R_21 = R_k1[a:a+b,:]
        R_31 = R_k1[a+b:a+b+c,:]
        R_41 = R_k1[a+b+c:a+b+c+d,:]

        R_k2 = l[:,a:a+b]
                
        R_22 = R_k2[a:a+b,:]        
        R_32 = R_k2[a+b:a+b+c,:]    
        R_42 = R_k2[a+b+c:a+b+c+d,:]      
        
        R_k3 = l[:,a+b:a+b+c]
               
        R_33 = R_k3[a+b:a+b+c,:]    
        
        if q_:
            Q_1_T = q[0:a,:]
            Q_2_T = q[a:a+b,:]
            Q_3_T = q[a+b:a+b+c,:]
            Q_4_T = q[a+b+c:a+b+c+d,:]

        
        self.R_21 = R_21
        self.R_31 = R_31
        self.R_41 = R_41
 
        self.R_22 = R_22
        self.R_32 = R_32
        self.R_42 = R_42
         
        self.R_33 = R_33    
        
#         assert (self.R_21 == R_21).all()
#         assert (self.R_31 == R_31).all()
#         assert (self.R_41 == R_41).all()
# 
#         assert (self.R_22 == R_22).all()
#         assert (self.R_32 == R_32).all()
#         assert (self.R_42 == R_42).all()
#         
#         assert (self.R_33 == R_33).all()  
        
        if q_:
            self.Q_1_T = Q_1_T
            self.Q_2_T = Q_2_T
            self.Q_3_T = Q_3_T
            #self.Q_4_T = Q_4_T
         
        self.state[0]=True
#         self.state[1] = False # previous state matrices are invalid now

        
        
    def compute_state_matrices(self, max_model_order=None):
        
        '''
        computes the state and output matrices A and C, resp., of the state-space-model
        by applying a singular value decomposition to the projection matrix P_i_ref
        the state space model matrices are obtained by appropriate truncation 
        of the svd matrices at max_model_order
        '''        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
        
        assert self.state[0]
        
        num_analised_channels = self.prep_data.num_analised_channels
        num_block_rows = self.num_block_rows
        
        R_21 = self.R_21
        R_31 = self.R_31
        R_41 = self.R_41
        use_q=False
        if use_q:  
            # somewhere it is written, that Q can be ommitted since multiplication with q only transforms the subspace into a similar subspace (or something like that)
            # it was tried with different data sets and equal results were obtained regarding the stable modes
            Q_1_T = self.Q_1_T 
            P_i_ref = np.vstack((R_21,R_31,R_41)).dot(Q_1_T)
        else:
            P_i_ref = np.vstack((R_21,R_31,R_41))
        
        print('Computing state matrices A and C...')

        [U,S,V_T] = np.linalg.svd(P_i_ref)
        
        #print(P_i_ref.shape)
        #print(U.shape)
        #print(S.shape)
        #print(V_T.shape)
        
     
        # choose highest possible model order
        if max_model_order is None:
            max_model_order=len(S)
        else:
            max_model_order = min(max_model_order,len(S))       
        
        S = S[:max_model_order]
        U = U[:num_analised_channels*(num_block_rows+1),:max_model_order]
        V_T = V_T[:max_model_order,:]
        
        #print(U.shape)
        #print(S.shape)
        #print(V_T.shape)
        
        self.S = S
        self.U = U
        self.V_T = V_T
        
        self.max_model_order = max_model_order
        
        self.state[1] = True
        self.state[2] = False # previous modal params are invalid now
        
    def compute_modal_params(self, plot_=False  ): 
        '''
        c.p.    DeCock 2007 Subspace Identification Methods (estimation algo 1) -> with noisy data unstable, fast
                Peeters 1999 Reference Based Stochastic Subspace Identificaiton for Ouput-Only Modal Analysis (estimation algo 2) -> stable, slow
                estimation_algo 0 for reference
        '''
        estimation_algo=2
        assert self.state[1]
        
        max_model_order = self.max_model_order
        num_block_rows = self.num_block_rows
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels = self.prep_data.num_ref_channels
        sampling_rate = self.prep_data.sampling_rate
        total_time_steps = self.prep_data.total_time_steps
        N = int(total_time_steps - 2*num_block_rows)
        
        R_21 = self.R_21
        R_31 = self.R_31
        R_41 = self.R_41
    
        R_22 = self.R_22
        R_32 = self.R_32
        R_42 = self.R_42

        R_33 = self.R_33
        
        U = self.U
        S = self.S
        S_2 = np.diag(np.power(S,-0.5))
        V_T = self.V_T
        
        O = np.dot(U, np.diag(np.power(S,0.5)))
        
        print('Computing modal parameters...')
        eigenvalues = np.zeros((max_model_order, max_model_order),dtype=np.complex128)
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros((num_analised_channels, max_model_order, max_model_order),dtype=complex)
        modal_contributions = np.zeros((max_model_order, max_model_order))


        for order in range(1,max_model_order):    
            print('.',end='', flush=True) 
            
            V = V_T[:order,:].T
            
            # usually used equation computation of A, C
            
            if estimation_algo == 0:
                
                On_up = O[:num_analised_channels * num_block_rows,:order]
                On_upi = np.linalg.pinv(On_up)
                
                On_down = O[num_analised_channels:num_analised_channels * (num_block_rows+1) ,:order]
                A = np.dot(On_upi, On_down)
                C=O[:num_analised_channels,:order]  
                
            elif estimation_algo == 1:
                #direct computation of A, C, Q, R and S (DeCock 2007)
                
                On_up = O[:num_analised_channels * num_block_rows,:order]
                On_upi = np.linalg.pinv(On_up)
                QSR1 = np.vstack([np.hstack([On_upi.dot(R_41), On_upi.dot(R_42), np.zeros((order,num_analised_channels-num_ref_channels))]),
                                  np.hstack([R_21,           R_22,          np.zeros((num_ref_channels, num_analised_channels-num_ref_channels))]),
                                  np.hstack([R_31,           R_32,            R_33])])
                
                VVT=np.identity(num_ref_channels*num_block_rows)-V.dot(V.T)
        
                QSR2 = np.vstack([np.hstack([VVT, np.zeros((num_ref_channels*num_block_rows, num_analised_channels))]),
                                  np.hstack([np.zeros((num_analised_channels, num_ref_channels*num_block_rows)), np.identity(num_analised_channels)]),])
    
                
                QSR = 1/N*QSR1.dot(QSR2).dot(QSR1.T)
                    
                Q = QSR[:order,:order]
                S = QSR[:order,order:order+num_analised_channels]
                R = QSR[order:order+num_analised_channels,order:order+num_analised_channels]
                AC = np.vstack([On_upi.dot(R_41),R_21,R_31]).dot(V.dot(S_2[:order,:order]))
                
                A=AC[:order,:]
                C=AC[order:,:]
                
            elif estimation_algo == 2:
                # residual-based computation of Q, R  and S (Peeters 1999)
                Q_1_T = self.Q_1_T
                Q_2_T = self.Q_2_T
                Q_3_T = self.Q_3_T
                #Q_4_T = self.Q_4_T    
                 
                P_i_1 = np.hstack((R_41, R_42)).dot(np.vstack((Q_1_T,Q_2_T)))
                O_i_1 = O[:num_analised_channels * num_block_rows,:order]
                O_i = O[:,:order]
                 
                Y_i_i = np.vstack([np.hstack([R_21, R_22, np.zeros((num_ref_channels, num_analised_channels-num_ref_channels))]),
                                  np.hstack([R_31,R_32, R_33])]).dot(np.vstack((Q_1_T, Q_2_T, Q_3_T)))
                 
                P_i_ref = np.vstack((R_21,R_31,R_41)).dot(Q_1_T)
                 
                X_i = np.linalg.pinv(O_i).dot(P_i_ref)
                X_i_1 = np.linalg.pinv(O_i_1).dot(P_i_1)
                 
                X_i_1_Y_i = np.vstack((X_i_1, Y_i_i))
                 
                AC = X_i_1_Y_i.dot(np.linalg.pinv(X_i))
                A= AC[:order,:]
                C= AC[order:,:]
                 
                roh_w_v = X_i_1_Y_i-AC.dot(X_i)
                 
                QSR = roh_w_v.dot(roh_w_v.T)
                     
                Q = QSR[:order,:order]
                S = QSR[:order,order:order+num_analised_channels]
                R = QSR[order:order+num_analised_channels,order:order+num_analised_channels]
            
            
            eigval, eigvec_r = np.linalg.eig(A)
            
            conj_indices = self.remove_conjugates_new(eigval, eigvec_r,inds_only=True)
            
            for i,ind in enumerate(conj_indices):
                
                lambda_i =eigval[ind]
                eigenvalues[order,i]=lambda_i
                #continue
                a_i = np.abs(np.arctan2(np.imag(lambda_i),np.real(lambda_i)))
                b_i = np.log(np.abs(lambda_i))
                freq_i = np.sqrt(a_i**2+b_i**2)*sampling_rate/2/np.pi
                damping_i = 100*np.abs(b_i)/np.sqrt(a_i**2+b_i**2)
                mode_shape_i = np.dot(C, eigvec_r[:,ind])
                mode_shape_i = np.array(mode_shape_i, dtype=complex)
                
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                #mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))                
                        
                k = np.argmax(np.abs(mode_shape_i))
                s_ik = mode_shape_i[k]
                alpha_ik = np.angle(s_ik)
                e_k = np.zeros((num_analised_channels,1))
                e_k[k,0]=1
                mode_shape_i *= np.exp(-1j*alpha_ik)
                
                modal_frequencies[order,i]=freq_i
                modal_damping[order,i]=damping_i
                mode_shapes[:,i,order]=mode_shape_i
                
                
                
            if estimation_algo ==0:
                continue
            
            
            try:
                P = scipy.linalg.solve_discrete_are(a=A.T, b=C.T, q=Q, r=R, s=S, balanced=True)
            except Exception as e:
                print('Correlations of residuals are not symmetric. Skiping Modal Contributions')
                continue
#                 try:
#                     Q = (Q + Q.T)*0.5
#                     R = (R + R.T)*0.5
#                     P = scipy.linalg.solve_discrete_are(a=A.T, b=C.T, q=Q, r=R, s=S, balanced=True)
#                 except Exception as e:
#                     print('Can not estimate Kalman Gain at order {}. Skipping Modal Contributions!'.format(order))
#                     #print(e)
#                     continue
                
                
            APCS = A.dot(P).dot(C.T)+S
            CPCR = C.dot(P).dot(C.T)+R
            K = np.linalg.solve( CPCR.T,APCS.T,).T
            
            
            
            A_0 = np.diag(eigval)
            C_0 = C.dot(eigvec_r)
            K_0 = np.linalg.inv(eigvec_r).dot(K)
            
            j= self.prep_data.measurement.shape[0]
            #j=12000
            states = np.zeros((order+1,j),dtype=np.complex64)

            
            AKC = (A_0-K_0.dot(C_0))
            AKC = np.array(AKC, dtype=np.complex64)

            K_0m = K_0.dot(self.prep_data.measurement[:j,:].T)
            K_0m = np.array(K_0m, dtype = np.complex64)
            global use_cython
            if use_cython:
                states = estimate_states(AKC, K_0m)#@UndefinedVariable
            else:
                
                for k in range(j-1):
     
                    states[:,k+1] = K_0m[:,k] + np.dot(AKC, states[:,k])


            Y = self.prep_data.measurement[:j,:].T
            norm = 1/np.einsum('ji,ji->j', Y,Y)
            
            
            meas_synth_single = []    
            for i,ind in enumerate(conj_indices):
                
                lambda_i =eigval[ind]
                
                ident = eigval == lambda_i.conj()
                ident[ind] = 1                
                ident=np.diag(ident)
                
                C_0I=C_0.dot(ident)
                
                meas_synth = C_0I.dot(states)
                meas_synth = meas_synth.real
                meas_synth_single.append(meas_synth)

                mYT = np.einsum('ji,ji->j', meas_synth,Y)
                
                modal_contributions[order,i] = np.mean(norm*mYT)
                
                
                
            #print(np.sum(modal_contributions[order,:]))
            plot_=False
            if plot_ and order>10:
                print(modal_contributions[order,:][modal_contributions[order,:]!=0],np.sum(modal_contributions[order,:]))
                import matplotlib.pyplot as plot
#                 axes=[]
#                 for i in range(len(conj_indices)):
#                     plot.figure()
#                     plot.plot(meas_synth_single[i][0,500:1000])
#                     ax=plot.gca()
#                     ax.set_xlim((0,500))
#                     ax.set_ylim((-0.0015,0.0015))
#                     ax.set_xticklabels([])
#                     ax.set_yticklabels([])
#                 plot.figure()
#                 plot.plot(self.prep_data.measurement[500:1000,0])
#                 ax=plot.gca()
#                 ax.set_xlim((0,500))
#                 ax.set_ylim((-0.0015,0.0015))
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
#                 plot.show()
                
                fig,axes=plot.subplots(len(conj_indices)+1,2, sharex='col', sharey='col', squeeze=False)
                #print(axes)
                j4=int(np.floor(j/4))
                ft_freq = np.fft.rfftfreq(j4, d = (1/self.prep_data.sampling_rate))    
                meas_synth_all = np.zeros((num_analised_channels,j))
                for (ax1,ax2),meas_synth in zip(axes, meas_synth_single):
                    #print(ax1,ax2, axes)
                    ax1.plot(meas_synth[0,:j])
                    
                    ft = np.fft.rfft(meas_synth[0,:j4] * np.hanning(j4))
                    for i in range(1,4):
                        ft += np.fft.rfft(meas_synth[0,i*j4:(i+1)*j4] * np.hanning(j4))
                    ft /= 4   
                      
                    ax2.plot(ft_freq,np.abs(ft))
                    meas_synth_all+=meas_synth
      
                axes[-1,0].plot(self.prep_data.measurement[0:j,0],alpha=.5)
                axes[-1,0].plot(meas_synth_all[0,:j])
                axes[-1,0].plot(meas_synth_all[0,:j]-self.prep_data.measurement[:j,0],alpha=.25)
                axes[-1,0].set_xlim(0,j)
                #ft_freq = np.fft.rfftfreq(j/4, d = (1/self.prep_data.sampling_rate))
                  
                ft_meas = np.fft.rfft(self.prep_data.measurement[0:j4,0] * np.hanning(j4))
                for i in range(1,4):
                    ft_meas += np.fft.rfft(self.prep_data.measurement[i*j4:(i+1)*j4,0]* np.hanning(j4))
                ft_meas /= 4               
                axes[-1,1].plot(ft_freq,np.abs(ft_meas),alpha=.5)
                  
                ft_synth = np.fft.rfft(meas_synth_all[0,:j4] * np.hanning(j4))
                for i in range(1,4):
                    ft_synth += np.fft.rfft(meas_synth_all[0,i*j4:(i+1)*j4] * np.hanning(j4))
                ft_synth /= 4   
                axes[-1,1].plot(ft_freq,np.abs(ft_synth))
                
                axes[-1,1].set_xlim(0,self.prep_data.sampling_rate/5)
                  
                plot.show()
                
        self.modal_contributions = modal_contributions
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues
            
        self.state[2]=True
        print('.',end='\n', flush=True)  
    
    @staticmethod
    def remove_conjugates_new (eigval, eigvec_r, eigvec_l=None, inds_only=False):
        '''
        finds and removes conjugates
        keeps the second occurance of a conjugate pair (usually the one with the negative imaginary part)
        
        eigvec_l.shape = [order+1, order+1]
        eigval.shape = [order+1,1]
        '''
        #return vectors, eigval
        num_val=len(eigval)
        conj_indices=deque()
        
        for i in range(num_val):
            this_val=eigval[i]
            this_conj_val = np.conj(this_val)
            if this_val == this_conj_val: #remove real eigvals
                #continue
                conj_indices.append(i)
            for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
                if eigval[j] == this_conj_val:

                    conj_indices.append(j)
                    break

        conj_indices=list(set(range(num_val)).difference(conj_indices))
        
        if inds_only:
            return conj_indices
        
        if eigvec_l is None:
            
            eigvec_r = eigvec_r[:,conj_indices]
            eigval = eigval[conj_indices]
    
            return eigval,eigvec_r      
        
        else:             
            eigvec_l = eigvec_l[:,conj_indices]
            eigvec_r = eigvec_r[:,conj_indices]
            eigval = eigval[conj_indices]
    
            return eigval,eigvec_l,eigvec_r      

    
    @staticmethod
    def integrate_quantities(vector, accel_channels, velo_channels, omega):
        # input quantities = [a, v, d]
        # output quantities = [d, d, d]
        # converts amplitude and phase
        #                     phase + 180; magn / omega^2
        
        vector[accel_channels] *= -1       / (omega ** 2)
        #                    phase + 90; magn / omega
        vector[velo_channels] *=  1j        / omega
        
        return vector   
    
    def save_state(self, fname):
        
        print('Saving results to  {}...'.format(fname))
        
        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        out_dict={'self.state':self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time']=self.start_time
        #out_dict['self.prep_data']=self.prep_data
        if self.state[0]:# subspace matrix
            out_dict['self.num_block_rows'] = self.num_block_rows
            out_dict['self.Hankel_matrix'] = self.Hankel_matrix
            out_dict['self.R_21'] = self.R_21
            out_dict['self.R_31'] = self.R_31
            out_dict['self.R_41'] = self.R_41
            out_dict['self.R_22'] = self.R_22
            out_dict['self.R_32'] = self.R_32
            out_dict['self.R_42'] = self.R_42
            out_dict['self.R_33'] = self.R_33
            out_dict['self.Q_1_T'] = self.Q_1_T
            out_dict['self.Q_2_T'] = self.Q_2_T
            out_dict['self.Q_3_T'] = self.Q_3_T
            #out_dict['self.Q_4_T'] = self.Q_4_T
        if self.state[1]:# singular value decomposition / state matrices
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.S'] = self.S
            out_dict['self.U'] = self.U
            out_dict['self.V_T'] = self.V_T              
        if self.state[2]:# modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.modal_contributions'] = self.modal_contributions
            
        np.savez_compressed(fname, **out_dict)

    @classmethod 
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname, allow_pickle=True)    
        #             0         1           2             3
        #self.state= [Hankel, QR_decomp.,  State matr.,  Modal Par.]
        if 'self.state' in in_dict:
            state= list(in_dict['self.state'])
        else:
            return
        

        
        assert isinstance(prep_data, PreprocessData)
        setup_name= str(in_dict['self.setup_name'].item())
        start_time=in_dict['self.start_time'].item()
        assert setup_name == prep_data.setup_name
        start_time = prep_data.start_time
        
        assert start_time == prep_data.start_time
        #prep_data = in_dict['self.prep_data'].item()
        ssi_object = cls(prep_data)

        if state[0]:# subspace matrix
            print('Subspace Matrix Built')
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
            ssi_object.Hankel_matrix= in_dict['self.Hankel_matrix']
            ssi_object.R_21= in_dict['self.R_21']
            ssi_object.R_31= in_dict['self.R_31']
            ssi_object.R_41= in_dict['self.R_41']
            ssi_object.R_22= in_dict['self.R_22']
            ssi_object.R_32= in_dict['self.R_32']
            ssi_object.R_42= in_dict['self.R_42']
            ssi_object.R_33= in_dict['self.R_33']
            if 'self.Q_1_T' in in_dict:
                ssi_object.Q_1_T= in_dict['self.Q_1_T'] 
                ssi_object.Q_2_T= in_dict['self.Q_2_T'] 
                ssi_object.Q_3_T= in_dict['self.Q_3_T'] 
                #ssi_object.Q_4_T= in_dict['self.Q_4_T'] 
        if state[1]:# singular value decomposition / state matrices
            print('State Matrices Computed')
            ssi_object.max_model_order= int(in_dict['self.max_model_order'])
            ssi_object.S= in_dict['self.S']
            ssi_object.U= in_dict['self.U']
            ssi_object.V_T= in_dict['self.V_T']            
        if state[2]:# modal params
            print('Modal Parameters Computed')
            ssi_object.modal_frequencies= in_dict['self.modal_frequencies']
            ssi_object.modal_damping= in_dict['self.modal_damping']
            ssi_object.mode_shapes= in_dict['self.mode_shapes']
            ssi_object.modal_contributions= in_dict['self.modal_contributions']
            if 'self.eigenvalues' in in_dict:
                ssi_object.eigenvalues = in_dict['self.eigenvalues'] 
                
        ssi_object.state = state
        return ssi_object
    
    @staticmethod
    def rescale_mode_shape(modeshape):
        #scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape    
def main():   
    
    pass

if __name__ =='__main__':
    global use_cython 
    try:
        import pyximport
        pyximport.install(setup_args={"include_dirs":np.get_include()})
        from cython_code.cython_helpers import estimate_states#@UnresolvedImport
        use_cython=True
    except:
        print('Not using Cython extensions. Python based state estimation possibly errorneous/untested')
        #global use_cython
        use_cython = False
    main()
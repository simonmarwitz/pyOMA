# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:41:56 2014

@author: volkmar
"""

import numpy as np
from scipy import linalg
#import sys
import os
#import json

import multiprocessing as mp
import ctypes as c
from collections import deque
import datetime
#from copy import deepcopy

from PreprocessingTools import PreprocessData

#from decomp_qr_VZ import qr
#####################################################################################################################################


    
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
        
        # Extract reference time series 
        extract_length = int(total_time_steps - (num_block_rows) + 1)
               
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        #print(all_channels)
        
                     
        if (num_ref_channels < num_analised_channels):
            
            refs = (measurement[0:extract_length,ref_channels])
                     
        else:
            refs = measurement

       
           
        ###############################################################################
        ######## Create transpose of the block Hankel matrix [Y_(0|2i-1)]^T ###########
        ###############################################################################
        
        print('Creating block Hankel matrix...')
        
        i = num_block_rows
        j = total_time_steps - 2*i
        
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
        
        shape = Hankel_matrix_T.shape
        print('Hankel shape = ', shape)

        Q, R = linalg.qr(Hankel_matrix_T)
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
        
        Q_1 = Q[0:(num_ref_channels*i),:]
        #Q_12 = Q[0:(num_ref_channels*(i+1)),:]
        #Q_123 = Q[0:(num_ref_channels*(i+1)+num_roving_channels),:]
        
        
        P_i_ref = R[(num_ref_channels*i):((num_ref_channels + num_analised_channels) * i),0:(num_ref_channels*i)]
        P_i_ref = np.dot(P_i_ref, Q_1)
         
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
        
        print('max_model_order = ', max_model_order)
        print('self.max_model_order = ', self.max_model_order)

        
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
            manager=mp.Manager()        
            pool = mp.Pool()       
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

    #return

    
def main():
    pass

if __name__ =='__main__':
    main()
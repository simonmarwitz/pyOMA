# -*- coding: utf-8 -*-
'''
Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''

import numpy as np
import scipy.linalg 
#import sys
import os
#import json

import multiprocessing as mp
import ctypes as c
from collections import deque
import datetime
#from copy import deepcopy

from PreprocessingTools import PreprocessData
#from StabilDiagram import main_stabil, StabilPlot, nearly_equal

#import pydevd
'''
TODO:
- change channels numbers such, that user input channels start at 1 while internally they start at 0
    affects: ref_channels, roving_channels and channel-dof-assignments
- generally define unit tests to check functionality after changes
- 

'''
    
class BRSSICovRef(object):
    
    def __init__(self,prep_data):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_data, PreprocessData)
        self.prep_data =prep_data
        self.setup_name = prep_data.setup_name
        self.start_time = prep_data.start_time
        #             0         1           2         
        #self.state= [Toeplitz, State Mat., Modal Par.
        self.state  =[False,    False,      False]
        
        self.num_block_columns = None
        self.num_block_rows = None
        self.toeplitz_matrix = None
        
        self.max_model_order = None
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
            
            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Columns:'
            num_block_columns = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Maximum Model Order:'
            max_model_order= int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Use Multiprocessing:'
            multiprocessing= f.__next__().strip('\n').strip(' ')=='yes'
            
        ssi_object = cls(prep_data)
        ssi_object.build_toeplitz_cov(num_block_columns, multiprocess=multiprocessing)
        ssi_object.compute_state_matrices(max_model_order)
        ssi_object.compute_modal_params(multiprocessing)
        
        return ssi_object
        
    def build_toeplitz_cov(self, num_block_columns, num_block_rows=None, multiprocess=True):
        '''
        Builds a Block-Toeplitz Matrix of Covariances with varying time lags
            | <- num_block_columns*num_ref_channels-> |_
            |     R_i      R_i-1      ...      R_1    |^
            |     R_i+1    R_i        ...      R_2    |num_block_rows*num_analised_channels
            |     ...      ...        ...      ...    |v
            |     R_2i-1   ...        ...      R_i    |_
        '''
        #print(multiprocess)
        assert isinstance(num_block_columns, int)
        if num_block_rows is None:
            num_block_rows=num_block_columns+1
        assert isinstance(num_block_rows, int)
        
        self.num_block_columns=num_block_columns
        self.num_block_rows=num_block_rows
        total_time_steps = self.prep_data.total_time_steps
        ref_channels = sorted(self.prep_data.ref_channels)
        roving_channels = self.prep_data.roving_channels
        measurement = self.prep_data.measurement
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels 
        # Extract reference time series for covariances 
        extract_length = int(total_time_steps - (num_block_columns + num_block_rows) + 1)
        
        
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        #print(all_channels)
        
        refs = (measurement[0:extract_length,ref_channels]).T
        
        ### Create Toeplitz matrix and fill it with covariances 
        #    |    R_i    R_i-1    ...    R_0    |                                     #
        #    |    R_i+1  R_i      ...    R_1    |                                     #
        #    |    ...    ...      ...    ...    |                                     #
        #    |    R_2i-1 ...      ...    R_i    |                                     #
        
        print('Computing covariances...')
        n, m = num_analised_channels*num_block_rows, num_ref_channels*num_block_columns
        
        if multiprocess:
            toeplitz_memory = mp.Array(c.c_double, np.zeros(n*m)) # shared memory, can be used by multiple processes @UndefinedVariable
            toeplitz_shape = (n,m)
            measurement_shape=measurement.shape
            measurement_memory = mp.Array(c.c_double, measurement.reshape(measurement.size, 1)) # @UndefinedVariable
            
            refs_memory = mp.Array(c.c_double, refs.reshape(refs.size,1 ))# @UndefinedVariable
            refs_shape=refs.shape
            
            pool=mp.Pool(initializer=self.init_child_process, initargs=(refs_memory,  measurement_memory, toeplitz_memory, ))# @UndefinedVariable
        
            
            for i in range(0,num_block_rows):
                if i == 0:
                    # in multiprocessing, errors in the single processes are sometimes not reported to console, in this case 
                    # the resulting block toeplitzmatrix will be filled with zeros
                    # try to change pool.apply_async to pool.apply and run in debug mode
                    for ii in range(0,self.num_block_columns):#fill the first block row
                        pool.apply_async(self.compute_covariance , args=(i, 
                                                            ii, 
                                                            num_block_columns, 
                                                            extract_length, 
                                                            ref_channels, 
                                                            all_channels, 
                                                            refs_shape, 
                                                            measurement_shape,
                                                            toeplitz_shape))
                else: #fill the first block column
                    ii = 0
                    pool.apply_async(self.compute_covariance , args=(i, 
                                                            ii, 
                                                            num_block_columns, 
                                                            extract_length, 
                                                            ref_channels, 
                                                            all_channels, 
                                                            refs_shape, 
                                                            measurement_shape,
                                                            toeplitz_shape))
                        
            pool.close()
            pool.join()               
            
            del measurement_memory
            del refs_memory
            del refs  

      
            Toeplitz_matrix = np.frombuffer(toeplitz_memory.get_obj()).reshape((n,m)) 

            
            for i in range(1,num_block_rows): #finish assembling block toeplitz matrix
                # copys and shifts contents from previous block row to next block row
                # shifts by one block column to the right
                previous_Toeplitz_row = (i-1)*num_analised_channels
                this_block = Toeplitz_matrix[previous_Toeplitz_row:(previous_Toeplitz_row+num_analised_channels),
                                             0:(num_ref_channels * num_block_columns - num_ref_channels)]
                begin_Toeplitz_row = i*num_analised_channels
                Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                num_ref_channels:(num_ref_channels * num_block_columns)] = this_block
    
        else: # old single threaded way
            
            Toeplitz_matrix = np.zeros((n,m))

            for i in range(0,num_block_rows):
                if i == 0:
                    for ii in range(0,num_block_columns):
                        
                        begin_extract = num_block_columns + i - (ii)
                        current_signals = measurement[begin_extract : (begin_extract+extract_length), all_channels].T
                        this_block = (np.dot(refs, current_signals.T.conj()) / refs.shape[1]).T/extract_length
                        #print(this_block)
                        #covariances = np.cov(refs,current_signals)
                        #this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
                        
                        begin_Toeplitz_row = i*num_analised_channels
                        
                        Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                        ii*num_ref_channels:(ii*num_ref_channels+num_ref_channels)] = this_block
                else: 
                    previous_Toeplitz_row = (i-1)*num_analised_channels
                    this_block = Toeplitz_matrix[previous_Toeplitz_row:(previous_Toeplitz_row+num_analised_channels),
                                                  0:num_ref_channels * (num_block_columns-1)]
                    begin_Toeplitz_row = i*num_analised_channels
                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                     num_ref_channels:(num_ref_channels * num_block_columns)] = this_block
                     
                    begin_extract = num_block_columns + i
                    current_signals = (measurement[begin_extract:(begin_extract + extract_length),all_channels]).T
                    
                    this_block = (np.dot(refs, current_signals.T.conj()) / refs.shape[1]).T/extract_length
                    #print(this_block)
                    #covariances = np.cov(refs,current_signals)
                    #this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
                     
                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
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
#             #plot.errorbar(range(num_block_rows+num_block_rows-1), means, yerr=np.sqrt(sigma_r[inds,inds]))
#             #print(np.sqrt(sigma_r[inds,inds]))
#                 
#             #plot.plot(vec_R[inds,0])
#             #plot.plot(vec_R[inds,1])
#             plot.plot(range(1,num_block_columns+num_block_rows), means)
#         plot.show()
        
        self.toeplitz_matrix = Toeplitz_matrix              
        self.state[0]=True
         
        
    def init_child_process(self, refs_memory_, measurement_memory_, toeplitz_memory_):
        #make the  memory arrays available to the child processes
        global refs_memory
        refs_memory = refs_memory_
        
        global measurement_memory
        measurement_memory = measurement_memory_   
        
        global toeplitz_memory
        toeplitz_memory = toeplitz_memory_
    
   
    def compute_covariance(self, i,ii, num_block_columns, extract_length, ref_channels, all_channels, refs_shape, measurement_shape, toeplitz_shape):
        
        num_analised_channels = len(all_channels)
        num_ref_channels = len(ref_channels)
        
        measurement = np.frombuffer(measurement_memory.get_obj()).reshape(measurement_shape)
        refs = np.frombuffer(refs_memory.get_obj()).reshape(refs_shape)
        toeplitz = np.frombuffer(toeplitz_memory.get_obj()).reshape(toeplitz_shape)
            
        begin_extract = num_block_columns + i - (ii) 

        current_signals = (measurement[begin_extract:(begin_extract + extract_length), all_channels]).T

        this_block = (np.dot(refs, current_signals.T.conj())).T/extract_length
        
        
        begin_toeplitz_row = i*num_analised_channels 
        end_toeplitz_row = begin_toeplitz_row+num_analised_channels 
        begin_toeplitz_col = ii*num_ref_channels
        end_toeplitz_col = begin_toeplitz_col + num_ref_channels
        
        with toeplitz_memory.get_lock():
    
            toeplitz[begin_toeplitz_row: end_toeplitz_row,
                            begin_toeplitz_col:end_toeplitz_col] = this_block
          
    def compute_state_matrices(self, max_model_order=None, max_modes=None):
        '''
        computes the state and output matrix of the state-space-model
        by applying a singular value decomposition to the block-toeplitz-matrix of covariances
        the state space model matrices are obtained by appropriate truncation 
        of the svd matrices at max_model_order
        the decision whether to take merged covariances is taken automatically
        max_modes is the number of expected modes for estimation of the state transition /
        system matrix A of the "crystal clear (TM)" algorithm of structural vibrations AS (Artemis)
        '''
        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
            
        assert self.state[0]
        

        toeplitz_matrix = self.toeplitz_matrix
        num_channels = self.prep_data.num_analised_channels
        num_block_columns = self.num_block_columns
        num_block_rows = self.num_block_rows
        print('Computing state matrices...')
        #print(toeplitz_matrix.shape, np.linalg.matrix_rank(toeplitz_matrix))
        [U,S,V_T] = np.linalg.svd(toeplitz_matrix,0)
        #print(U.shape, S.shape, V_T.shape)
    
        # choose highest possible model order
        if max_model_order is None:
            max_model_order=len(S)
        else:
            max_model_order = min(max_model_order,len(S))
    
        S_2 = np.diag(np.power(S[:max_model_order], 0.5))
        U = U[:,:max_model_order]
        Oi = np.dot(U, S_2)
        C = Oi[:num_channels,:]   
        
        Oi0=Oi[:(num_channels * (num_block_rows - 1)),:]
        Oi1=Oi[num_channels:(num_channels * num_block_rows),:]
        
        if max_modes is not None:
            [u,s,v_t]=np.linalg.svd(Oi0,0)
            s = 1./s[:max_modes]
            Oi0p= np.dot(np.transpose(v_t[:max_modes,:]), np.multiply(s[:, np.newaxis], np.transpose(u[:,:max_modes])))
        else:
            Oi0p = np.linalg.pinv(Oi0, rcond=1e-12)
            
        A = np.dot(Oi0p,Oi1)
        self.Oi = Oi
        self.state_matrix = A
        self.output_matrix = C
        self.max_model_order=max_model_order
        
        self.state[1]=True
        self.state[2] = False # previous modal params are invalid now
        

    def compute_modal_params(self, multiprocessing=True, max_model_order=None): 
        
        if max_model_order is not None:
            assert max_model_order<=self.max_model_order
            self.max_model_order=max_model_order
        
        assert self.state[1]
        multiprocessing=False
        print('Computing modal parameters...')
        max_model_order = self.max_model_order
        num_analised_channels = self.prep_data.num_analised_channels
        num_block_rows = self.num_block_rows
        #state_matrix = self.state_matrix
        Oi = self.Oi
        output_matrix = self.output_matrix
        sampling_rate = self.prep_data.sampling_rate
        
        #b_accel_channels = np.array([False for i in range(num_analised_channels)])
        accel_channels=self.prep_data.accel_channels
        #b_velo_channels = np.array([False for i in range(num_analised_channels)])
        velo_channels=self.prep_data.velo_channels
        
        modal_frequencies = np.zeros((max_model_order, max_model_order))        
        modal_damping = np.zeros((max_model_order, max_model_order))          
        eigenvalues = np.zeros((max_model_order, max_model_order), dtype=complex)        
        mode_shapes = np.zeros((num_analised_channels, max_model_order, max_model_order),dtype=complex)
            
        if multiprocessing:
            manager=mp.Manager()#@UndefinedVariable    
            pool = mp.Pool()  #@UndefinedVariable         
            return_dict=manager.dict()
            
            #balanced allocation of work (numpy.linalg.eig takes approx. n^3 operations)
            work_slice_size = sum([n**3 for n in range(max_model_order)])/max_model_order
            current_size = 0
            current_orders = []
            for order in range(0,max_model_order,1):
                current_orders.append(order)
                current_size += order**3
                if current_size >= work_slice_size:
                    pool.apply_async(self.multiprocess_evd , args=(Oi[:,:order], current_orders, return_dict, num_analised_channels, num_block_rows))
                    current_orders = []
                    current_size = 0
            pool.close()
            pool.join()
            
#         import matplotlib.pyplot as plot
#         fig=plot.figure()
#         ax1=fig.add_subplot(111)
#         fig=plot.figure()
#         ax2=fig.add_subplot(111)
        for order in range(1,max_model_order,1):    
            
            if multiprocessing:
                eigenvalues_single, eigenvectors_single = return_dict[order]
            else:
                #eigenvalues_paired, eigenvectors_paired = np.linalg.eig(state_matrix[0:order, 0:order])
                Oi0=Oi[:(num_analised_channels * (num_block_rows - 1)),:order]
                Oi1=Oi[num_analised_channels:(num_analised_channels * num_block_rows),:order]
                    
                a = np.dot(np.linalg.pinv(Oi0), Oi1)  
                eigenvalues_paired, eigvec_l, eigenvectors_paired = scipy.linalg.eig(a=a[0:order, 0:order],b=None,left=True,right=True)
                
                eigenvalues_single,eigenvectors_single = \
                    self.remove_conjugates_new(eigenvalues_paired,eigenvectors_paired)
#                 ax1.plot(eigenvalues_single.real,eigenvalues_single.imag, ls='', marker='o')
                
            #lambdas=[]
            for index,k in enumerate(eigenvalues_single): 
                
                lambda_k = np.log(complex(k)) * sampling_rate
                #lambdas.append(lambda_k)
                freq_j = np.abs(lambda_k) / (2*np.pi)        
                damping_j = np.real(lambda_k)/np.abs(lambda_k) * (-100)  
                mode_shapes_j = np.dot(output_matrix[:, 0:order], eigenvectors_single[:,index])
            
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))                
                        
                modal_frequencies[order,index]=freq_j
                modal_damping[order,index]=damping_j
                eigenvalues[order,index]=k
                mode_shapes[:,index,order]=mode_shapes_j
            #lambdas = np.array(lambdas)
#             ax2.plot(lambdas.real,lambdas.imag, ls='', marker='o')
#         
#         ax1.axvline(x=0)
#         ax1.axhline(y=0)
#         ax2.axvline(x=0)
#         ax2.axhline(y=0)
#                 
#         xlim=np.max(np.abs(ax1.get_xlim()))
#         ax1.set_xlim((-xlim,xlim))
#         ylim=np.max(np.abs(ax1.get_ylim()))
#         ax1.set_ylim((-ylim,ylim))
#         
#         plot.show()
        
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues
            
        self.state[2]=True
        
    
    def multiprocess_evd(self, Oi, truncation_orders, return_dict, num_channels, num_block_rows):
        
        Oi0=Oi[:(num_channels * (num_block_rows - 1)),:]
        Oi1=Oi[num_channels:(num_channels * num_block_rows),:]
            
        a = np.dot(np.linalg.pinv(Oi0), Oi1)  
        for truncation_order in truncation_orders:
            eigenvalues_paired, eigvec_l, eigenvectors_paired = scipy.linalg.eig(a=a[0:truncation_order, 0:truncation_order],b=None,left=True,right=True)
    
            eigenvalues_single,eigenvectors_single = \
                    BRSSICovRef.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            return_dict[truncation_order] = (eigenvalues_single, eigenvectors_single)
        
        return
    
    @staticmethod
    def remove_conjugates_new (eigval, eigvec_r, eigvec_l=None):
        '''
        removes conjugates
        
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
                conj_indices.append(i)
            for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
                if eigval[j] == this_conj_val:
        
                    #if not np.allclose(eigvec_l[j],eigvec_l[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_l is not')
                    #    continue
        
                    #if not np.allclose(eigvec_r[j],eigvec_r[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_r is not')
                    #    continue
                    
                    conj_indices.append(j)
                    break
                
        #print('indices of complex conjugate: {}'.format(conj_indices))
        conj_indices=list(set(range(num_val)).difference(conj_indices))
        #print('indices to keep and return: {}'.format(conj_indices))
        
        if eigvec_l is None:
            
            eigvec_r = eigvec_r[:,conj_indices]
            eigval = eigval[conj_indices]
    
            return eigval,eigvec_r      
        
        else:             
            eigvec_l = eigvec_l[:,conj_indices]
            eigvec_r = eigvec_r[:,conj_indices]
            eigval = eigval[conj_indices]
    
            return eigval,eigvec_l,eigvec_r      
                  
#     @staticmethod
#     def remove_conjugates_new (vectors, values):
#         '''
#         removes conjugates and marks the vectors which appear in pairs
#         
#         vectors.shape = [order+1, order+1]
#         values.shape = [order+1,1]
#         '''
#         num_val=vectors.shape[1]
#         conj_indices=deque()
#         
#         for i in range(num_val):
#             this_vec=vectors[:,i]
#             this_conj_vec = np.conj(this_vec)
#             this_val=values[i]
#             this_conj_val = np.conj(this_val)
#             if this_val == this_conj_val: #remove real eigenvalues
#                 continue
#             for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
#                 if vectors[0,j] == this_conj_vec[0] and \
#                    vectors[-1,j] == this_conj_vec[-1] and \
#                    values[j] == this_conj_val:
#                     # saves computation time this function gets called many times and 
#                     #numpy's np.all() function causes a lot of computation time
#                     conj_indices.append(i)
#                     break
#         conj_indices=list(conj_indices)
#         vector = vectors[:,conj_indices]
#         value = values[conj_indices]
# 
#         return vector,value
    
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
            
        #             0         1           2           
        #self.state= [Toeplitz, State Mat., Modal Par.]
        out_dict={'self.state':self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time']=self.start_time
        #out_dict['self.prep_data']=self.prep_data
        if self.state[0]:# covariances
            out_dict['self.toeplitz_matrix'] = self.toeplitz_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[1]:# state models
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.state_matrix'] = self.state_matrix
            out_dict['self.output_matrix'] = self.output_matrix
        if self.state[2]:# modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.eigenvalues'] = self.eigenvalues
            
        np.savez_compressed(fname, **out_dict)
        
    @classmethod 
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname)    
        #             0         1           2          
        #self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state= list(in_dict['self.state'])
        else:
            return
        
        for this_state, state_string in zip(state, ['Covariance Matrices Built',
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
        if state[0]:# covariances
            ssi_object.toeplitz_matrix = in_dict['self.toeplitz_matrix']
            ssi_object.num_block_columns = int(in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
        if state[1]:# state models
            ssi_object.max_model_order = int(in_dict['self.max_model_order'])
            ssi_object.state_matrix= in_dict['self.state_matrix']
            ssi_object.output_matrix = in_dict['self.output_matrix']
        if state[2]:# modal params
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']
            ssi_object.eigenvalues = in_dict['self.eigenvalues']
        
        return ssi_object
    
    @staticmethod
    def rescale_mode_shape(modeshape):
        #scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape
    
class PogerSSICovRef(object):
    
    def __init__(self,):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        
        #             0         1           2         
        #self.state= [Toeplitz, State Mat., Modal Par.
        self.state  =[False,    False,      False]
        
        self.setup_name = 'merged_'
        self.start_times = []
        
        self.setups = []
        self.sampling_rate = None
        self.num_ref_channels = None
        
        self.ssi_ref_channels = None
        self.merged_chan_dofs = None
        
        self.subspace_matrix = None
        self.num_analised_channels = None
        self.num_block_columns = None
        self.num_block_rows = None       
    
        
        self.max_model_order = None
        self.state_matrix = None
        self.output_matrix = None
        
        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None
        
        
            
    def add_setup(self, prep_data):
        '''
        todo: 
        check that ref_channels are equal in each setup (by number and by DOF)
        check that covariances were computed with equal tau_max
        '''
        assert isinstance(prep_data, PreprocessData)
      
        # assure chan_dofs were assigned
        assert prep_data.chan_dofs
        
        if self.sampling_rate is not None:
            assert prep_data.sampling_rate == self.sampling_rate
        else:
            self.sampling_rate = prep_data.sampling_rate
        
        if self.num_ref_channels is not None:
            assert self.num_ref_channels == prep_data.num_ref_channels
        else:
            self.num_ref_channels = prep_data.num_ref_channels
        
        self.setup_name +=prep_data.setup_name
        self.start_times.append(prep_data.start_time)
        # extract needed information and store them in a dictionary
        self.setups.append({'prep_data':prep_data
                           })

         
        print('Added setup "{}" with {} channels'.format(prep_data.setup_name, prep_data.num_analised_channels))
        
        self.state[0] = True  
        
    
    def pair_channels(self, ):
        '''
        pairs channels from all given setups for the poger merging methods
        
        ssi_reference channels are common to all setups
        rescale reference channels are common to at least two setups        
        
        finds common dofs from all setups and their respective channels
        generates new channel_dof_assignments with ascending channel numbers
        rescale reference channels are assumed to be equal to ssi_reference channels
        '''
        
        print('pairing channels and dofs')
        setups = self.setups
        merged_chan_dofs = []
        merged_accel_channels = []
        merged_velo_channels = []
        merged_disp_channels = []
        #extract dofs from each setup
        for setup in setups:    
            prep_data = setup['prep_data']

            chan_dofs = []
            accel_channels = []
            velo_channels = []
            disp_channels = []
            prep_data.chan_dofs.sort(key=lambda x: x[0])# chan dofs are now sorted by channel number
            channel = 0
            for chan_dof in prep_data.chan_dofs:
                if channel == chan_dof[0]:
                    chan_dofs.append(chan_dof[1:4])
                else: 
                    chan_dofs.append(None)
                    
                accel_channels.append(channel in prep_data.accel_channels)
                velo_channels.append(channel in prep_data.velo_channels)
                disp_channels.append(channel in prep_data.disp_channels)
                
                channel += 1
            merged_chan_dofs.append(chan_dofs)
            
            merged_accel_channels.append(accel_channels)
            merged_velo_channels.append(velo_channels)
            merged_disp_channels.append(disp_channels)
            
        #print(merged_chan_dofs)
        # find dofs common to all setups
        import copy
        ssi_ref_dofs = copy.deepcopy(merged_chan_dofs[0])        
        for chan_dofs in merged_chan_dofs[1:]:
            new_ref_dofs = []
            for node,az,elev in chan_dofs:
                for rnode, raz,relev in ssi_ref_dofs:
                    if node == rnode and az == raz and elev == relev:
                        new_ref_dofs.append((rnode,raz,relev))
                        break
            ssi_ref_dofs = new_ref_dofs
            if len(ssi_ref_dofs) == 0:
                break
        
        # find channels to the respective common dofs
        ssi_ref_channels = []
        for setup_num,(setup, chan_dofs) in enumerate(zip(setups, merged_chan_dofs)):
            prep_data = setup['prep_data']
            ref_channels = []
            for rnode,raz,relev in ssi_ref_dofs:
                index = None
                for i,(node,az,elev) in enumerate(chan_dofs):
                    if node == rnode and az == raz and elev == relev:
                        index = i
                        break
                else:
                    raise
                #chan_dofs.index(ref_dof)
                channel = prep_data.chan_dofs[index][0]
                # if it is not in ref_channels, covariances have to be regenerated, thus we can break here
                assert channel in prep_data.ref_channels
                ref_channels.append(int(channel))
            ssi_ref_channels.append(ref_channels)
        
            
        # delete channels of the reference dofs 
        for setup, chan_dofs,accel_channels, velo_channels, disp_channels in zip(setups[1:], merged_chan_dofs[1:], merged_accel_channels[1:],merged_velo_channels[1:],merged_disp_channels[1:]):
            prep_data = setup['prep_data']
            for rnode,raz,relev in ssi_ref_dofs:
                index = None
                for i,(node,az,elev) in enumerate(chan_dofs):
                    if node == rnode and az == raz and elev == relev:
                        index = i
                        break
                else:
                    raise
                # remove the channel_dof_assignment of the reference channels for all setups 
                del chan_dofs[index]
                del accel_channels[index]
                del velo_channels[index]                
                del disp_channels[index]
                    
        #flatten chan_dofs and add ascending channel numbers
        flattened = []
        channel = 0
        for sublist in merged_chan_dofs:
            for val in sublist:
                val.insert(0,channel)
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
        
        
        print(merged_accel_channels)
        print(merged_velo_channels)
        print(merged_chan_dofs)
        
        
        self.merged_accel_channels = merged_accel_channels
        self.merged_velo_channels = merged_velo_channels
        self.merged_disp_channels = merged_disp_channels
        
        self.ssi_ref_channels = ssi_ref_channels
        self.merged_chan_dofs = merged_chan_dofs
        self.merged_num_channels = len(merged_chan_dofs)
        
        self.start_time = min(self.start_times)
        
        
        return ssi_ref_channels, merged_chan_dofs

        
    def build_merged_subspace_matrix(self, num_block_columns, num_block_rows=None):
        '''
        Builds a Block-HankelMatrix of Covariances with varying time lags
            | <- num_block_columns*num_ref_channels->|_
            |     R_1      R_2      ...      R_i     |^
            |     R_2      R_3      ...      R_2     |num_block_rows*(num_num_ref_channels*num_setups)
            |     ...      ...      ...      ...     |v
            |     R_i      ...      ...      R_2i-1  |_
            
            
            R_1 =   | R_1^1          |
                    | R_1^2          |
                    | ...            |
                    | R_1^num_setups |
        '''

        assert isinstance(num_block_columns, int)
        if num_block_rows is None:
            num_block_rows=num_block_columns#-10
        assert isinstance(num_block_rows, int)
        setups = self.setups
        
        #num_block_rows = self.num_block_rows
        num_analised_channels = sum([setup['prep_data'].num_analised_channels for setup in setups])
        #num_block_columns = self.num_block_columns
        ssi_ref_channels = self.ssi_ref_channels
        num_ref_channels = len(ssi_ref_channels[0])
        num_setups = len(setups)
        
        subspace_matrix= np.zeros(((num_block_rows+1)*num_analised_channels, num_block_columns*num_ref_channels))
        for block_row in range(num_block_rows+1):
            sum_analised_channels=0
            for this_num_setup, setup in enumerate(setups):
                
                this_analised_channels = setup['prep_data'].num_analised_channels
                ref_inds = np.array([setup['prep_data'].ref_channels.index(ref_channel) for ref_channel in ssi_ref_channels[this_num_setup]])
                
                # shape (num_analised_channels, tau_max * num_ref_channels)
                this_block_column = setup['prep_data'].corr_mats_mean[:,block_row*num_ref_channels:(num_block_columns+block_row)*num_ref_channels]
                
                #ref_inds = np.hstack([ref_inds+block_column*this_analised_channels for block_column in range(num_block_columns)])
                #print(this_block_column.shape, ref_inds,ref_inds.shape, subspace_matrix.shape)
                #shape (num_block_rows+1)*num_analised_channels,num_block_columns*num_ref_channels
                subspace_matrix[block_row*num_analised_channels+sum_analised_channels:block_row*num_analised_channels+sum_analised_channels+this_analised_channels,:]=this_block_column
                
                sum_analised_channels += this_analised_channels
        #print(np.where(subspace_matrix==0))
            # block_row    0                                            1
            # setup 0: zeile 0*this_analised_channels ... 1*this_analised_channels,   3*this_analised_channels ... 4*this_analised_channels
            # setup 1: zeile 1*this_analised_channels ... 2*this_analised_channels,   4*this_analised_channels ... 5*this_analised_channels
            # setup 2: zeile 2*this_analised_channels ... 3*this_analised_channels,   5*this_analised_channels ... 6*this_analised_channels  
            # (bc*num_setups+setup)*num_ref_channels
        self.subspace_matrix = subspace_matrix
        self.num_analised_channels = num_analised_channels
        self.num_block_rows = num_block_rows
        self.num_block_columns = num_block_columns
        
     
          
    def compute_state_matrices(self, max_model_order=None, max_modes=None):

        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)        

        subspace_matrix = self.subspace_matrix
        num_channels = self.num_analised_channels
        num_block_rows = self.num_block_rows # p
        print('Computing state matrices with pinv-based method...')
        
        #[U,S,V_T] = np.linalg.svd(subspace_matrix,1)
        [U,S,V_T] = scipy.linalg.svd(subspace_matrix,1)
        #[U,S,V_T] = scipy.sparse.linalg.svds(subspace_matrix,k=max_model_order)

        #print(S.shape)
        # choose highest possible model order
        if max_model_order is None:
            max_model_order=len(S)        
        else:
            max_model_order = min(max_model_order,len(S))

        #print(S.shape)
        #S_2 = np.diag(np.power(np.copy(S)[:max_model_order], 0.5))
        #print(U.shape)
        U = U[:,:max_model_order]
        V_T = V_T[:max_model_order,:]     
        
        self.U = U
        self.S = S
        self.V_T = V_T
        
        self.max_model_order=max_model_order
        
        self.state[1] = True
        

    def compute_modal_params(self, multiprocessing=True, max_model_order=None): 
        
        max_model_order = self.max_model_order
        num_block_rows = self.num_block_rows
        num_analised_channels = self.num_analised_channels
        merged_num_channels = self.merged_num_channels
        num_ref_channels = self.num_ref_channels
        sampling_rate = self.sampling_rate
        
        U = self.U
        S = self.S
        S_2 = np.diag(np.power(S[:max_model_order],-0.5))
        V_T = self.V_T
        
        O = np.dot(U, S_2)
        
        print('Computing modal parameters...')
    
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros((merged_num_channels, max_model_order, max_model_order),dtype=complex)
        eigenvalues = np.zeros((max_model_order, max_model_order), dtype=complex)


        for order in range(1,max_model_order):    
            print('.',end='', flush=True) 
            
            V = V_T[:order,:].T
            
            On_up = O[:num_analised_channels * num_block_rows,:order]
            
            On_down = O[num_analised_channels:num_analised_channels * (num_block_rows+1) ,:order]
            state_matrix = np.dot(np.linalg.pinv(On_up), On_down)             
    
            C=O[:num_analised_channels,:order]      
            
            
            eigval, eigvec_r = np.linalg.eig(state_matrix)
                     
            
            conj_indices = self.remove_conjugates_new(eigval, eigvec_r,inds_only=True)

            for i,ind in enumerate(conj_indices):
                
                lambda_i =eigval[ind]
                
                ident = eigval == lambda_i.conj()
                ident[ind] = 1                
                ident=np.diag(ident)
                
                                
                a_i = np.abs(np.arctan2(np.imag(lambda_i),np.real(lambda_i)))
                b_i = np.log(np.abs(lambda_i))
                freq_i = np.sqrt(a_i**2+b_i**2)*sampling_rate/2/np.pi
                damping_i = 100*np.abs(b_i)/np.sqrt(a_i**2+b_i**2)
                mode_shape_i = np.dot(C, eigvec_r[:,ind])
                mode_shape_i = np.array(mode_shape_i, dtype=complex)
                
                mode_shape_i = self.rescale_by_references(mode_shape_i)
                
                mode_shape_i = self.integrate_quantities(mode_shape_i, self.merged_accel_channels, self.merged_velo_channels, freq_i*2*np.pi)  
    
                        
                k = np.argmax(np.abs(mode_shape_i))
                s_ik = mode_shape_i[k]
                alpha_ik = np.angle(s_ik)
                e_k = np.zeros((num_analised_channels,1))
                e_k[k,0]=1
                mode_shape_i *= np.exp(-1j*alpha_ik)
                
                modal_frequencies[order,i]=freq_i
                modal_damping[order,i]=damping_i
                mode_shapes[:,i,order]=mode_shape_i    
                eigenvalues[order,i]=lambda_i
                
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes   
        self.eigenvalues = eigenvalues 

    def rescale_by_references(self, mode_shape):
        '''
        This is PoGer Rescaling
        not implemented yet, only a dummy method
        self.rescale_references 
            = [channel_pairs_1, channel_pairs_2,...]
                 1,2,... num_instance
             channel_pairs = [(ref_chan_base, ref_chan_this),..., (None, rov_chan_this),...]
        channel_pairs have been determined by comparison of channel-dof-assignments
        note: reference channels for SSI need not necessarily be reference channels for rescaling and vice versa
        '''
        
        new_mode_shape = np.zeros((self.merged_num_channels), dtype=complex)
        
               
        start_row_scaled = 0
        end_row_scaled = self.setups[0]['prep_data'].num_analised_channels
        
        row_unscaled = self.setups[0]['prep_data'].num_analised_channels
        
        new_mode_shape[start_row_scaled:end_row_scaled] = mode_shape[start_row_scaled:end_row_scaled]
        
        #for ref_channel_pairs, rov_channel_pairs in zip(self.rescale_ref_channels, self.rescale_rov_channels):
        for setup_num,setup in enumerate(self.setups):
            if setup_num == 0: continue
            #ssi_ref_channels is ref_channels with respect to setup not to merged mode shape 
            
            base_refs = self.ssi_ref_channels[0]# [pair[0] for pair in ref_channel_pairs]
            
            this_refs = self.ssi_ref_channels[setup_num]#[pair[1] for pair in ref_channel_pairs]
            this_all = setup['prep_data'].roving_channels+setup['prep_data'].ref_channels
            this_rovs = list(set(this_all).difference(this_refs)) #[pair[0] for pair in rov_channel_pairs]
            
            this_refs = [int(ref+row_unscaled) for ref in this_refs]
            this_rovs = [rov+row_unscaled for rov in this_rovs]
            
            #print(base_refs, this_refs, this_rovs)
            mode_refs_base = mode_shape[base_refs]
            mode_refs_this = mode_shape[this_refs]
            mode_refs_this_conj = mode_refs_this.conj()
            mode_rovs_this = mode_shape[this_rovs]
            
            numer = np.inner(mode_refs_this_conj, mode_refs_base )
            denom = np.inner(mode_refs_this_conj, mode_refs_this )
            scale_fact=numer/denom    
                
            start_row_scaled = end_row_scaled
            end_row_scaled += len(this_rovs)
            
            new_mode_shape[start_row_scaled:end_row_scaled] = scale_fact * mode_rovs_this
            
            row_unscaled += setup['prep_data'].num_analised_channels
            
        #print( np.where(new_mode_shape==0))
        return new_mode_shape    

    @staticmethod
    def remove_conjugates_new (eigval, eigvec_r, eigvec_l=None, inds_only=False):
        '''
        finds and removes conjugates
        keeps the second occurance of a conjugate pair (usually the one with the negative imaginary part)
        
        eigvec_l.shape = [order+1, order+1]
        eigval.shape = [order+1,1]
        '''
        
        num_val=len(eigval)
        conj_indices=deque()
        
        for i in range(num_val):
            this_val=eigval[i]
            this_conj_val = np.conj(this_val)
            if this_val == this_conj_val: #remove real eigvals
                continue
                #conj_indices.append(i)
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
        
        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        #             0         1           2           
        #self.state= [Toeplitz, State Mat., Modal Par.]
        out_dict={'self.state':self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time']=self.start_time
        #out_dict['self.prep_data']=self.prep_data
        if self.state[0]:# covariances
            out_dict['self.toeplitz_matrix'] = self.toeplitz_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[1]:# state models
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.state_matrix'] = self.state_matrix
            out_dict['self.output_matrix'] = self.output_matrix
        if self.state[2]:# modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.eigenvalues'] = self.eigenvalues
            
        np.savez_compressed(fname, **out_dict)
        
    @classmethod 
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname)    
        #             0         1           2          
        #self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state= list(in_dict['self.state'])
        else:
            return
        
        for this_state, state_string in zip(state, ['Covariance Matrices Built',
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
        if state[0]:# covariances
            ssi_object.toeplitz_matrix = in_dict['self.toeplitz_matrix']
            ssi_object.num_block_columns = int(in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
        if state[1]:# state models
            ssi_object.max_model_order = int(in_dict['self.max_model_order'])
            ssi_object.state_matrix= in_dict['self.state_matrix']
            ssi_object.output_matrix = in_dict['self.output_matrix']
        if state[2]:# modal params
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']
            ssi_object.eigenvalues = in_dict['self.eigenvalues']
        
        return ssi_object
    
    @staticmethod
    def rescale_mode_shape(modeshape):
        #scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape
def main():
    pass

if __name__ =='__main__':
    pass
    #main()
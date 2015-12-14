
from SSICovRef import SSICovRef
import numpy as np

class AutoKMeans(object):
    
    def __init__(self, modal_data,
                 stab_frequency=0.01, stab_damping=0.05, stab_MAC=0.02):
        '''
        stab_* in %
        '''
        super().__init__()
        
        assert isinstance(modal_data, SSICovRef)
        
        self.modal_data =modal_data
        
        self.max_model_order = self.modal_data.max_model_order
        
        self.modal_frequencies = np.ma.array(self.modal_data.modal_frequencies)
        self.modal_damping = self.modal_data.modal_damping
        self.mode_shapes = self.modal_data.mode_shapes
        
        self.stab_frequency = stab_frequency
        self.stab_damping = stab_damping
        self.stab_MAC = stab_MAC
    
    def calculate_stabilization_values(self):
        print('Checking stabilisation criteria...')     
          
        # Direction 1: model order, Direction 2: current pole, Direction 3: previous pole:
        self.freq_diffs = np.zeros((self.max_model_order, self.max_model_order, self.max_model_order))
        self.damp_diffs = np.zeros((self.max_model_order, self.max_model_order, self.max_model_order))
        self.MAC_diffs = np.zeros((self.max_model_order, self.max_model_order, self.max_model_order))
        
        self.MPC_matrix = np.zeros((self.max_model_order, self.max_model_order))
        self.MP_matrix = np.zeros((self.max_model_order, self.max_model_order))
        self.MPD_matrix = np.zeros((self.max_model_order, self.max_model_order))
        
        previous_freq_row = self.modal_frequencies[0,:]
        previous_damp_row = self.modal_damping[0,:]
        previous_mode_shapes_row = self.mode_shapes[:,:,0]
        
        previous_non_zero_entries  = np.nonzero(previous_freq_row) # tuple with array of indizes of non-zero frequencies
        previous_length = len(previous_non_zero_entries[0])
 
        previous_freq = previous_freq_row[previous_non_zero_entries]
        previous_damp = previous_damp_row[previous_non_zero_entries]
        previous_mode_shapes = previous_mode_shapes_row[:,previous_non_zero_entries[0]]
        
        for current_order in range(1,self.max_model_order):      
     
            current_freq_row = self.modal_frequencies[(current_order),:]
            current_damp_row = self.modal_damping[(current_order),:]
            current_mode_shapes_row = self.mode_shapes[:,:,current_order]
    
            current_non_zero_entries = np.nonzero(current_freq_row)            
            current_length = len(current_non_zero_entries[0])
            
            if not current_length: continue
            
            current_freq = current_freq_row[current_non_zero_entries]
            current_damp = current_damp_row[current_non_zero_entries]
            current_mode_shapes = current_mode_shapes_row[:,current_non_zero_entries[0]] 
                
            self.MAC_diffs[current_order, :current_length,:previous_length] = np.transpose(1 - self.calculateMAC(previous_mode_shapes[:,:previous_length], current_mode_shapes[:,:current_length]))
            
            self.MPC_matrix[current_order, :current_length] = self.calculateMPC(current_mode_shapes[:,:current_length])
            
            self.freq_diffs[current_order, :current_length, :previous_length] = np.abs((np.repeat(np.expand_dims(previous_freq, axis=1), current_freq.shape[0], axis = 1)-current_freq) / current_freq).T
            
            self.damp_diffs[current_order, :current_length, :previous_length] = np.abs((np.repeat(np.expand_dims(previous_damp, axis=1), current_damp.shape[0], axis = 1)-current_damp) / current_damp).T
            
            self.MPD_matrix[current_order, :current_length], self.MP_matrix[current_order, :current_length] =  self.calculateMPD(current_mode_shapes[:,:current_length])
                
            previous_freq=current_freq
            previous_damp=current_damp
            previous_mode_shapes=current_mode_shapes
            previous_length=current_length
   
    
    
    def automaticSelection(self, ):
        
        """The automatic modal analysis done in three stages clustering. 
        1st stage: values sorted according to their soft and hard criteria by a 2-means partitioning algorithm
        2nd stage: hierarchical clustering with automatic or user defined intercluster distance 
                   the automatic distance is based on the 'df' and 'MAC' values from the centroids obtained in the first stage
                   -----------------------------------
                   | d = weight*df + 1 - weight*MAC  |
                   -----------------------------------
        3rd stage: 2-means partitioning of the physical and spurious poles. """
        
        #t0 = time()
        #try:
            #sh = shelve.open(self.read_SSI, flag='r', writeback=False)
            #self.modal_values = sh[self.timestamp]['Modal values']
            #self.mode_shapes = sh[self.timestamp]['Mode shapes']
            #self.conj_bool = sh[self.timestamp]['Complex conjugate']                    
            #self.max_order = len(self.modal_values)  
            #dec_fact = sh[self.timestamp]['Decimate']        
            #filt_w = sh[self.timestamp]['Filter']           
            #sh.close()
        #except:
            #sys.exit('Provide the shelve containing the SSI results..')
        
#         for index in range(self.max_order):
#             current_length = len(self.modal_values['order_{0}'.format(index)]['frequencies']) 
#             
#             if index > 0:     
#                 for i in range(current_length):
#                     hv1 = 1
#                     hv2 = 1
#                     if(self.modal_values['order_{0}'.format(index)]['frequencies'][i]<30):                            
#                         f_diff = fcl.calculateDelta(self.modal_values['order_{0}'.format(index)]['frequencies'][i],self.modal_values['order_{0}'.format(index-1)]['frequencies'])                        
#                         val_f, idx_f = min((val_f, idx_f) for (idx_f, val_f) in enumerate(f_diff)) # determine the closest mode
#                         xi_diff = fcl.calculateDelta(self.modal_values['order_{0}'.format(index)]['damping'][i],self.modal_values['order_{0}'.format(index-1)]['damping'][idx_f])
#                         lambda_diff = fcl.calculateDelta(self.modal_values['order_{0}'.format(index)]['eigenvalues'][i],self.modal_values['order_{0}'.format(index-1)]['eigenvalues'][idx_f])
#                                 
#                         MAC = fcl.calculateMAC(self.mode_shapes['order_{0}'.format(index)][i],self.mode_shapes['order_{0}'.format(index-1)][idx_f])             
#                         MPC = fcl.calculateMAC(self.mode_shapes['order_{0}'.format(index)][i],np.conj(np.array(self.mode_shapes['order_{0}'.format(index)][i])))
#                         MPD, MP = fcl.calculateMPD(self.mode_shapes['order_{0}'.format(index)][i])
#                     
#                         if self.modal_values['order_{0}'.format(index)]['damping'][i] < 0:
#                             hv1 = 0 
#                         if self.modal_values['order_{0}'.format(index)]['damping'][i] > 15:
#                             hv2 = 0
#                     
#                         hv3 = self.conj_bool['order_{0}'.format(index)][0][i]
#                         self.all_modes.append([index, i, val_f, xi_diff, lambda_diff, MAC, MPC, MPD, MP, hv1, hv2, hv3])
        
        ######################################### 1st stage of clustering ############################################
        
        # represent all the vectors by their soft criteria 
        all_poles = np.vstack([self.all_modes])[:,2:8]        
        iter = self.config['System identification']['Clustering parameters']['Kmeans  iteration']        
        # the k-means algorithm is sensitive to the initial starting values in order to converge to a solution
        # therefore two starting attempts are introduced        
        self.ctr_init = np.array([[1e-9,1e-9,1e-9,1,1,1e-9],[1,1,1,1e-9,1e-9,1]])
        self.ctr, idx = vq.kmeans2(all_poles, self.ctr_init, iter)
        #if Counter(idx)[1]<10:
        factor = 0.8
        while Counter(idx)[1]<10 and factor >0:
                print('The initial centroid for the spurious data was reinitialized with a factor of {0}..'.format(factor))
                self.ctr_init = np.array([[1e-9,1e-9,1e-9,1,1,1e-9],[factor*1,factor*1,factor*1,1e-9,1e-9,factor*1]])
                self.ctr, idx = vq.kmeans2(all_poles, self.ctr_init, iter)            
                factor = factor - 0.05                
        print('Possibly physical poles 1st stage: {0}\nSpurious poles 1st stage: {1}'.format(Counter(idx)[0],Counter(idx)[1]))
        
        if not Counter(idx)[1]: 
            raise IndexError
                    
        # track the corresponding values by the code book, i.e. add back index, order values and hard validation criteria        
        cls_data = fcl.getClustersKMeans(idx, self.all_modes)
        physical_poles_soft = cls_data[0]                       # physical poles given by the soft criteria
        self.spurious_poles_1st_stage = cls_data[1]         # spurious poles given by the soft criteria  
                  
        # apply the hard validation criteria      
        self.physical_poles_1st_stage = [pole for pole in physical_poles_soft if pole[9:12] == [1,1,1]] 
        spurious_poles_hard = [pole for pole in physical_poles_soft if not pole[9:12] == [1,1,1]]        
        self.spurious_poles_1st_stage += spurious_poles_hard   # append the additional spurious poles after hard validation
        
        try:
            weight_f = self.config['System identification']['Clustering parameters']['Hierarchical']['Weight frequency']
            weight_MAC = self.config['System identification']['Clustering parameters']['Hierarchical']['Weight MAC']
        except:
            weight_f = 1
            weight_MAC = 1
                
        if self.config['System identification']['Clustering parameters']['Hierarchical']['Intercluster distance']:
            self.threshold = self.config['System identification']['Clustering parameters']['Hierarchical']['Intercluster distance']
            
        else:
            self.threshold = weight_f * self.ctr[0][0] + weight_MAC * (1 - self.ctr[0][3])
            print('The cut-off level for clustering is {0}..'.format(self.threshold))
        
        # get the modal parameters and append to the indicators
        poles_initial = []     
        for indicator in self.physical_poles_1st_stage:
            poles_initial.append([self.modal_values['order_{0}'.format(int(indicator[0]))]['frequencies'][int(indicator[1])],\
                                self.modal_values['order_{0}'.format(int(indicator[0]))]['damping'][int(indicator[1])],\
                                self.mode_shapes['order_{0}'.format(int(indicator[0]))][int(indicator[1])],\
                                self.modal_values['order_{0}'.format(int(indicator[0]))]['eigenvalues'][int(indicator[1])],\
                                indicator[2],indicator[3],indicator[4],indicator[5],indicator[6],indicator[7],indicator[8]])

        ######################################### 2nd stage of clustering ############################################
        
        # at the moment, the assembling of the distance matrix is slow, it should be optimized !
        length_mat = len(poles_initial)
        proximity_matrix = np.ascontiguousarray(np.zeros((length_mat, length_mat))) # allocate contiguous memory space
        for i in range(length_mat):    
            for j in range(i+1, length_mat):
                # alternatively, the continuous-time eigenvalues lambda can be used i.e. poles_1st_stage[i][3]
                freq_i = poles_initial[i][0] 
                freq_j = poles_initial[j][0]
                mode_shape_i = poles_initial[i][2]
                mode_shape_j = poles_initial[j][2]                        
                delta_freq = fcl.calculateDelta(freq_i,freq_j)
                MAC = 1 - fcl.calculateMAC(mode_shape_i,mode_shape_j)       
                distance = weight_f * delta_freq + weight_MAC * MAC        
                proximity_matrix.itemset((i,j), distance)
                proximity_matrix.itemset((j,i), distance)
        
        self.proximity_matrix_sq = squareform(proximity_matrix)
        linkage_matrix = hierarchy.linkage(self.proximity_matrix_sq, method='single')
        cls_assignments = hierarchy.fcluster(linkage_matrix, self.threshold, criterion='distance')          
        all_poles_2nd_stage, self.poles_2nd_stage = fcl.getClustersHierarchical(cls_assignments, poles_initial)
        all_poles_2nd_stage_sorted = sorted(all_poles_2nd_stage, key=lambda x: x[0],reverse=True)
        std_poles_2nd_stage = []
        # prepare the list to find the standard deviation
        for set in all_poles_2nd_stage_sorted:
            std_poles_2nd_stage_one_mode = [np.std(x) for x in set[1:]]
            std_poles_2nd_stage.append(std_poles_2nd_stage_one_mode) 
        
        ######################################### 3rd stage of clustering ############################################
        
        # pairwise sorted clusters by the nr. of poles                
        poles_sorted_cluster_pairwise = sorted(self.poles_2nd_stage, key=lambda x: x[1], reverse=True)
        # split into a list of two tuples for plotting 
        poles_sorted_cluster = list(zip(*poles_sorted_cluster_pairwise))
        # 1D vector with each cluster count    
        nr_poles_cluster = list(poles_sorted_cluster[1])
        # add 0 elements clusters
        new_sets = [x for x in list(poles_sorted_cluster[1]) if x > max(poles_sorted_cluster[1])/20] 
        for i in range(len(new_sets)):
            nr_poles_cluster.append(0)                      
        
        _, self.idx = vq.kmeans2(np.array(nr_poles_cluster), np.array([max(nr_poles_cluster), 1e-12]), iter)
        
        print('Number of physical modes: {0}'.format(Counter(self.idx)[0]))
        self.physical_poles_3rd_stage = poles_sorted_cluster_pairwise[0:Counter(self.idx)[0]+1] # keep clusters containing physical poles
        std_poles_2nd_stage = std_poles_2nd_stage[0:Counter(self.idx)[0]]
        self.physical_poles_3rd_stage.sort(key=lambda x: x[0])                           # pairwise sorted clusters by frequency
        self.physical_poles_3rd_stage = list(zip(*self.physical_poles_3rd_stage))        # group values into tuples         
        print('Clustering succeeded!')
        # write the analysis information to a text file

        export_modes = 'AUTOMATIC MODAL ANALYSIS\n'\
                    + '========================\n\n'\
                    + 'Time:\t\t\t'         + str(fcl.convertWinUTC_2UnixEpochDateTime(int(self.timestamp))) + '\n'\
                    + 'Frequencies [Hz]:\t' + str(list(self.physical_poles_3rd_stage[0]))                    + '\n'\
                    + 'Damping [%]:\t\t'    + str(list(self.physical_poles_3rd_stage[2]))                    + '\n'\
                    + 'Mode shapes:\n'      + str(np.array(self.physical_poles_3rd_stage[3]))                + '\n'\
                    + 'Number of poles:\t'  + str(list(self.physical_poles_3rd_stage[1]))                    + '\n'\
                    + '\u0394f  [-]:\t\t'   + str(list(self.physical_poles_3rd_stage[4]))                    + '\n'\
                    + '\u0394xi [-]:\t\t'   + str(list(self.physical_poles_3rd_stage[5]))                    + '\n'\
                    + 'MAC [-]:\t\t'        + str(list(self.physical_poles_3rd_stage[6]))                    + '\n'\
                    + 'MPC [-]:\t\t'        + str(list(self.physical_poles_3rd_stage[7]))                    + '\n'\
                    + 'MP  [\u00b0]:\t\t'   + str(list(self.physical_poles_3rd_stage[8]))                    + '\n'\
                    + 'MPD [-]:\t\t'        + str(list(self.physical_poles_3rd_stage[9]))                    + '\n\n'\
                    + 'SSI parameters\n'\
                    + '=======================\n'\
                    + 'Maximum order :\t\t'     + str(self.max_order) + '\n'\
                    + 'Block rows :\t\t'        + str(self.config['System identification']['Model parameters']['Block rows'])    + '\n'\
                    + 'Block columns :\t\t'     + str(self.config['System identification']['Model parameters']['Block columns']) + '\n'\
                    + 'Decimation :\t\t'        + str(dec_fact)             + '\n'\
                    + 'Filtering :\t\t'         + str(filt_w)               + '\n'\
                    + 'Cut-off distance :\t'    + str(self.threshold)       + '\n'\
                    + 'Weight freq dist :\t'    + str(weight_f)             + '\n'\
                    + 'Weight MAC :\t\t'        + str(weight_MAC)           + '\n\n'\
                    + 'Standard deviation\n'\
                    + '=======================\n'\
                    + 'Frequencies [-]:\t'      + str([x[0] for x in std_poles_2nd_stage]) + '\n'\
                    + 'Damping [-]:\t\t'        + str([x[1] for x in std_poles_2nd_stage]) + '\n'\
                    + 'MAC [-]:\t\t'            + str([x[2] for x in std_poles_2nd_stage]) + '\n'\
                    + 'MPC [-]:\t\t'            + str([x[3] for x in std_poles_2nd_stage]) + '\n'\
                    + 'MPD [-]:\t\t'            + str([x[4] for x in std_poles_2nd_stage]) 
                                                                                           
        f = open(self.write_a[0] + '.txt', 'w')
        f.write(export_modes)
        f.close()
           
        # write detected modes to a shelve structure for later manipulation 
        sh = shelve.open(self.write_a[1] + '.slv', flag='c', writeback=False)
        shelve_data = {'Modal_parameters':{     'Frequencies':        list(self.physical_poles_3rd_stage[0]),\
                                                'Damping':            list(self.physical_poles_3rd_stage[2]),\
                                                'Mode_shapes':        list(self.physical_poles_3rd_stage[3])},\
                             'Indicators':      {'Number of poles':   list(self.physical_poles_3rd_stage[1]),\
                                                 'df':                list(self.physical_poles_3rd_stage[4]),\
                                                 'dxi':               list(self.physical_poles_3rd_stage[5]),\
                                                 'MAC':               list(self.physical_poles_3rd_stage[6]),\
                                                 'MPC':               list(self.physical_poles_3rd_stage[7]),\
                                                 'MP':                list(self.physical_poles_3rd_stage[8]),\
                                                 'MPD':               list(self.physical_poles_3rd_stage[9])},\
                            'Clustering':      {'Physical poles 1st stage': self.physical_poles_1st_stage,\
                                                'Spurious poles 1st stage': self.spurious_poles_1st_stage,\
                                                'Poles 2nd stage':          self.poles_2nd_stage,\
                                                'Physical poles 3rd stage': self.physical_poles_3rd_stage,\
                                                'Initial centroids'       : self.ctr_init,\
                                                'Final centroids':          self.ctr,\
                                                'Indices':                  self.idx,\
                                                'Threshold':                self.threshold,\
                                                'Proximity matrix':         self.proximity_matrix_sq}}
        
        sh[self.timestamp] = shelve_data
        sh.close()        
        print('Automatic system identification completed in {0} sec'.format(time()-t0))
        
    def plotAndsaveClusteringData(self):    
        """ Plot relevant results of the clustering."""
        
        fig_initial_size = [8.125, 6.125]  # display and plot everything maximized (2 * default fig size)        
        plt.rcParams.update({'font.size': 18})
        font_size = 18
        try:
            sh = shelve.open(self.read_shelve_a, flag='r', writeback=False)
            self.proximity_matrix_sq = sh[self.timestamp]['Clustering']['Proximity matrix']
            self.ctr_init = sh[self.timestamp]['Clustering']['Initial centroids']
            self.ctr = sh[self.timestamp]['Clustering']['Final centroids']
            self.idx = sh[self.timestamp]['Clustering']['Indices']
            self.threshold = sh[self.timestamp]['Clustering']['Threshold']
            self.physical_poles_1st_stage = sh[self.timestamp]['Clustering']['Physical poles 1st stage']
            self.spurious_poles_1st_stage = sh[self.timestamp]['Clustering']['Spurious poles 1st stage']
            self.poles_2nd_stage = sh[self.timestamp]['Clustering']['Poles 2nd stage']
            self.physical_poles_3rd_stage = sh[self.timestamp]['Clustering']['Physical poles 3rd stage']
            sh.close()
        except:
            print('Provide the shelve containing the clustering information..')
            raise
                
        try:
            sh = shelve.open(self.read_SSI, flag='r', writeback=False)
            self.modal_values = sh[self.timestamp]['Modal values']
            sh.close()
        except:
            print('Provide also the shelve containing the modal information..')
            raise
                            
        # arrange the data for the stabilization plot in the form [frequency, model_order]    
        ph_poles_1st_stage_plot = []
        for indicator in self.physical_poles_1st_stage:
            ph_poles_1st_stage_plot.append([self.modal_values['order_{0}'.format(int(indicator[0]))]['frequencies'][int(indicator[1])],\
                                indicator[0]])
        sp_poles_1st_stage_plot = []                        
        for indicator in self.spurious_poles_1st_stage:
            sp_poles_1st_stage_plot.append([self.modal_values['order_{0}'.format(int(indicator[0]))]['frequencies'][int(indicator[1])],\
                                indicator[0]])
        
        #=========================== the 1st plot -2 ways to show: automatized stabilization plot ==================     
        #fig, ax = plt.subplots()                       
        fig = plt.figure(figsize = [i*2 for i in fig_initial_size])
        ax = fig.add_subplot(111)
        ax.scatter(np.array(ph_poles_1st_stage_plot)[:, 0], np.array(ph_poles_1st_stage_plot)[:, 1], marker='x', c='k', s=10, label='stable pole') # facecolors='none', edgecolors='r', s=20, label='stable pole')   
        ax.scatter(np.array(sp_poles_1st_stage_plot)[:, 0], np.array(sp_poles_1st_stage_plot)[:, 1], marker='x', c='k', s=10, alpha=0.2, label='pole') # facecolors='none', edgecolors='k', s=20, alpha=0.2, label='pole')
        ax.autoscale_view(tight=True)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Model Order [-]')
        ax.set_title('Stabilization Diagram')
        plt.tight_layout()
        plt.savefig('Stabilization_plot_imposed_' + self.timestamp + '.' + self.format_plot, format = self.format_plot)
        fig.canvas.draw()        
        #plt.show(block=False)        
        
        fig = plt.figure(figsize = [16.25, 7.5])
        ax1 = fig.add_subplot(122)  
        ax1.scatter(np.array(ph_poles_1st_stage_plot)[:, 0], np.array(ph_poles_1st_stage_plot)[:, 1], marker='.', c='k', s=12, label='stable pole') # facecolors='none', edgecolors='r', s=20, label='stable pole') 
        ax1.autoscale_view(tight=True)                
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Model Order [-]')
        ax1.set_title('Stabilization Diagram - cleared')
        ax2 = fig.add_subplot(121,sharex=ax1,sharey=ax1)
        ax2.scatter(np.array(ph_poles_1st_stage_plot)[:, 0], np.array(ph_poles_1st_stage_plot)[:, 1], marker='.', c='k', s=12) # facecolors='none', edgecolors='r', s=20, label='stable pole')
        ax2.scatter(np.array(sp_poles_1st_stage_plot)[:, 0], np.array(sp_poles_1st_stage_plot)[:, 1], marker='.', c='k', s=12) # facecolors='none', edgecolors='k', s=20, alpha=0.2, label='pole')
        ax2.autoscale_view(tight=True)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Model Order [-]')
        ax2.set_title('Stabilization Diagram - full')
        plt.tight_layout()
        plt.savefig('Stabilization_plot_' + self.timestamp + '.' + self.format_plot, format = self.format_plot)
        #plt.show()        
                 
        self.physical_poles_1st_stage = [poles[2:8] for poles in self.physical_poles_1st_stage]
        self.spurious_poles_1st_stage = [poles[2:8] for poles in self.spurious_poles_1st_stage]
                
        #================================ the 2nd plot: 1st clustering results ==========================
        labels = (r'd($f_i,f_j$) [-]', r'd($\xi_i,\xi_j$) [-]', r'd($\lambda_i,\lambda_j$) [-]', 'MAC [-]', 'MPC [-]','MPD [-]')        
        # for different 2D representations enter the pair of coordinates pointing to the labels above
        index_label = [[5,0,0,3,3,4],[1,1,2,1,0,1]] 
        for label in range(len(index_label[0])):
            #fig, ax = plt.subplots()
            fig = plt.figure(figsize = [i*2 for i in fig_initial_size])
            ax = fig.add_subplot(111)
            ax.scatter(np.array(self.physical_poles_1st_stage)[:, index_label[0][label]], np.array(self.physical_poles_1st_stage)[:, index_label[1][label]], facecolors='none', marker='x', edgecolors='k', s=25, label='physical pole')
            ax.scatter(np.array(self.spurious_poles_1st_stage)[:, index_label[0][label]], np.array(self.spurious_poles_1st_stage)[:, index_label[1][label]], facecolors='none', alpha=0.2, edgecolors='k', s=25, label='spurious pole')
            ax.autoscale_view(tight=True)
            ax.scatter(self.ctr_init[0][index_label[0][label]], self.ctr_init[0][index_label[1][label]], facecolors='none', marker='s', edgecolors='r', linewidths=3, s=250, label = 'ideal physical centroid')
            ax.scatter(self.ctr_init[1][index_label[0][label]], self.ctr_init[1][index_label[1][label]], facecolors='none', marker='s', edgecolors='b', linewidths=3, s=250, label = 'ideal spurious centroid')
            ax.scatter(self.ctr[0,index_label[0][label]], self.ctr[0,index_label[1][label]], facecolors='none', marker='D', edgecolors='r', linewidths=3, s=250, label = 'physical centroid')
            ax.scatter(self.ctr[1,index_label[0][label]], self.ctr[1,index_label[1][label]], facecolors='none', marker='D', edgecolors='b', linewidths=3, s=250, label = 'spurious centroid')
            ax.set_xlabel(labels[index_label[0][label]], size = font_size)
            ax.set_ylabel(labels[index_label[1][label]], size = font_size)            
            # when saving the figure with default dimensions the legend needs more adjustments
            # plotted inside the figure, it may overwrite the data points 
            box = ax.get_position()
            ax.set_position([box.x0 + box.width*0.05, box.y0 + box.height*0.05, box.width, box.height])
            label_objs  = []
            label_texts = []    
            for collection in ax.collections:
                collection_label = collection.get_label()                            
                label_objs.append(collection)
                label_texts.append(collection_label)
            ax.legend(label_objs, label_texts,  loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=3, fontsize = 16, markerscale = 0.8)
            #plt.tight_layout()
            #plt.savefig('Clustering_quality_' + str(label) + '_' + self.timestamp + '.' + self.format_plot, format = self.format_plot, bbox_inches='tight')
            #plt.show()  
            #fig.get_size_inches()     
                        
        poles_sorted_cluster_pairwise = sorted(self.poles_2nd_stage, key=lambda x: x[1],reverse=True)        
        poles_sorted_cluster = list(zip(*poles_sorted_cluster_pairwise)) 
        nr_poles_cluster = list(poles_sorted_cluster[1])
                 
        freq_sorted_cluster_pairwise = sorted(self.poles_2nd_stage, key=lambda x: x[0]) # pairwise sorted clusters by frequency
        freq_sorted_cluster = list(zip(*freq_sorted_cluster_pairwise))                  # split into a list of two tuples for plotting
        
        #================================ the 3rd plot: 2nd clustering results ================================
        #fig, ax = plt.subplots()
        #fig = plt.figure(figsize = [i*2 for i in fig_initial_size])
        fig = plt.figure(figsize = [16.25, 8])
        ax = fig.add_subplot(111)
        ax.autoscale_view(tight=True) 
        # by uncommenting this part, the x axis can display all the labels of the clusters
        fig.canvas.draw()        
        lbl = [item.get_text() for item in ax.get_xticklabels()]
        lbl = np.ndarray.round(np.array(freq_sorted_cluster[0]),decimals=3)
        labels_loc = np.arange(len(freq_sorted_cluster[0]))
        ax.yaxis.grid()
        ax.set_xticks(labels_loc)
        ax.set_xticklabels(lbl, rotation=40, ha='right')        
        plt.bar(range(len(freq_sorted_cluster[0])), freq_sorted_cluster[1], width=0.5, align='center', color='r', linewidth=0.2)
        plt.xlim(xmax=35)
        plt.ylim(ymax=70)
        plt.ylabel('Nr. of elements')
        plt.xlabel('Frequency [Hz]')
        #plt.title('Clusters')
        plt.tight_layout()
        plt.savefig('Cluster_elements_2_' + self.timestamp + '.' + self.format_plot, format = self.format_plot)
        #plt.show()
                
        #================================ the 4th plot: 3rd clustering results ================================
        #fig, ax = plt.subplots()
        #fig = plt.figure(figsize = [i*2 for i in fig_initial_size])
        fig = plt.figure(figsize = [16.25, 8])
        ax = fig.add_subplot(111)
        #colors = ([([0,0,0],[1,0,0])[i] for i in self.idx])
        colors = ([('r','b')[i] for i in self.idx])
        plt.bar(np.arange(len(nr_poles_cluster)), nr_poles_cluster, align='center', color=colors, linewidth=0.1, edgecolor='k')
        ax.autoscale_view(tight=True)
        ax.set_xlabel('Mode set')
        ax.set_ylabel('Nr. of elements')
        plt.tight_layout()
        plt.savefig('Physical_set_' + self.timestamp + '.' + self.format_plot, format = self.format_plot)
        #plt.show()
                
        #================================ the 5th plot: 2nd and 3rd clustering results ==========================
        #fig = plt.figure()
        fig = plt.figure(figsize = [i*2 for i in fig_initial_size])        
        ax1 = fig.add_subplot(211)        
        ax1.scatter(np.array(ph_poles_1st_stage_plot)[:, 0], np.array(ph_poles_1st_stage_plot)[:, 1], marker='.', c='k', s=10, label='stable pole')
        ax1.autoscale_view(tight=True)
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Model order [-]')
        ax1.set_title('Stabilization Diagram')
        for line in self.physical_poles_3rd_stage[0]:
            ax1.axvline(line, c='b')

        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.bar(poles_sorted_cluster[0], poles_sorted_cluster[1], width=0.01, align='center')
        ax2.axhline(poles_sorted_cluster_pairwise[Counter(self.idx)[0]][1], c='r', ls='--', linewidth=2)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Nr. of elements')
        ax2.set_title('Clusters')
        plt.tight_layout()
        plt.xlim(xmin=0)
        plt.savefig('Main_plot_clusters_' + self.timestamp + '.' + self.format_plot, format=self.format_plot)
        #plt.show()
                
        #================================ the 6th plot: visualize linkages =============================
        rel_matrix = hierarchy.linkage(self.proximity_matrix_sq, method='single')
        lvs = hierarchy.leaves_list(rel_matrix)
           
        def _llf(id):
            if len(lvs)>500:
                if (np.where(id==lvs)[0][0]%100==0):
                    return str(np.where(id==lvs)[0][0])
                else:
                    return str('')
            else:
                if (np.where(id==lvs)[0][0]%10==0):
                    return str(np.where(id==lvs)[0][0])
                else:
                    return str('')  
        #fig, ax = plt.subplots()
        fig = plt.figure(figsize = [i*2 for i in fig_initial_size])
        ax = fig.add_subplot(111)
        hierarchy.dendrogram(rel_matrix, leaf_label_func=_llf, color_threshold=self.threshold, leaf_font_size=16, leaf_rotation=40)
        ax=plt.gca()
        ax.set_xlabel('Mode number [-]')
        ax.set_ylabel('Distance [-]')
        #ax.set_title('Dendrogram', fontsize = font_size, fontweight='bold')
        ax.axhline(self.threshold, c='r', ls='--', linewidth=3)
        plt.tight_layout()
        plt.savefig('Dendrogram_' + self.timestamp + '.' + self.format_plot, format=self.format_plot)
        #plt.show()
        plt.close('all') 
        '''A Warning message in the form "can't invoke "event" command:  application has been destroyed"
           will be shown when running multiple files. It seems to be a harmless glitch in Tk. 
           It is triggered by making a figure and then closing it.'''
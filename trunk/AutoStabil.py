import numpy as np
from StabilDiagram import StabilPlot

class AutoStabil(StabilPlot):
    
    def __init__(self,modal_data, prep_data=None,
                 stab_frequency=0.01, stab_damping=0.05, stab_MAC=0.02,
                 d_range=None, mpc_min=None, mp_max=None, mpd_max=None):
        
        super().__init__(modal_data, prep_data,
                 stab_frequency, stab_damping, stab_MAC,
                 d_range, mpc_min, mp_max, mpd_max)
        
    def auto_select(self, frange, drange, num_modes=2):
        '''
        Select Modes by their hard criteria
        Soft criteria are taken from the stabilization diagram
        select num_modes per model order (selection of closely spaced modes)
        num modes must be multiple of 2
        '''
        
        assert isinstance(frange, (tuple, list))
        assert len(frange)==2
        
        assert isinstance(drange,(tuple,list))
        assert len(drange)==2
        
        assert isinstance(num_modes, int)
        assert num_modes >= 1
        assert num_modes % 2 == 0
        #self.update_stabilization(df_max, dd_max, d_mac, d_range, mpc_min, mp_max, mpd_max, n_range)
        self.frange = frange
        self.drange=drange
        self.num_modes = num_modes
        
        self.modal_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask
        
        # select by hard criteria
        mask_sf_min = self.modal_frequencies >= frange[0]         # stable in frequency
        mask_sf_max = self.modal_frequencies <= frange[1]
        mask_sf = np.logical_and(mask_sf_max, mask_sf_min)
        #print(frange)
        
        mask_sd_min = self.modal_frequencies >= drange[0]        # stable in damping
        mask_sd_max = self.modal_frequencies <= drange[1]   
        mask_sd = np.logical_and(mask_sd_max, mask_sd_min)
        #print(drange)
        
        #select by soft criteria
        mask_sc = self.masks[3]   #stable in all criteria 0: only f, 1: only d, 2: only_v, 3: in all, 4:all poles
        
        mask_all = np.logical_and(mask_sf, mask_sd)
        mask_all = np.logical_not(mask_all)
        mask_all = np.logical_or(mask_all, mask_sc)
        self.modal_frequencies.mask = np.ma.nomask
        clusters = [self.modal_frequencies>-1 for i in range(num_modes)]
        
        self.modal_frequencies.mask = mask_all
        
        #modal_frequencies = self.modal_frequencies[~self.modal_frequencies.mask]
        #print(self.modal_frequencies)
        mf = self.modal_frequencies.compressed()
        mean = np.median(mf)
        #print(self.stab_frequency, self.stab_damping, self.stab_MAC, self.d_range, self.mpc_min, self.mp_max, self.mpd_max, self.frange, self.drange)
        #print(mean)
        #clusters = [[] for i in range(num_modes)]
        
        
        
        for order in range(self.max_model_order):
            for mode in range(self.max_model_order):
                freq = self.modal_frequencies[order, mode]
                if freq: 
                    #print(order,mode, freq)
                    if freq <= mean:
                        clusters[0][order,mode]=False
                    else:
                        clusters[1][order,mode]=False
                        
        self.modal_frequencies.mask = mask_all
        self.modal_damping = np.ma.array(self.modal_damping)
        self.modal_damping.mask = mask_all
        #self.order_dummy.mask = np.ma.nomask
        indices = []
        for cluster_ind in range(num_modes):
            factor = 2
            while True:
                for array in [self.modal_frequencies, self.modal_damping]:
                    cluster = clusters[cluster_ind]
                    array.mask = cluster
                    this_mean = array.mean()
                    this_std = array.std()
                    #print(this_mean, this_std)
                    a_1 = array < this_mean - factor*this_std
                    a_2 = array > this_mean + factor*this_std
                    a_1 = np.logical_or(a_1,a_2)
                    a_2 = np.logical_or(a_1,cluster)
                    if a_2.all(): 
                        #print('no more values left', cluster_ind)
                        factor*=1.1
                    if (a_2 == cluster).all(): 
                        #print('cluster is not changing', cluster_ind)
                        factor*=0.9
                    if a_2.count() <=1:
                        break
                    clusters[cluster_ind] = a_2
                else:
                    #print('continuing', cluster_ind, clusters[cluster_ind].count())
                    continue
                break
            #print('cluster_length', np.ma.count(clusters[cluster_ind]), cluster_ind)
            cluster = clusters[cluster_ind]
            #self.modal_frequencies.mask = cluster
            #clusters[cluster_ind] = ~(array == np.ma.median(array))
            #median = np.ma.median(self.modal_frequencies)
            #print(array != np.ma.median(array))
            
            #self.modal_frequencies.mask = np.ma.nomask
            #clusters[cluster_ind] = self.modal_frequencies != median
            #self.modal_frequencies.mask = clusters[cluster_ind]
            #print(median, self.modal_frequencies.compressed())
            #self.modal_frequencies.mask = np.ma.nomask
            
            #print(np.argwhere(array == np.ma.median(array)),np.ma.median(array))
            self.modal_frequencies.mask =  mask_all
            #self.modal_damping = np.ma.array(self.modal_damping)
            self.modal_damping.mask = mask_all
            self.order_dummy.mask =  mask_all
            
            index = np.unravel_index(np.argmax(cluster),cluster.shape)
            indices.append(index)
            x = self.modal_frequencies[index]
            y = self.order_dummy[index]
            if not x: return
            if not y: return
            
            #print(self.modal_frequencies[index], self.modal_damping[index], self.order_dummy[index], self.modal_data.mode_shapes[:,index[1],index[0]])
            #self.cursor.add_datapoints([index])
            self.select_modes.append(index)
            #self.ax.scatter(x,y,facecolors='none',edgecolors='red',s=200, visible=True)
        
        self.modal_damping.mask = np.ma.nomask 
        self.modal_frequencies.mask = np.ma.nomask     
        self.order_dummy.mask= np.ma.nomask 
            #print(order, this_freqs)
            #break
        #self.masks[3] = clusters[1]
        
        #self.modal_frequencies.mask = np.ma.nomask
        
        #mask_sv = self.modal_frequencies != 0         # stable in mode shape (MAC)
        #mask_sa = self.modal_frequencies != 0         # stable in all criteria
        #mask_other = self.modal_frequencies != 0
        #for ii in range(self.modal_frequencies.shape[0]):
        #    print(self.modal_frequencies[ii,:])
        #    break
        
        assert len(indices)==num_modes
        self.scatter_this(3)
         
    #def update_stabilization(self, df_max=None, dd_max=None, d_mac=None, d_range=None, mpc_min=None, mp_max=None, mpd_max=None, n_range = None):
    def update_stabilization(self, df_max, dd_max, d_mac, d_range, mpc_min, mp_max, mpd_max, n_range):
        #print('update_stabil')
        super().update_stabilization(df_max, dd_max, d_mac, d_range, mpc_min, mp_max, mpd_max, n_range)
        #self.auto_select(self.frange, self.drange, self.num_modes)
        
        
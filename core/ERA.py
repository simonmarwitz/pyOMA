# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg
from collections import deque
import os
from core.PreprocessingTools import PreprocessData


class ERA(object):

    def __init__(self, prep_data):
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_data, PreprocessData)
        self.prep_data = prep_data
        self.setup_name = prep_data.setup_name
        self.start_time = prep_data.start_time
        #             0         1           2
        # self.state= [SHankelMatrix, State Mat., Modal Par.
        self.state = [False, False, False]

        self.num_block_columns = None
        self.num_block_rows = None
        self.toeplitz_matrix = None
        self.hankel_matrix = None  # anil

        self.max_model_order = None
        self.state_matrix = None
        self.output_matrix = None

        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None

    def CalculateFRF(self):
        '''
        function by anil
        FRF(Frequency response function) is convertion of signal from time to frequency domain.
        The following function performs this conversion.
        '''
        measurement = self.prep_data.measurement
        num_channels = measurement.shape[1]
        num_time_steps = self.prep_data.F.shape[0]
        acceleration_fft = np.zeros(
            (num_time_steps // 2 + 1, num_channels), dtype=complex)

        F_fft = np.fft.rfft(np.hamming(num_time_steps) * self.prep_data.F)

        for channel in range(num_channels):  # loop over channels
            fft_this_channel = np.fft.rfft(np.hamming(
                num_time_steps) * measurement[:, channel])
            acceleration_fft[:, channel] = fft_this_channel

        FRF = np.zeros_like(acceleration_fft)

        for channel in range(num_channels):
            FRF[:, channel] = acceleration_fft[:, channel] / F_fft

        IRF = np.zeros((num_time_steps, num_channels))

        for channel in range(num_channels):  # loop over channels
            ifft_this_channel = np.fft.irfft(FRF[:, channel])

            IRF[:, channel] = ifft_this_channel

        self.IFRF = IRF.T

    def build_hankel_matrix(self, num_block_columns):
        '''
        author: Anil
        Constructs a shifted hankel matrix.
        '''

        IRFT = self.IFRF
        num_channels = self.prep_data.num_analised_channels
        num_block_rows = num_block_columns + 1

        self.num_block_columns = num_block_columns
        self.num_block_rows = num_block_rows

        Hankel_matrix = np.zeros(
            (num_channels *
             num_block_rows,
             num_block_columns),
            dtype=complex)
        for i in range(0, num_block_rows):
            j = i + 1
            this_block = IRFT[0:num_channels, j:(num_block_columns + j)]
            begin_row = i * num_channels
            Hankel_matrix[begin_row:(
                begin_row + num_channels), 0:num_block_columns] = this_block

        self.hankel_matrix = Hankel_matrix
        self.state[0] = True

    def compute_state_matrices(self, max_model_order=None):
        '''

        '''
        if max_model_order is not None:
            assert isinstance(max_model_order, int)

        assert self.state[0]

        hankel_matrix = self.hankel_matrix  # anil
        num_channels = self.prep_data.num_analised_channels
        num_block_columns = self.num_block_columns
        num_block_rows = self.num_block_rows
        print('Computing state matrices...')

        [U, S, V_T] = np.linalg.svd(hankel_matrix, 0)  # anil

        # anil
        S1 = np.diag(S)
        S_sqrt = np.sqrt(S1)
        p1 = np.dot(U, S_sqrt)
        # p2=np.dot(S_sqrt,V_T)

        #A=np.dot(np.linalg.pinv(p1), hankel_matrix, np.linalg.pinv(p2))
        # A=A.real
        C = p1[:num_channels, :]
        # C=C.real
        # p1=p1.real

        self.Oi = p1
        #self.state_matrix = A
        self.output_matrix = C
        self.max_model_order = max_model_order

        self.state[1] = True
        self.state[2] = False  # previous modal params are invalid now

    def compute_modal_params(self, max_model_order=None):

        if max_model_order is not None:
            assert max_model_order <= self.max_model_order
            self.max_model_order = max_model_order

        assert self.state[1]
        print('Computing modal parameters...')
        max_model_order = self.max_model_order
        num_analised_channels = self.prep_data.num_analised_channels
        num_block_rows = self.num_block_rows
        #state_matrix = self.state_matrix
        Oi = self.Oi
        output_matrix = self.output_matrix
        sampling_rate = self.prep_data.sampling_rate

        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        eigenvalues = np.zeros(
            (max_model_order, max_model_order), dtype=complex)
        mode_shapes = np.zeros(
            (num_analised_channels,
             max_model_order,
             max_model_order),
            dtype=complex)

        for order in range(1, max_model_order, 1):

            Oi0 = Oi[:(num_analised_channels * (num_block_rows - 1)), :order]
            Oi1 = Oi[num_analised_channels:(
                num_analised_channels * num_block_rows), :order]

            a = np.dot(np.linalg.pinv(Oi0), Oi1)
            eigenvalues_paired, eigvec_l, eigenvectors_paired = scipy.linalg.eig(
                a=a[0:order, 0:order], b=None, left=True, right=True)

            eigenvalues_single, eigenvectors_single = self.remove_conjugates_new(
                eigenvalues_paired, eigenvectors_paired)

            for index, k in enumerate(eigenvalues_single):

                lambda_k = np.log(complex(k)) * sampling_rate
                freq_j = np.abs(lambda_k) / (2 * np.pi)
                damping_j = np.real(lambda_k) / np.abs(lambda_k) * (-100)
                mode_shapes_j = np.dot(
                    output_matrix[:, 0:order], eigenvectors_single[:, index])

                modal_frequencies[order, index] = freq_j
                modal_damping[order, index] = damping_j
                eigenvalues[order, index] = k
                mode_shapes[:, index, order] = mode_shapes_j

        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues

        self.state[2] = True

    @staticmethod
    def remove_conjugates_new(eigval, eigvec_r, eigvec_l=None):
        '''
        removes conjugates

        eigvec_l.shape = [order+1, order+1]
        eigval.shape = [order+1,1]
        '''
        # return vectors, eigval
        num_val = len(eigval)
        conj_indices = deque()

        for i in range(num_val):
            this_val = eigval[i]
            this_conj_val = np.conj(this_val)
            if this_val == this_conj_val:  # remove real eigvals
                conj_indices.append(i)
            for j in range(
                    i + 1, num_val):  # catches unordered conjugates but takes slightly longer
                if eigval[j] == this_conj_val:

                    # if not np.allclose(eigvec_l[j],eigvec_l[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_l is not')
                    #    continue

                    # if not np.allclose(eigvec_r[j],eigvec_r[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_r is not')
                    #    continue

                    conj_indices.append(j)
                    break

        #print('indices of complex conjugate: {}'.format(conj_indices))
        conj_indices = list(set(range(num_val)).difference(conj_indices))
        #print('indices to keep and return: {}'.format(conj_indices))

        if eigvec_l is None:

            eigvec_r = eigvec_r[:, conj_indices]
            eigval = eigval[conj_indices]

            return eigval, eigvec_r

        else:
            eigvec_l = eigvec_l[:, conj_indices]
            eigvec_r = eigvec_r[:, conj_indices]
            eigval = eigval[conj_indices]

            return eigval, eigvec_l, eigvec_r

    def save_state(self, fname):

        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        #             0         1           2
        # self.state= [SHankelMatrix, State Mat., Modal Par.]
        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time
        # out_dict['self.prep_data']=self.prep_data
        if self.state[0]:  # SHankelMatrix
            #out_dict['self.toeplitz_matrix'] = self.toeplitz_matrix
            out_dict['self.hankel_matrix'] = self.hankel_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[1]:  # state models
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.state_matrix'] = self.state_matrix
            out_dict['self.output_matrix'] = self.output_matrix
        if self.state[2]:  # modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.eigenvalues'] = self.eigenvalues

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname)
        #             0         1           2
        # self.state= [SHankelMatrix, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['Shifted Hankel Matrices Built',
                                                    'State Matrices Computed',
                                                    'Modal Parameters Computed',
                                                    ]):
            if this_state:
                print(state_string)

        assert isinstance(prep_data, PreprocessData)
        setup_name = str(in_dict['self.setup_name'].item())
        start_time = in_dict['self.start_time'].item()
        assert setup_name == prep_data.setup_name
        start_time = prep_data.start_time

        assert start_time == prep_data.start_time
        #prep_data = in_dict['self.prep_data'].item()
        ssi_object = cls(prep_data)
        ssi_object.state = state
        if state[0]:  # SHankelMatrix
            ssi_object.hankel_matrix = in_dict['self.hankel_matrix']
            ssi_object.num_block_columns = int(
                in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
        if state[1]:  # state models
            ssi_object.max_model_order = int(in_dict['self.max_model_order'])
            ssi_object.state_matrix = in_dict['self.state_matrix']
            ssi_object.output_matrix = in_dict['self.output_matrix']
        if state[2]:  # modal params
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']
            ssi_object.eigenvalues = in_dict['self.eigenvalues']

        return ssi_object

    @staticmethod
    def rescale_mode_shape(modeshape):
        # scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape


def main():

    os.chdir(
        '/ismhome/staff/womo1998/Projects/2018_eigensystem_realization_algorithm/code/')
    # Data Preprocessing Class
    from core.PreprocessingTools import PreprocessData, GeometryProcessor

    # Modal Analysis PostProcessing Class e.g. Stabilization Diagram
    from core.StabilDiagram import StabilCalc, StabilPlot, StabilGUI, start_stabil_gui

    # Modeshape Plot
    from core.PlotMSH import ModeShapePlot, start_msh_gui

    '''
    - test files were generated from a simple 20 DOF spring-mass system (e.g. a uniform cantilever)
    - damping was applied using Rayleigh damping (1 % and 3 % for lowest and highest mode respectively)
    - inputs are noise/impulse forces at the respective node(s)
    - no measurement noise was added
    - time histories were generated by Newmark integration of the equation of motion of the system
    - sampling rate was 1536 Hz (time delta = 0.000651042 s)

    '''
    test_files = ['../data/impuls_node_4.npz',
                  '../data/impuls_node_8.npz',
                  '../data/noise_node_4.npz',
                  '../data/noise_node_8.npz']

    # set the test file to be read i.e from 0th to 3rd file
    test_file = test_files[0]

    data = np.load(test_file)
    # print(data['f_nat_d'])

    # numerical results
    f_nat_d = data['f_nat_d']
    damping = data['damping']
    modeshapes = data['modeshapes']
    print(f_nat_d)
    # sampling rate
    Fs = 1 / data['tDelta']

    # Force
    if '4' in test_file:
        F = data['F'][4, :]
    else:
        F = data['F'][8, :]

    # acceleration
    measurement = data['X_dotdot'].T

    # creating the geometry for plotting the identified modeshapes
    geometry_data = GeometryProcessor()

    # define nodes of the cantilever beam: {nodename: (x,y,z),...}
    geometry_data.add_nodes({str(i): (0, 0, i) for i in range(20)})

    # define lines connecting the nodes: [(start_nodename,end_nodename),...]
    geometry_data.add_lines([(i, i + 1) for i in range(19)])

    # initialize the Preprocessor Class
    prep_data = PreprocessData(measurement=measurement,
                               sampling_rate=Fs, F=F)  # anil

    # add Channel-DOF assignments: [ (chan_num, node_name, az, elev,
    # chan_name), ...]
    prep_data.add_chan_dofs([[i, str(i), 0, 0] for i in range(20)])

    # prep_data.decimate_data(decimate_factor=4)

    from ERA import ERA

    modal_data = ERA(prep_data)
    modal_data.CalculateFRF()
    modal_data.build_hankel_matrix(num_block_columns=250)
    modal_data.compute_state_matrices(max_model_order=100)
    modal_data.compute_modal_params()

    stabil_data = StabilCalc(modal_data)
    stabil_plot = StabilPlot(stabil_data)
    start_stabil_gui(stabil_plot, modal_data, geometry_data, prep_data)

    mode_shape_plot = ModeShapePlot(
        geometry_data, stabil_data, modal_data, prep_data)
    start_msh_gui(mode_shape_plot)

    '''
    Your Task:
    - Clean up your code
    - extend/change the save_state and load_state functions to enable saving results and resuming work from saved results
    - add proper descriptions for each function and comments, where needed
    - also add the literature that was used in the respective description of the functions

    - analyze all four sets of input data i.e. apply the ERA and select modes at a proper model order from the stabilization diagram
    - compare identified modal parameters with numerical results (natural frequencies, damping ratios, mode shapes)
    - for comparison of the mode shapes use the modal assurance criterion (MAC, see lecture notes)
    - get real measurement data from the pole along with the output-only modal parameters from your colleagues
    - analyze the experimental data i.e. apply the ERA and compare the results

    - write a report containing
        introduction
        theory of the ERA
        description of the code/flowcharts
        numerical example and results
        experimental example and results
        conclusions and ideas for future improvements/work
        references (add them numbered in the order of appearance in the text)
    '''


def main_oma_uq():

    from PreprocessingTools import PreprocessData, GeometryProcessor

    # Modal Analysis PostProcessing Class e.g. Stabilization Diagram
    from StabilDiagram import StabilCalc, StabilPlot, StabilGUI, start_stabil_gui

    # Modeshape Plot
    from PlotMSH import ModeShapePlot, start_msh_gui

    jid = '184d16a5f0b5'

    # creating the geometry for plotting the identified modeshapes
    geometry_data = GeometryProcessor.load_geometry(
        f'/dev/shm/womo1998/{jid}/grid.txt',
        f'/dev/shm/womo1998/{jid}/lines.txt')

    prep_data = PreprocessData.load_state(
        f'/dev/shm/womo1998/{jid}/prep_data.npz')

    arrs = np.load(f'/dev/shm/womo1998/{jid}/IRF_data.npz')
    t_vals = arrs['t_vals']
    IRF_matrix = arrs['IRF_matrix']
    F_matrix = arrs['F_matrix']
    ener_mat = arrs['ener_mat']
    amp_mat = arrs['amp_mat']

    from ERA import ERA

    modal_data = ERA(prep_data)
    modal_data.IFRF = IRF_matrix
    modal_data.build_hankel_matrix(num_block_columns=250)
    modal_data.compute_state_matrices(max_model_order=100)
    modal_data.compute_modal_params()

    stabil_data = StabilCalc(modal_data)
    stabil_plot = StabilPlot(stabil_data)
    start_stabil_gui(stabil_plot, modal_data, geometry_data, prep_data)

    mode_shape_plot = ModeShapePlot(
        geometry_data, stabil_data, modal_data, prep_data)
    start_msh_gui(mode_shape_plot)


if __name__ == '__main__':
    main_oma_uq()

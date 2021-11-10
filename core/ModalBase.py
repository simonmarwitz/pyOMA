'''
Module that contains the basic class, of which all other OMA classes
should be inherited.

@author: womo1998
'''

from core.PreProcessingTools import PreProcessSignals
import numpy as np
from collections import deque
import os


class ModalBase(object):
    '''
    Base Class from which all other modal analysis classes should be inherited
        * provides commonly used functions s.t. these don't have to be copied to
          each class
        * object type checks in post-processing functions can check for
          modal base instead of each possible modal analysis class
    '''

    def __init__(self, prep_data=None):
        super().__init__()
        if prep_data is not None:
            assert isinstance(prep_data, PreProcessSignals)
            self.setup_name = prep_data.setup_name
            self.start_time = prep_data.start_time
        else:
            self.setup_name = ''
            self.start_time = None
        self.prep_data = prep_data

        self.max_model_order = None

        self.eigenvalues = None
        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None

    @staticmethod
    def remove_conjugates(eigval, eigvec_r, eigvec_l=None, inds_only=False):
        '''
        finds conjugates:
        :math:`\\lambda_i = \\overline{\\lambda_j} \\text{for} i \\neq j`

        unstable poles i.e. negatively damped poles
        :math:`[\\ln(|\\lambda|)<0]: |\\lambda_i|> 1`

        overdamped poles
        :math:`[\\operatorname{atan}(\\Im/\\Re)=0]`
        i.e. real poles: :math:`\\Im(\\lambda_i)==0`

        imaginary poles i.e. nyquist frequency:
        :math:`\\Re(\\lambda_i)==0`

        keeps the second occurance of a conjugate pair (usually the one
        with the negative imaginary part)

        eigvec_l.shape = [order+1, order+1]
        eigval.shape = [order+1,1]
        '''

        num_val = len(eigval)
        conj_indices = deque()

        for i in range(num_val):
            this_val = eigval[i]
            this_conj_val = np.conj(this_val)
            # remove overdamped poles  i.e. real eigvals
            if this_val == this_conj_val:
                conj_indices.append(i)
            # remove negatively damped poles i.e. unstable poles
            elif np.abs(this_val) > 1:
                conj_indices.append(i)
            # catches unordered conjugates but takes slightly longer
            for j in range(i + 1, num_val):
                if eigval[j] == this_conj_val:
                    conj_indices.append(j)
                    break

        conj_indices = list(set(range(num_val)).difference(conj_indices))

        if inds_only:
            return conj_indices

        if eigvec_l is None:

            eigvec_r = eigvec_r[:, conj_indices]
            eigval = eigval[conj_indices]

            return eigval, eigvec_r

        else:
            eigvec_l = eigvec_l[:, conj_indices]
            eigvec_r = eigvec_r[:, conj_indices]
            eigval = eigval[conj_indices]

            return eigval, eigvec_l, eigvec_r

    @classmethod
    def init_from_config(cls, conf_file, prep_data):
        '''
        A method for initializing a modal object from configuration data
        bypassing common operations in explicit code for semi-automated
        analyses

        This is a stub of the method that must be reimplemented by every
        derived class

        '''

        assert os.path.exists(conf_file)
        assert isinstance(prep_data, PreProcessSignals)

        with open(conf_file, 'r') as _:
            # read configuration parameters line by line
            pass

        modal_object = cls(prep_data)

        return modal_object

    @staticmethod
    def integrate_quantities(vector, accel_channels, velo_channels, omega):
        # input quantities = [a, v, d]
        # output quantities = [d, d, d]
        # converts amplitude and phase
        #                     phase + 180; magn / omega^2

        vector[accel_channels] *= -1 / (omega ** 2)
        #                    phase + 90; magn / omega
        vector[velo_channels] *= 1j / omega

        return vector

    @staticmethod
    def rescale_mode_shape(modeshape, doehler_style=False):
        # scaling of mode shape
        if doehler_style:
            k = np.argmax(np.abs(modeshape))
            alpha = np.angle(modeshape[k])
            return modeshape * np.exp(-1j * alpha)
        else:
            modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
            return modeshape

    def save_state(self, fname):
        '''
        Saves the state of the object to a compressed numpy archive file
        This is only a stub for reimplementing the method in a derived class
        '''

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name

        raise NotImplementedError(
            'This method must be fully reimplemented by every derived class.')

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_data):
        '''
        Loads the state of the object from a compressed numpy archive file
        and returns the object
        This is only a stub for reimplementing the method in a derived class
        '''

        print('Now loading previous results from  {}'.format(fname))

        assert os.path.exists(fname)
        assert isinstance(prep_data, PreProcessSignals)
        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['',
                                                    ]):
            if this_state:
                print(state_string)

        setup_name = str(in_dict['self.setup_name'].item())
        assert setup_name == prep_data.setup_name

        modal_object = cls(prep_data)
        modal_object.state = state

        raise NotImplementedError(
            'This method must be fully reimplemented by every derived class.')

        return modal_object

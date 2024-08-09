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

Created on 08.03.2021

@author: womo1998
'''
import numpy as np

def nearly_equal(a, b, sig_fig=5):
    return (a == b or
            int(a * 10**sig_fig) == int(b * 10**sig_fig)
            )


def simplePbar(total):
    '''
    Divide the range of total in 100 discrete steps
        if total < 100 i.e. stepsize > 1: some steps may occur twice, printout must occur multiple times
        if total > 100 i.e. stepsize < 1: there are gaps, where no printout occurs
    For each call raise the step value by stepsize until step = total*100/total
    Check if the step is approximately 100
    
    For the last call, additionally a carriage return must be printed
    
    '''
    stepsize = 100 / total
    last = 0
    ncalls = 0
    while True:
        ncalls += 1
        while ncalls * stepsize // 1 > last:
            print('.', end='', flush=True)
            last += 1
        if ncalls == total:#np.isclose(step, 100):
            print('', end='\n', flush=True)
        yield


def calc_xyz(az, elev, r=1):
    if np.abs(az) > 2 * np.pi:
        print('You probably forgot to convert to radians ', az)
    if np.abs(elev) > 2 * np.pi:
        print('You probably forgot to convert to radians ', elev)
    # for elevation angle defined from XY-plane up
    x = r * np.cos(elev) * np.cos(az)
    # x=r*np.sin(elev)*np.cos(az) # for elevation angle defined from Z-axis
    # down
    # for elevation angle defined from XY-plane up
    y = r * np.cos(elev) * np.sin(az)
    # y=r*np.sin(elev)*np.sin(az)# for elevation angle defined from Z-axis down
    z = r * np.sin(elev)  # for elevation angle defined from XY-plane up
    # z=r*np.cos(elev)# for elevation angle defined from Z-axis down

    # correct numerical noise
    for a in (x, y, z):
        if np.allclose(a, 0):
            a *= 0

    return x, y, z

def get_method_dict():
    from pyOMA.core.PLSCF import PLSCF
    from pyOMA.core.PRCE import PRCE
    from pyOMA.core.SSICovRef import BRSSICovRef, PogerSSICovRef
    from pyOMA.core.SSIData import SSIData, SSIDataMC
    from pyOMA.core.VarSSIRef import VarSSIRef
    method_dict = {'Reference-based Covariance-Driven Stochastic Subspace Identification': BRSSICovRef, 
                   'Reference-based Data-Driven Stochastic Subspace Identification': SSIDataMC, 
                   'Stochastic Subspace Identification with Uncertainty Estimation': VarSSIRef,
                   'Poly-reference Least Squares Complex Frequency': PLSCF,
                   'Poly-reference Complex Exponential': PRCE, }
    return method_dict
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

'''
Created on 08.03.2021

@author: womo1998
'''
import numpy as np


def nearly_equal(a,b,sig_fig=5):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )



def calc_xyz(az, elev, r=1):
    if np.abs(az)>2*np.pi:
        print('You probably forgot to convert to radians ',az)
    if np.abs(elev)>2*np.pi:
        print('You probably forgot to convert to radians ', elev)
    x=r*np.cos(elev)*np.cos(az) # for elevation angle defined from XY-plane up
    #x=r*np.sin(elev)*np.cos(az) # for elevation angle defined from Z-axis down
    y=r*np.cos(elev)*np.sin(az) # for elevation angle defined from XY-plane up
    #y=r*np.sin(elev)*np.sin(az)# for elevation angle defined from Z-axis down
    z=r*np.sin(elev)# for elevation angle defined from XY-plane up
    #z=r*np.cos(elev)# for elevation angle defined from Z-axis down
    
    #correct numerical noise
    for a in (x,y,z):
        if np.allclose(a,0): a*=0
    
    return x,y,z
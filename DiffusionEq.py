#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:17:44 2018

@author: sascha
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import animation as anim
from scipy.signal import savgol_filter
import time

class InitConcentration():
    
    def __init__(self, x, ss):
        #self.stepsize = (x[-1] - x[0])/(x.size-1)
        self.stepsize = ss
        self.range = [x[0], x[-1]] # From To
        print("Stepsize is %f"%self.stepsize)
    
    def function(self, x):
        return np.exp(-x**2)
    
    def Derivative(self, x):
        #extrapolatedpoint = np.array([ExtrapPoint])
        #xExt = np.concatenate((x[:, None], extrapolatedpoint[:, None]), axis = None)
        WindowSize = (2*np.around(x.size/100)+1).astype(int)
        x = savgol_filter(x,WindowSize,3)
        
        # Compute first derivative
        xder = np.diff(x) / self.stepsize   
            
        # Extrapolate continuation of derivative
        ExtrapPoint = xder[-2] + (xder[-2]-xder[-3])
        xder = np.concatenate((xder, ExtrapPoint), axis = None)
        
        #return xder
        xder_smoothed = savgol_filter(xder,WindowSize,3)
        return xder_smoothed
        
    def SecondDerivative(self, x):
        print('One...')
        FirstDeriv = self.Derivative(x)
        print('Two...')
        SecondDeriv = self.Derivative(np.squeeze(FirstDeriv))
        return SecondDeriv
    

stepsize = 0.1
dt = 0.01
# dt < (dx)^2/2
dt = (stepsize**2)/3

x = np.arange(-10, 10 + stepsize,stepsize)

# Diffusion constant D
D = 0.5
timesteps = 1000

conc = InitConcentration(x, stepsize)

concentr = np.zeros([timesteps, x.size])
concentr[0, :] = conc.function(x)
start = time.time()
for t in range(1, timesteps):
    print(t)
    print('Elapsed time is %f sec'%(time.time()-start))
    NablaConc = np.squeeze(conc.SecondDerivative(concentr[t - 1, :]))
    plt.plot(NablaConc, color = 'red')
    plt.plot(concentr[t-1, :], color = 'green')
    plt.ylim([-1, 1])
    plt.show()
    plt.show()
    dcdt = D * NablaConc
    concentr[t, :] = concentr[t - 1, :] + np.squeeze(dcdt)[0:x.size]*dt
    
    
"""Create Animation"""
fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-0.5, 1.1))

# create the first plot
point, = ax.plot(x, concentr[0, :], 'o')
#ax.scatter(L[0, :], L[1, :],color='red',s=10);
ax.legend()
ax.set_xlim([-10, 10])
ax.set_ylim([-0.5, 1.1])

# second option - move the point position at every frame
def update_point(n, x, concentr, point):
    point.set_data(np.array([x, concentr[n, :]]))
    return point

# fargs: additional arguments to pass to func (second positional argument) at each call
ani = anim.FuncAnimation(fig, update_point, timesteps, fargs=(x, concentr, point))
ani.save('/home/sascha/Desktop/test2.mp4', fps=40, extra_args=['-vcodec', 'libx264'])

plt.show()

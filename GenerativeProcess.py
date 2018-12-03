#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:23:31 2018

@author: Sascha Froelich (sascha.froelich@tu-dresden.de)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim

class GenerativeProcess:
    def S(self,x):
        return 1/(1+np.exp(-x))
    
    def softmax(self,alpha,k):
        #return np.exp(k*alpha)/sum(np.exp(k*alpha))
        """Implementation to avoid Overflows for large entries of alpha"""
        TempSum = 0
        for i in range(alpha.size):
            TempSum += (np.exp(alpha[i]-alpha))**k
    
        return 1/TempSum
        
    def run(self,N,kappa,timesteps,L,dt=0.1):
        if L.ndim == 1:
            L = np.reshape(L,[2, N])
            
        lbd = 1/8;
        
        rho = np.zeros([N,N]);
        for i in range(N):
            for j in range(N):
                if i == j:
                    rho[i, j] = 0
                elif j == i + 1:
                    rho[i, j] = 1.5
                elif j == i - 1:
                    rho[i, j] = 0.5
                else:
                    rho[i, j] = 1
        
        rho[-1, 0] = 1.5;
        rho[0, -1] = 0.5  
        
        History = np.zeros([N,timesteps]);
        a = np.zeros([N,1])
        a[0] = 0.5
        """
        a = np.array([[-5.2],
                      [-6.08],
                      [0.31],
                      [2.29],
                      [-2.55],
                      [-4.70]]);"""
        posHistory = np.zeros([2,timesteps]);
        
        for t in range(timesteps):
                da = kappa*(-lbd*a -rho@self.S(a) + np.ones([N,1])) + \
                        np.random.normal(0,0,size=a.size)
                a[:, 0] = a[:, 0] + da[:, 0]*dt
                
                pos = L @ self.softmax(a,2)
                posHistory[0, t] = pos[0]
                posHistory[1, t] = pos[1]
                History[:, t] = a[:, 0]
                
        HiddenStatesTrajectory = History
        ObservedTrajectory = posHistory
        return ObservedTrajectory, HiddenStatesTrajectory

    def savefig(self,N,kappa,timesteps,L,dt=0.1):
        _, History = self.run(N,kappa,timesteps,L,dt=0.1)
        plt.figure()
        plt.scatter(L[0, :], L[1, :]);
        plt.title('Attractor points')
        
        plt.figure()
        for n in range(N):
            plt.plot(range(timesteps), History[n, :], label='Variable %i'%n)
            plt.title('State a')
        plt.legend()
        plt.savefig('/home/sascha/Desktop/Pres Stefan/1100 timesteps/1100timesteps.png')
        plt.show()
        print('Figure saved as /home/sascha/Desktop/Python/SHC/figure')
    
    def figure(self,N,kappa,timesteps,L,dt=0.1):
        ObservedTrajectory, HiddenStatesTrajectory = self.run(N,kappa,timesteps,L,dt=0.1)
        plt.figure()
        plt.scatter(L[0, :], L[1, :]);
        plt.title('Attractor points')
        
        plt.figure()
        for n in range(N):
            plt.plot(range(timesteps), HiddenStatesTrajectory[n, :], label='Variable %i'%n)
            plt.title('Hidden State a')
        plt.legend()
        plt.show()
        
        return ObservedTrajectory, HiddenStatesTrajectory
    
    def animation(self,N,kappa,timesteps,L,dt=0.1,name='animation'):
        posHistory, _ = self.run(N,kappa,timesteps,L,dt=0.1)
        """Create Animation"""
        fig = plt.figure()
        ax = plt.axes(xlim=(L[0, :].min()-0.1, L[0, :].max()+0.1), ylim=(L[1, :].min()-0.1, L[1, :].max()+0.1))
        
        # create the parametric curve
        x=posHistory[0, :]
        y=posHistory[1, :]
        
        # create the first plot
        point, = ax.plot([x[0]], [y[0]], 'o')
        #ax.scatter(L[0, :], L[1, :],color='red',s=10);
        ax.legend()
        ax.set_xlim([L[0, :].min()-0.1, L[0, :].max()+0.1])
        ax.set_ylim([L[1, :].min()-0.1, L[1, :].max()+0.1])
        
        # second option - move the point position at every frame
        def update_point(n, x, y, point):
            point.set_data(np.array([x[n], y[n]]))
            return point
        
        # fargs: additional arguments to pass to func (second positional argument) at each call
        ani = anim.FuncAnimation(fig, update_point, timesteps, fargs=(x, y, point))
        ani.save('/home/sascha/Desktop/Python/SHC/{0}.mp4'.format(name), fps=30, extra_args=['-vcodec', 'libx264'])
        
        plt.show()
        
        print("Animation saved as /home/sascha/Desktop/Python/SHC/{0}.mp4".format(name))

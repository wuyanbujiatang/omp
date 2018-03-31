#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 08:37:38 2018
@author: wsj

"""
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
def omp1(y,phi,psi,K):
    A = phi*psi.H
    M,N = np.shape(A)
    s = np.zeros((N,1))
    v = y
    num_iters = K
    aug_A = []
    for i in range(num_iters):
        product = A.H*v
        g = (abs(product)).tolist()
        pos = g.index(max(g))
        if len(aug_A)==0:
            aug_A = copy.copy(A[:,pos])
        else:
            m,n = np.shape(aug_A)
            b = copy.copy(A[:,pos])
            for j in range(n):
                b = b - aug_A[:,j]*aug_A[:,j].H*A[:,pos]/(aug_A[:,j].H*aug_A[:,j])
            aug_A = np.hstack((aug_A,b))
#        A = np.delete(A,pos,axis=1)
        A[:,pos] = np.zeros((M,1))
        weight = aug_A[:,-1].H*y/(aug_A[:,-1].H*aug_A[:,-1])
        v = v - aug_A[:,-1]*weight
        s[pos] = s[pos] + weight
        if np.linalg.norm(v)<1e-6:
            break
    return psi.H*s
def omp2(y,phi,psi,K):
    A = phi*psi.H
    M,N = np.shape(A)
    v = y
    num_iters = K
    aug_A = []
    pos_array = np.zeros(num_iters)
    s = np.zeros((N,1))
    for i in range(num_iters):
        product = A.H*v
        g = (abs(product)).tolist()
        pos = g.index(max(g))
        if len(aug_A)==0:
            aug_A = copy.copy(A[:,pos])
#            print(aug_A)
        else:
            aug_A = np.hstack((aug_A,A[:,pos]))
#            print(aug_A)
        weight = (aug_A.H*aug_A).I*aug_A.H*y
        A[:,pos] = np.zeros((M,1))
#        print(aug_A)
        v = y - aug_A*weight
        pos_array[i] = pos
        print(pos_array)
        if np.linalg.norm(v)<1e-11:
            break
    s[pos_array.tolist()] = weight
    return psi.H*s
def my_plot(x,x_r):
    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(8,6))  
    line1, = axes.plot(x_r,'r-s')
    line2, = axes.plot(x,'b--o')
    axes.legend((line1,line2),('recovered signal','src signal'),loc = 'upper right')
    axes.set_xlabel(u'time')
    axes.set_ylabel(u'signal')
    axes.set_title('omp algorithm')
if __name__ == '__main__':
    N = 256
    M = 64
    K = 8
    x = np.zeros((N,1))
    index_k = random.sample(range(N),K)
    x[index_k] = 5*np.random.randn(K,1)
    psi = np.eye(N,N)
    phi = np.random.randn(M,N)
    phi = np.mat(phi)
    psi = np.mat(psi)
    x = np.mat(x)
    y = phi*psi.H*x
    my_plot(omp2(y,phi,psi,K),x)
#    print(mp(y,phi,psi,K)-x)
    

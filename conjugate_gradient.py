# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:23:31 2019

@author: jiryi
"""

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

def data_prepare(dimension):
    
    basis1 = [1.0/(i+1) for i in np.arange(dimension)]
    basis2 = [1.0/(dimension+i) for i in np.arange(dimension)]
    A = spla.hankel(basis1,basis2)
    b = np.ones((dimension,))
    
    return A,b

def conjugate_gradient(x):
    
    # data preparation
    dimension = len(x)
    A, b = data_prepare(dimension)
    r = np.matmul(A,x) - b
    p = -r
    r_norm = np.linalg.norm(r)
    r_norm_arr = []
    r_norm_arr.append(r_norm)
    
    # parameter set up
    tol = 10**(-6)
    MaxIte = 500
    i = 0
    
    print("\n")
    print("Now report results in %s-D case" % dimension)
    while r_norm > tol:
        
        alpha = np.dot(r,r) / np.dot(p,np.matmul(A,p))
        x = x + alpha*p
        r_prev = r
        r = r + alpha*np.matmul(A,p)
        beta = np.dot(r,r) / np.dot(r_prev,r_prev)
        p = -r + beta*p
        r_norm = np.linalg.norm(r)
        
        i = i+1
        r_norm_arr.append(r_norm)
        
        if i%50 == 0:
            print("Iteration: {}, residual norm: {}\n".format(i,r_norm))
        
        if i > MaxIte:
            print("Cannot converge within {} iterations.\n".format(MaxIte))
            break
    
    print("Terminate at iteration {}, the residual norm is {}\n".format(i,r_norm))
    print("the final solution is:\n {}".format(x))
    
    return x,r_norm_arr

def vis(residual_norm,dimension):
    plt.figure()
    plt.plot(np.log10(r_norm_arr),"-*")
    plt.xlabel("# of iterations")
    plt.ylabel("residual norm in log10 scale")
    plt.title("%s-D case" % dimension)
    plt.show()

if __name__ == "__main__":
    
    dimension = [5, 8, 12, 20]
    for d in dimension:
        
        x = np.zeros((d,))
        x,r_norm_arr = conjugate_gradient(x)
        vis(r_norm_arr,d)
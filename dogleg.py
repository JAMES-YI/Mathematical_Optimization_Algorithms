# -*- coding: utf-8 -*-
"""
This file is to demonstrate the application of dogleg method 
in trust region algorithm to find solution for the Rosenbrock function.

Created on Sat Feb 23 12:35:48 2019

@author: jiryi
"""

import numpy as np
import matplotlib.pyplot as plt
import rosenbrock as rb

# In[], trust region algorithm

def TR_search(x):
    """
    r_upper,
    eta,
    tau, 
    """
    
    r_upper = 2; eta = 0.2; 
    tol = 10**(-3); maxite = 500; 
    funceval = rb.rosenbrock_funcval
    gradeval = rb.rosenbrock_gradval
    hesseval = rb.rosenbrock_hessval
    
    grad = gradeval(x).reshape((2,))
    r = 0.5; i = 0; gradn = np.linalg.norm(grad); 
    # tau = 1.5
    
    func_arr = []; gradn_arr = []; x_arr = []
    x_arr.append(x)
    
    while True:
        
        if i>maxite or gradn<tol:
            break
        
        func = funceval(x)
        hess = hesseval(x)
        
        func_arr.append(func)
        gradn_arr.append(gradn)
        
        p = dogleg(grad,hess,r)
        p_n = np.linalg.norm(p) 
        
        func_next = funceval(x+p)
        mfunc = qdapp(func,grad,hess,np.zeros((2,)))
        mfunc_next = qdapp(func,grad,hess,p)
        rho = (func - func_next) / (mfunc - mfunc_next)
        
        # update TR radius
        if rho < 1/4:
            r = (1/4)*r
        else:
            if rho > 3/4 and p_n == r:
                r = np.min([2*r,r_upper])
            else: # 1/4 <= rho <= 3/4; rho > 3/4 and p_n != r
                r = r
        
        # update searching solution
        if rho > eta:
            x = x + p
        else:
            x = x
        
        grad = gradeval(x).reshape((2,))
        gradn = np.linalg.norm(grad)
        i = i+1
        x_arr.append(x)
    
    print('At iteration {}:\n'.format(i))
    print('\t the solution is {}'.format(x))
    print('\t the function value is {}'.format(func))
    print('\t the gradient is {}'.format(grad))
    print('\t the gradient norm is {}'.format(np.linalg.norm(grad)))
    
    return x_arr,func_arr,gradn_arr

# In[], dogleg search for direction
def dogleg(grad,hess,r):
    
    p_u = - grad * (np.dot(grad,grad)) / np.dot(grad,np.matmul(hess,grad))
    p_b = np.linalg.solve(hess,-grad)
    p_un = np.linalg.norm(p_u)
    if p_un < r:
        tau = 1.5
        p = p_u + (tau-1) * (p_b - p_u)
    elif p_un > r:
        tau = 0.5
        p = tau * p_u
            
    """
    if tau > 0 and tau <= 1:
        p = tau * p_u
    elif tau > 1 and tau <= 2:
        p = p_u + (tau-1) * (p_b - p_u)
    else:
        print("Error! The tau  should be in [0,2].\n")
    """
    return p

# In[], compute approximate quadratic m(p)
def qdapp(func,grad,hess,p):
    
    qdval = func + np.dot(grad,p) + \
            (1/2)*np.dot(p,np.matmul(hess,p))
    
    return qdval

# In[], at point [1.2,1.2], and [-1.2,1]
# converge, [1.2,1.2], [1.2,0.8], [2,-1], [0.6,0.6]
    
x1 = np.array([1.2,1.2])
x_arr1,func_arr1,gradn_arr1 = TR_search(x1)
x2 = np.array([2,-1])
x_arr2,func_arr2,gradn_arr2 = TR_search(x2)

plt.figure()
plt.plot(np.log10(func_arr1),'-*'); plt.plot(np.log10(func_arr2),'-*');
plt.legend(('Start [1.2,1.2]','Start [2,-1]'))
plt.xlabel('Iteration'); plt.ylabel('Function value in log scale');
plt.show()

plt.figure()
plt.plot(np.log10(gradn_arr1),'-*'); plt.plot(np.log10(gradn_arr2),'-*');
plt.legend(('Start [1.2,1.2]','Start [2,-1]'))
plt.xlabel('Iteration'); plt.ylabel('Gradient norm in log scale');
plt.show()

x_arr1 = np.concatenate(x_arr1).reshape((-1,2))
x_arr2 = np.concatenate(x_arr2).reshape((-1,2))
plt.figure()
plt.plot(x_arr1[:,0],x_arr1[:,1],'*'); plt.plot(x_arr2[:,0],x_arr2[:,1],'*')
plt.plot(1,1,'r^')
plt.legend(('Searching start [1.2,1.2]',
            'Searching start [2,-1]',
            'Optimal solution'))
plt.xlabel('x1'); plt.ylabel('x2');
plt.show()



#  

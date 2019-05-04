# -*- coding: utf-8 -*-
"""
This file is to demonstrate the steepest descent method and Newton method 
for solving 
    minimize_{x} f(x) := 100*(x_2 - x_1^2)^2 + (1-x_1)^2

x_k+1 = x_k + \alpha_k p_k
steepest descent method: p_k =  - gradf_k
newton's method: hessf_k p_k = - gradf_k
\alpha_k, by backtracking line search
                    
Created on Sat Feb  9 10:40:42 2019

@author: jiryi
"""

import numpy as np
import numpy.linalg as la
import rosenbrock as rb
import matplotlib.pyplot as plt

# In[] steepest descend algorithm
def steepes_sech(x_init):
    """
    x_init: initial point
    
    stopping criterion: grad(x) < 10^{-3} or MaxIte
    
    return: x, f(x), grad(x) at each iteration
    """
    MaxIte = 500; tol = 10**(-3); ite = 0;
    x = x_init
    gradval = rb.rosenbrock_gradval
    
    funcval_arr = []; gradn_arr = []; x_arr = []; 
    i = 0
    alp_arr = []
    while True:
        p = - np.array(gradval(x)).reshape((2,))
        alp, fcurr, gradcurr = bktk_step(x,p)
        alp_arr.append(alp)
        gradn = la.norm(gradcurr,2)
        x_arr.append(x); funcval_arr.append(fcurr); gradn_arr.append(gradn);
        if gradn < tol or ite > MaxIte:
            break
        x = x + alp*p; ite = ite + 1;
        i = i+1
        
    print("Steepest descent method, in final iteration {}\n".format(ite))
    print("the solution is\n {}".format(x))
    print("the function value is\n {}".format(fcurr))
    print("the gradient is\n {}".format(gradcurr))
    print("the visualization of logarithm of step size is\n")
    step_vis(alp_arr)
    
    return x_arr, funcval_arr, gradn_arr

# In[] newton's method
def newton_sech(x_init):
    """
    x_init: initial point
    
    stopping criterion: grad(x) < 10^{-3} or MaxIte
    
    return: x, f(x), grad(x) at each iteration
    """
    MaxIte = 500; tol = 10**(-3); ite = 0;
    x = x_init
    gradval = rb.rosenbrock_gradval; hessval = rb.rosenbrock_hessval;
    x_arr = []; funcval_arr = []; gradn_arr = [];
    
    i = 0
    alp_arr = []
    
    while True:
        gradcurr = gradval(x).reshape((2,)); hesscurr = hessval(x);
        p =  - la.solve(hesscurr,gradcurr)
        alp, fcurr, _ = bktk_step(x,p)
        alp_arr.append(alp)
        gradn = la.norm(gradcurr,2)
        x_arr.append(x); funcval_arr.append(fcurr); gradn_arr.append(gradn);
        if gradn < tol or ite > MaxIte:
            break
        x = x + alp*p; ite = ite + 1;
        i = i+1
    
    print("Newton's method, in the final iteration {}\n".format(ite))
    print("the solution is\n {}".format(x))
    print("the function value is\n {}".format(fcurr))
    print("the gradient is\n {}".format(gradcurr))
    print("the visualization of logarithm of step size is\n")
    step_vis(alp_arr)
    
    return x_arr, funcval_arr, gradn_arr

# In[] backtracking line search
def bktk_step(x,p,alp_init=1,alp_sc=0.98,augf=0.7):
    """
    backtracking line seach for finding step size
    
    x, current point
    p, searching direction
    alp_init: initial step size
    alp_sc: scaling factor for controlling step size
    augf: scaling factor for the linear term
    
    return alp
    
    alp_init, alp_sc, augf: 1, 0.98, 0.7 can guarantee convergence in 
                            about 300 iterations for (1.2,1.2)
                            about 30 iterations for (-1.2,1)
        
    """
    alp = alp_init
    funcval = rb.rosenbrock_funcval
    gradval = rb.rosenbrock_gradval
    
    fcurr = funcval(x); gradcurr = np.array(gradval(x)).reshape((2,));
    while True:
        fnext = funcval(x+alp*p)
        fnext_app = fcurr + augf*alp*np.dot(np.transpose(gradcurr),p)
        if fnext <= fnext_app:
            break
        alp = alp_sc*alp
    return alp, fcurr, gradcurr

# In[] searching path visualization
def path_vis(x_arr):
    
    return 0

# In[] function value and gradient norm visualization
def visualization(funcval_arr, gradn_arr):
    plt.figure()
    plt.plot(funcval_arr,'r*--',linewidth=2,markersize=12)
    plt.xlabel('Num of iteration')
    plt.ylabel('Function value')
    plt.show()
    
    plt.figure()
    plt.plot(gradn_arr,'go-',linewidth=2,markersize=12)
    plt.xlabel('Num of iteration')
    plt.ylabel('Gradient norm')
    plt.show()
    return 0

def step_vis(alp_arr):
    plt.figure()
    plt.plot(np.array(alp_arr),'*-')
    plt.title('Step size $alpha$')
    plt.xlabel('Num of iteration')
    plt.ylabel('$log(alpha)$')


if __name__ == "__main__":
    x_start1 = np.array([1.2,1.2])
    x_start2 = np.array([-1.2,1])
    
    print("Steepest descent algorithm, with x0 =\n {}".format(x_start1))
    x_sdstart1, func_sdstart1, gradn_sdstart1 = steepes_sech(x_start1)
    visualization(func_sdstart1,gradn_sdstart1)
    path_vis(x_sdstart1)
    print("Steepest descent algorithm, with x0 =\n {}".format(x_start2))
    x_sdstart2, func_sdstart2, gradn_sdstart2 = steepes_sech(x_start2)
    visualization(func_sdstart2,gradn_sdstart2)
    path_vis(x_sdstart2)
    
    print("Newton's algorithm, with x0 =\n {}".format(x_start1))
    x_ntstart1, func_ntstart1, gradn_ntstart1 = newton_sech(x_start1)
    visualization(func_ntstart1,gradn_ntstart1)
    path_vis(x_ntstart1)
    print("Newton's algorithm, with x0 =\n {}".format(x_start2))
    x_ntstart2, func_ntstart2, gradn_ntstart2 = newton_sech(x_start2)
    visualization(func_ntstart2,gradn_ntstart2)
    path_vis(x_ntstart2)
    
    



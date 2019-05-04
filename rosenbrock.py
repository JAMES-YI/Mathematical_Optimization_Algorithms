# -*- coding: utf-8 -*-
"""
This file is to demonstrate HW2 for MATH optimization technique

rosenbrock function: f(x) = 100*(x_2 - x_1^2)^2 + (1-x_1)^2
grad(f(x)) = [-400*x1*x2 + 400*x1^3 + 2*x1 - 2;
                200*x2 - 200*x1^2]
Hess(f(x)) = [-400*x2 + 1200*x1^2 + 2, -400*x1;
              -400*x1, 200]
 
Created on Sat Feb  2 15:21:36 2019

@author: jiryi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# rosenbrock function
def rosenbrock_funcval(x):
    fval_x = 100*(x[1] - x[0]**2)**2 + (1-x[0])**2
    return fval_x

# gradient calculation
def rosenbrock_gradval(x):
    pgrad_1 = - 400*x[0]*x[1] + 400*x[0]**3 + 2*x[0] - 2
    pgrad_2 = 200*x[1] - 200*(x[0]**2)
    grad = np.array([pgrad_1,pgrad_2]).reshape((2,1))
    return grad
    
# Hessian calculation
def rosenbrock_hessval(x):
    pgrad_11 = -400*x[1] + 1200*x[0]**2 + 2
    pgrad_12 = -400*x[0]
    pgrad_22 = 200
    hess = np.array([[pgrad_11, pgrad_12],[pgrad_12, pgrad_22]])
    return hess

# function surface and contour
def rosenbrock_vis():
    resolution = 0.01
    x0 = np.arange(-6.0,6.0,resolution)
    x1 = np.arange(-2.0,2.0,resolution)
    X0,X1 = np.meshgrid(x0,x1)
    funcval = 100*(X0 - X1**2)**2 + (1-X0)**2
    
    plt.figure()
    plt.contour(X0,X1,funcval,10)
    plt.title('Contour of Rosenbrock function')
    plt.xlabel('x1'); plt.ylabel('x2'); 
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X0,X1,funcval)
    ax.scatter(1,1,0,c='r',marker='+',s=8*20)
    plt.title('Function surface')
    plt.xlabel('x1'); plt.ylabel('x2');
    # plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # test, function value, gradient, hessian at [1,1]
    x = np.array([2,1])
    func = rosenbrock_funcval(x)
    grad = rosenbrock_gradval(x)
    hess = rosenbrock_hessval(x)
    print('At point {}'.format(x))
    print('the function value is {}\n'.format(func))
    print('the gradient is\n {}'.format(grad))
    print('the Hessian is\n {}'.format(hess))
    rosenbrock_vis()
    
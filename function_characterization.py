# -*- coding: utf-8 -*-
"""
This file is to compute the

(1) f(x) = 10(x2 - x1^2)^2 + (1-x1)^2
(2) grad_f(x) = [-40*x1*x2 + 40*x1^3 + 2*x1 - 2,
             20*x2 - 20*x1^2]
(3) hess_f(x) = [-40*x2 + 120*x1^2 + 2, -40*x1;
             -40*x1, 20]
(4) contour of m(p) at fixed x
    m(p) = f(x) + grad_f(x)^Tp + (1/2) * p^T*hess_f(x)*p
         = f(x) + sum_i [grad_f(x)]_i p_i + sum_{i,j} (1/2) [hess_f(x)]_{ij} p_ip_j
(5) solve TRP under different radius r at fixed x
    min_p f(x) + grad_f(x)^Tp + (1/2) * p^T*hess_f(x)*p
    s.t    ||p|| <= r

Created on Sat Feb 23 10:37:27 2019

@author: jiryi
"""

import numpy as np
import matplotlib.pyplot as plt

# In[], compute function value, gradient, and Hessian

def funceval(x):
    func = 10*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    return func

def gradeval(x):
    gradp1 = -40*x[0]*x[1] + 40*x[0]**3 + 2*x[0] - 2
    gradp2 = 20*x[1] - 20*x[0]**2
    grad = np.array([gradp1,gradp2])
    return grad

def hesseval(x):
    hessp11 = -40*x[1] + 120*x[0]**2 + 2
    hessp12 = -40*x[0]
    hessp22 = 20
    hess = np.array([[hessp11, hessp12],
                     [hessp12, hessp22]])
    return hess

# In[], results visualization

def mpcontour(func,grad,hess):
    
    pd1 = np.arange(-3,3,0.1)
    pd2 = np.arange(-3,3,0.1)
    Pd1,Pd2 = np.meshgrid(pd1,pd2)
    m = func + (grad[0]*Pd1 + grad[1]*Pd2) + \
        (1/2)*(hess[0,0]*Pd1**2 + hess[0,1]*Pd1*Pd2 + hess[1,0]*Pd2*Pd1 + hess[1,1]*Pd2**2)
    circle = plt.Circle((0,0),2,color='r')
    fig,ax = plt.subplots()
    plt.contour(Pd1,Pd2,m,10)
    ax.add_artist(circle)
    plt.xlabel("x_1"); plt.ylabel("x_2");
    plt.title("Contour of m(p) and trust region")
    plt.show()

if __name__ == "__main__":
    
    # In[], at point x = [0,-1]
    x1 = np.array([0,-1])
    func1 = funceval(x1)
    grad1 = gradeval(x1)
    hess1 = hesseval(x1)
    print("At point {}:\n".format(x1))
    print("\t the function value is {},\n".format(func1))
    print("\t the gradient is {},\n".format(grad1))
    print("\t the Hessian is {},\n".format(hess1))
    print("\t the contour of m(p) is \n")
    mpcontour(func1,grad1,hess1)
    
    # when r = 0
    
    # when r = 2
    
    # In[], at point x = [0,0.5]
    x2 = np.array([0,0.5])
    func2 = funceval(x2)
    grad2 = gradeval(x2)
    hess2 = hesseval(x2)
    print("At point {}:\n".format(x2))
    print("\t the function value is {},\n".format(func2))
    print("\t the gradient is {},\n".format(grad2))
    print("\t the Hessian is {}.\n".format(hess2))
    print("\t the contour of m(p) is \n")
    mpcontour(func2,grad2,hess2)
    
    # when r = 0
    
    # when r = 2


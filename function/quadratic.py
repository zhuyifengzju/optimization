"""
Class for rosenbrock function

Rosenbrock function is a non-convex function for testing optimization algorithm.

Yifeng Zhu, 2018
"""
import numpy as np

class QuadraticFunction():
    """
    Simple case for quadratic function f(x) = a * ||x||_{2}

    """
    
    def __init__(self, a=1):
        """
        inputs:
           a: default 1, a common value
        """
        self.a = a

    def value(self, x):
        """
        inputs:
            x: input vector of n dims
        outputs:
            func_value: f(x)
        """
        ndims = x.shape[0]
        func_value = 0
        for i in range(ndims):
            func_value += x[i] ** 2
            
        return self.a * func_value

    def grad_value(self, x):
        """
        inputs:
            x: input vector of n dims
        outputs:
            a list of grad_value
        """
        grad_value = []
        ndims = x.shape[0]
        for i in range(ndims):
            grad_value.append(self.a * 2 * x[i])

        return np.array(grad_value)

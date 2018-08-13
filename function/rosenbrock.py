"""
Class for rosenbrock function

Rosenbrock function is a non-convex function for testing optimization algorithm.

Yifeng Zhu, 2018
"""
import numpy as np

class RosenbrockFunction():
    """
    Rosenbrock function has the form: f(x,y) = (a - x)^{2} + b (y - x^{2})^{2} in 2 dimension case

    For N dimension case, f(x) = \sum_{i=1}^{N-1}[100(x_{i+1} - x_{i}^{2})^{2} + (1-x_{i})^{2}]
    
    """
    
    def __init__(self, a=1, b=100):
        """
        inputs:
           a: default 1, a common value
           b: default 100, a common value
        """
        self.a = a
        self.b = b

    def value(self, x):
        """
        inputs:
            x: input vector of n dims
        outputs:
            func_value: f(x)
        """
        ndims = x.shape[0]
        func_value = 0
        for i in range(ndims-1):
            func_value += (self.a - x[i]) ** 2 + self.b * (x[i+1] - x[i]**2) ** 2
        return func_value

    def grad_value(self, x):
        """
        inputs:
            x: input vector of n dims
        outputs:
            a list of grad_value
        """
        grad_value = []
        ndims = x.shape[0]
        for i in range(ndims-1):
            grad_value.append(2*(x[i] - self.a) + 2 * self.b * (x[i+1] - x[i]**2) * (-2 * x[i]))

        grad_value.append(2 * self.b * (x[i+1] - x[i] ** 2))
        return np.array(grad_value)

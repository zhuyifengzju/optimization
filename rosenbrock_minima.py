"""Script to run different optimization methods on rosenbrock funciton."""

from function import *
from algorithm import steepest_descent
import numpy as np

def main():

    frosenbrock = RosenbrockFunction()
    f, grad = frosenbrock.value, frosenbrock.grad_value

    optimizer = steepest_descent.SteepestDescent(f,
                                                 grad,
                                                 1e-4,
                                                 0.9,
                                                 iterations=100)

    x = optimizer.optimize(x0=np.array([2, 10, 1, 10, 0]))
    print(f'Final Result x={x}, f(x)={f(x)}')


    


if __name__ == '__main__':
    main()

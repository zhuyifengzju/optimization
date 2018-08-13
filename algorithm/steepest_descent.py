"""Implementation of SteepestDescent."""

import numpy as np

def wolfe_cond(f, grad, xk, alphak, pk, c1):

    return f(xk + alphak * pk) <= f(xk) + c1 * alphak * np.dot(grad(xk), pk)


def strong_wolfe_cond(f, grad, xk, alphak, pk, c1, c2):
    return (wolfe_cond(f, grad, xk, alphak, pk, c1)
            and abs(np.dot(grad(xk+alphak), pk)) <= c2 * abs(np.dot(grad(xk), pk)))

class SteepestDescent():

    def __init__(self, f, grad, c1=1e-4, c2=0.9, iterations=50, error = 1e-6):
        self.f = f
        self.grad = grad
        self.c1 = c1
        self.c2 = c2
        self.iterations = iterations
        self.err = error

    def line_search(self, xk, pk):
        """
        Implementation from
        https://github.com/stormmax/non-convex/blob/master/non-convex.py

        TODO(yifeng): Still not sure why it's implemented like this.

        """
        alpha = 1.0
        f_alpha = lambda alpha: self.f(xk + alpha * pk)
        grad_alpha = lambda alpha: np.dot(self.grad(xk + alpha * pk), pk)
        strong_wolfe = lambda f, grad, alpha, pk, c1, c2: strong_wolfe_cond(f, grad, xk, alpha, pk, c1, c2)

        l = 0.0
        h = 1.0
        for i in range(20):
            if strong_wolfe(self.f, self.grad, alpha, pk, self.c1, self.c2):
                return alpha

            half = (l + h) / 2
            alpha = - grad_alpha(l) * (h ** 2) / (2 * (f_alpha(h) - f_alpha(l) - grad_alpha(l) * h))
            if alpha < l or alpha > h:
                alpha = half

            if grad_alpha(alpha) > 0:
                h = alpha
            elif grad_alpha(alpha) <= 0:
                l = alpha
        return alpha

        

    def optimize(self, x0):
        curr_x = np.array(x0)
        prev_x = curr_x

        for i in range(self.iterations+1):
            pk = - self.grad(curr_x)
            alpha = self.line_search(curr_x, pk)

            curr_x = curr_x + alpha * pk

            if i % 10 == 0:
                print(f'iter={i}, x={curr_x}, f(x)={self.f(curr_x)}, alpha={alpha}')

            if np.linalg.norm(curr_x - prev_x) < self.err:
                break
            prev_x = curr_x
        return curr_x
    
        
        




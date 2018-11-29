from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import GPy
from DIRECT import solve
from scipy.optimize import minimize

from objective_functions import *

class InclineOptimizer:
    def __init__(self, obj_f, n_inputs, lower_bounds, upper_bounds, \
        context_space):
        '''
        Contextual BO for learning to climb inclines

        obj_f:          Objective function
        n_inputs:       Number of inputs
        lower_bounds:   Sequence of lower bounds for the parameters (ordered)
        upper_bounds:   Sequence of upper bounds for the parameters (ordered)
        context_space:  Sequence of inclines to sample from
        '''
        self.obj_f = obj_f
        self.n_inputs = n_inputs
        self.n_contexts = 1
        self.context_index = 0
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.context_space = context_space

        self.parameters = np.concatenate([np.random.uniform(lower_bounds[i],
        upper_bounds[i], (10, 1)) for i in range(n_inputs)], axis=1)
        self.contexts = np.random.choice(context_space, (10, 1))
        self.X = np.concatenate((self.parameters, self.contexts), axis=1)
        self.Y = np.array([self.obj_f(self.X[i]) for i in range(10)])

        self.train_GP(self.X, self.Y)
        self.model.optimize()
        self.iterations = 0

    def train_GP(self, X, Y, kernel=None):
        '''
        Trains the GP model. The Matern 5/2 kernel is used by default

        X:      A 2-D input vector containing both parameters and context
        Y:      A 2-D output vector containing objective values
        kernel: See the GPy documentation for other kernel options
        '''
        if not kernel:
            kernel = GPy.kern.Matern52(input_dim=self.n_inputs + 1, ARD=True)
        self.model = GPy.models.GPRegression(X, Y, kernel)

    def plot(self, visible_dims=None):
        '''
        Wrapper function for GPy's GP plotting function.
        To specify the arguments further, call the function manually with:
        >> self.model.plot(...)

        visible_dims:   Sequence of parameter indices to be plotted (max 2)
                        Uses the first two parameters by default
        '''
        iters = range(1, self.iterations + 11)
        objectives = self.Y
        plt.plot(iters, objectives)
        plt.xlabel('Iterations')
        plt.ylabel('Objective value')
        plt.show()

    def optimize(self, n_iterations):
        for _ in range(n_iterations):
            try:
                self.iterations += 1
                print('Iteration ' + str(self.iterations))

                context = [self.context_space[self.context_index]]
                self.context_index = (self.context_index + 1) % \
                    len(self.context_space)

                # DIRECT
                def DIRECT_obj(x, user_data):
                    return -self.acq_f(np.concatenate((x, context))), 0
                x, fmin, ierror = solve(DIRECT_obj, self.lower_bounds, \
                    self.upper_bounds, maxf=500)

                # L-BFGS-B
                def LBFGS_obj(x):
                    return -self.acq_f(np.concatenate((x, context)))
                bounds = [(self.lower_bounds[i], self.upper_bounds[i]) \
                    for i in range(self.n_inputs)]
                res = minimize(LBFGS_obj, x, method='L-BFGS-B', bounds=bounds)

                self.parameters = np.concatenate((self.parameters, \
                    np.reshape(res.x, (1,self.n_inputs))))
                self.contexts = np.concatenate((self.contexts, \
                    np.reshape(context, (1,1))))
                self.X = np.concatenate((self.parameters,self.contexts), axis=1)
                self.Y = np.concatenate((self.Y, \
                    [self.obj_f(np.concatenate((res.x, context)))]))

                self.train_GP(self.X, self.Y)
                self.model.optimize()
            except Exception:
                print('Exception encountered during optimization')
                traceback.print_exc()

    def acq_f(self, x, alpha=-1, v=.01, delta=.1):
        x = np.reshape(x, (1, self.n_inputs + self.n_contexts))
        mean, var = self.model.predict(x)
        if alpha is -1:
            alpha = np.sqrt(v*(2*np.log((self.iterations**(((self.n_inputs) \
                / 2) + 2)) * (np.pi**2) / (3 * delta))))
        return mean + (alpha * var)

    def predict_optimal(self, context):
        context = np.array([context])
        def DIRECT_obj(x, user_data):
            return -self.acq_f(np.concatenate((x, context)), alpha=0), 0
        x,fmin,ierror= solve(DIRECT_obj, self.lower_bounds, self.upper_bounds)

        def LBFGS_obj(x):
            return -self.acq_f(np.concatenate((x, context)), alpha=0)
        bounds = [(self.lower_bounds[i], self.upper_bounds[i]) \
            for i in range(self.n_inputs)]
        res = minimize(LBFGS_obj, x, method='L-BFGS-B', bounds=bounds)
        return res.x

init_vrep()
co = InclineOptimizer(incline_obj_f, 3, [1, 0, 0], [60, 2 * np.pi, 1], [5,10,15])
co.optimize(50)
co.plot()
exit_vrep()

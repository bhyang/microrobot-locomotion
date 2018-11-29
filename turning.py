from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import GPy
import scipy
from DIRECT import solve
from scipy.optimize import minimize

from objective_functions import *

class TurningOptimizer:
    def __init__(self, obj_f, n_parameters, lower_bounds, upper_bounds, \
        context_space):
        '''
        Contextual BO for turning to target trajectories

        obj_f:          Objective function
        n_inputs:       Number of inputs
        lower_bounds:   Sequence of lower bounds for the parameters (ordered)
        upper_bounds:   Sequence of upper bounds for the parameters (ordered)
        context_space:  Sequence of targets to sample from
        '''
        self.obj_f = obj_f
        self.n_parameters = n_parameters
        self.n_contexts = 2

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.context_space = context_space

        # Initialize first 10 random guesses
        self.parameters = np.concatenate([np.random.uniform(lower_bounds[i],
        upper_bounds[i], (10, 1)) for i in range(n_parameters)], axis=1)
        self.contexts = np.stack([context_space[i % len(context_space)] \
            for i in range(10)])
        self.X = np.concatenate((self.parameters, self.contexts), axis=1)

        objs = []
        infos = []
        for i in range(10):
            obj, info = self.obj_f(self.X[i])
            objs.append(obj)
            infos.append(info)

        self.Y = np.array(objs)
        self.info = np.array(infos)

        # Fit dataset to GP model
        self.train_GP(self.X, self.Y)
        self.model.optimize()
        self.iterations = 0

    def train_GP(self, X, Y, kernel=None):
        '''
        Trains the GP model from scratch given inputs X and outputs Y
        The Matern 5/2 kernel is recommended by default

        X:      A 2-D input vector containing both parameters and context
        Y:      A 2-D output vector containing objective values
        kernel: See the GPy documentation for other kernel options
        '''
        if not kernel:
            kernel = GPy.kern.Matern52(input_dim=self.n_parameters + \
                self.n_contexts)
        self.model = GPy.models.GPRegression(X, Y, kernel)

    def plot(self, visible_dims=None):
        '''
        Wrapper function for GPy's GP plotting function.
        To specify the arguments further, call the function manually with:
        >> self.model.plot(...)

        visible_dims:   Sequence of parameter indices to be plotted (max 2)
                        Uses the first two parameters by default
        '''
        # Plotting learning curve
        assert self.iterations != 0, 'Unable to plot, not enough iterations'
        iters = range(1, self.iterations + 11)
        objectives = self.Y
        plt.plot(iters, objectives)
        plt.xlabel('Iterations')
        plt.ylabel('Objective value (m)')

        # Plotting parameters/objective graph
        if visible_dims:
            self.model.plot(visible_dims=visible_dims)
        else:
            self.model.plot(visible_dims=[0, 1])

        plt.xlabel('Frequency (radians / second)')
        plt.ylabel('Phase difference (radians)')
        plt.show()

    def optimize(self, n_iterations):
        '''
        Iteratively optimizes the objective cost function and updates GP model.

        n_iterations:   Number of optimization iterations to run
        '''
        for _ in range(n_iterations):
            try:
                self.iterations += 1
                print('Iteration ' + str(self.iterations))
                context = self.context_space[self.iterations % \
                    len(self.context_space)]

                # DIRECT
                def DIRECT_obj(x, user_data):
                    return self.acq_f(np.concatenate([x, context], axis=0)), 0
                x, fmin, ierror = solve(DIRECT_obj, \
                    self.lower_bounds[:self.n_parameters], \
                    self.upper_bounds[:self.n_parameters], maxf=500)

                # L-BFGS-B
                def LBFGS_obj(x):
                    return self.acq_f(np.concatenate([x, context], axis=0))
                bounds = [(self.lower_bounds[i], self.upper_bounds[i]) \
                    for i in range(self.n_parameters)]
                res = minimize(LBFGS_obj, x, method='L-BFGS-B', bounds=bounds)

                # Evalaute new x with true objective function and re-train GP
                self.parameters = np.concatenate((self.parameters, \
                    np.reshape(res.x, (1,self.n_parameters))))
                self.contexts = np.concatenate((self.contexts, \
                    np.reshape(context, (1,self.n_contexts))))
                self.X = np.concatenate((self.parameters,self.contexts),axis=1)

                # Evaluating obj_f
                obj, info = self.obj_f(np.concatenate([res.x, context]))

                self.Y = np.concatenate((self.Y, [obj]))
                self.info = np.concatenate((self.info, [info]))

                self.train_GP(self.X, self.Y)
                self.model.optimize()
            except Exception:
                print('Exception encountered during optimization')
                traceback.print_exc()

    def acq_f(self, x, epsilon=.2, use_mean=False):
        x = np.reshape(x, (1, self.n_parameters + self.n_contexts))
        mean, var = self.model.predict(x)
        if use_mean:
            return mean
        Z = -mean / np.sqrt(var)
        return -np.sqrt(var) * (Z * scipy.stats.norm.cdf(Z) + \
            scipy.stats.norm.pdf(Z))

    def predict_optimal(self, context):
        '''
        Returns the predicted optimal parameters for a given context

        context:    A scalar value representing a specified context.
                    This value doesn't have to be in the given context_space
        '''
        def DIRECT_obj(x, user_data):
            return self.acq_f(np.concatenate((x, context)), use_mean=True), 0
        x, fmin, ierror = solve(DIRECT_obj, self.lower_bounds, \
            self.upper_bounds, maxf=1000)

        def LBFGS_obj(x):
            return self.acq_f(np.concatenate((x, context)), use_mean=True)
        bounds = [(self.lower_bounds[i], self.upper_bounds[i]) \
            for i in range(self.n_parameters)]
        res = minimize(LBFGS_obj, x, method='L-BFGS-B', bounds=bounds)
        return obj_f(np.concatenate([res.x, context]))[0]

init_vrep()
load_scene('scenes/normal.ttt')
to = TurningOptimizer(turning_obj_f, 4, [0, 0, 0, 0], [10,10,10,10], [(4,0), \
    (4 * np.cos(np.pi / 8), 4 * np.sin(np.pi / 8)),(4 * np.cos(-np.pi / 8),\
    4 * np.sin(-np.pi / 8)),(4 * np.cos(np.pi / 4), 4 * np.sin(np.pi / 4)),\
    (4 * np.cos(-np.pi / 4), 4 * np.sin(-np.pi / 4))])
to.optimize(250)
to.plot()
exit_vrep()

from __future__ import division, print_function, absolute_import
from builtins import range

import opto
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression

from objective_functions import *

import numpy as np
import matplotlib.pyplot as plt
from dotmap import DotMap

init_vrep()
load_scene('scenes/normal.ttt')
obj_f = generate_f(parameter_mode='normal', objective_mode='single', steps=400)
task = OptTask(f=obj_f, n_parameters=4, n_objectives=1, \
    bounds=bounds(min=[1, -np.pi, 0, 0], max=[60, np.pi, 1, 1]), \
    vectorized=False)
stopCriteria = StopCriteria(maxEvals=50)

p = DotMap()
p.verbosity = 1
p.acq_func = EI(model=None, logs=None)
p.optimizer = opto.CMAES
p.model = regression.GP
opt = opto.BO(parameters=p, task=task, stopCriteria=stopCriteria)
opt.optimize()
logs = opt.get_logs()
print("Parameters: " + str(logs.get_parameters()))
print("Objectives: " + str(logs.get_objectives()))
exit_vrep()

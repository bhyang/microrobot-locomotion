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
obj_f = generate_f(parameter_mode='discovery', objective_mode='moo', steps=400)
task = OptTask(f=obj_f, n_parameters=8, n_objectives=2, name='MOO', \
    bounds=bounds(min=[1e-10,0,0,0,0,0,0,0], max=[60, 2 * np.pi, \
    2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi]), \
    vectorized=False)
stopCriteria = StopCriteria(maxEvals=1250)

p = DotMap()
p.verbosity = 1
p.acq_func = EI(model=None, logs=None)
p.optimizer = opto.CMAES
p.model = regression.GP
opt = opto.PAREGO(parameters=p, task=task, stopCriteria=stopCriteria)
opt.optimize()
logs = opt.get_logs()
print("Parameters: " + str(logs.get_parameters()))
print("Objectives: " + str(logs.get_objectives()))
exit_vrep()

import numpy as np
import matplotlib.pyplot as plt

import vrep_api.vrep as vrep
import traceback
from walker import Walker
from cpg import CpgController
from cpg_gaits import *

VALID_ERROR_CODES = (0 , 1)
ENV_VAR = {}

'''
Helpers
'''
def init_vrep():
    vrep.simxFinish(-1)
    client_id = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    assert client_id != -1, 'Failed to connect to remote API server'
    vrep.simxSynchronous(client_id, 1)
    ENV_VAR['client_id'] = client_id
    print('Initialized simulation environment')

def exit_vrep():
    vrep.simxFinish(ENV_VAR['client_id'])
    print('Exited simulation environment')

def load_scene(path):
    assert vrep.simxLoadScene(ENV_VAR['client_id'], path, 1, vrep.simx_opmode_blocking) in \
        VALID_ERROR_CODES, 'Error loading scene'
    print('Loading scene: ' + path)
    walker = Walker()
    ENV_VAR['walker'] = walker

def wait(steps):
    CLIENTID = ENV_VAR['client_id']
    for _ in range(steps): vrep.simxSynchronousTrigger(CLIENTID)

def distance(a, b):
    return np.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2))

def quatToDirection(quat):
    w,x,y,z = quat
    v0 = 2 * (x*z + w*y)
    v1 = 2 * (y*z - w*x)
    v2 = 1 - (2 * (x*x + y*y))
    return np.arctan2(v0, v1)


'''
Single/multiple objectives
'''
def generate_f(parameter_mode, objective_mode, gait=DualTripod, steps=400, \
    penalize_offset=True, client_id=0):
    '''
    parameter modes:
        normal (f, phase_offset, left_bias, right_bias)
        contextual (f, phase_offset, bias)
        discovery (f, phase_offset, 6 coupling terms)
    objective modes:
        single (distance)
        moo (distance, energy)
        target (distance to target)
    '''
    CLIENTID = ENV_VAR['client_id']
    walker = ENV_VAR['walker']

    def objective(x):
        try:
            walker.reset()
            vrep.simxStartSimulation(CLIENTID, vrep.simx_opmode_blocking)
            errorCode, start = vrep.simxGetObjectPosition(CLIENTID, \
                walker.base_handle, -1, vrep.simx_opmode_blocking)
            assert errorCode in VALID_ERROR_CODES, 'Walker not found'

            x = np.asarray(x)[0]
            print('\nParameters: ' + str(x))
            cpg_gait = gait
            if parameter_mode is 'normal':
                cpg_gait.f = x[0]
                cpg_gait.phase_offset = x[1]
                walker.set_left_bias(x[2])
                walker.set_right_bias(x[3])
            elif parameter_mode is 'contextual':
                cpg_gait.f = x[0]
                cpg_gait.phase_offset = x[1]
                walker.set_left_bias(x[2])
                walker.set_right_bias(x[2])
            elif parameter_mode is 'discovery':
                cpg_gait.f = x[0]
                cpg_gait.phase_offset = x[1]
                cpg_gait.phase_groupings = {
                    0: x[2],
                    1: x[3],
                    2: x[4],
                    3: x[5],
                    4: x[6],
                    5: x[7]
                }
            else:
                raise Exception('Invalid parameter mode')

            cpg_gait.coupling_phase_biases = \
                cpg_gait.generate_coupling(cpg_gait.phase_groupings)

            cpg = CpgController(cpg_gait)
            for _ in range(1000):
                cpg.update(plot=False)

            print('Running trial...')
            for _ in range(steps):
                output = cpg.output()
                if objective_mode is not 'normal':
                    walker.update_energy()
                for i in range(6):
                    walker.legs[i].extendZ(output[i][0])
                    walker.legs[i].extendX(output[i][1])
                wait(1)
                cpg.update()

            errorCode, end = vrep.simxGetObjectPosition(CLIENTID, \
                walker.base_handle, -1, vrep.simx_opmode_blocking)


            if objective_mode is not 'single':
                total_energy = walker.calculate_energy() / 5

            vrep.simxStopSimulation(CLIENTID, vrep.simx_opmode_blocking)
            vrep.simxGetPingTime(CLIENTID)
            vrep.simxClearIntegerSignal(CLIENTID, '', vrep.simx_opmode_blocking)
            vrep.simxClearStringSignal(CLIENTID, '', vrep.simx_opmode_blocking)
            vrep.simxClearFloatSignal(CLIENTID, '', vrep.simx_opmode_blocking)

            if objective_mode is 'target':
                target = x[-2], x[-1]
                return np.array([distance(end, target)])
            else:
                distance = start[0] - end[0]
                if penalize_offset: distance += (.1 * np.abs(end[1] - start[1]))
                if objective_mode is 'single':
                    print('Distance traveled: ' + str(distance))
                    return np.array([distance])
                else:
                    print('Distance traveled: ' + str(distance))
                    print('Total power: ' + str(total_energy) + 'W')
                    return np.array([distance, total_energy])
        except Exception as e:
            print('Encountered an exception, disconnecting from remote API server')
            vrep.simxStopSimulation(CLIENTID, vrep.simx_opmode_oneshot)
            vrep.simxGetPingTime(CLIENTID)
            vrep.simxFinish(CLIENTID)
            traceback.print_exc()
            exit()
    return objective

'''
Contextual objectives
'''
def incline_obj_f(x):
    print('\nParameters: ' + str(x))
    try:
        CLIENTID = ENV_VAR['client_id']
        load_scene('scenes/inclines/' + str(int(x[3])) + 'deg.ttt')
        walker = ENV_VAR['walker']

        vrep.simxSynchronous(CLIENTID, 1)
        vrep.simxStartSimulation(CLIENTID, vrep.simx_opmode_blocking)

        errorCode, start = vrep.simxGetObjectPosition(CLIENTID, \
            walker.base_handle, -1, vrep.simx_opmode_blocking)
        assert errorCode in VALID_ERROR_CODES, 'Error getting object position'

        gait = DualTripod
        gait.f = x[0]
        gait.phase_offset = x[1]
        gait.coupling_phase_biases = gait.coupling_phase_biases = \
            gait.generate_coupling(gait.phase_groupings)

        cpg = CpgController(gait)
        for _ in range(1000):
            cpg.update(plot=False)

        print('Running trial...')
        for _ in range(100):
            output = cpg.output()
            for i in range(6):
                walker.legs[i].extendZ(output[i][0] * x[2])
                walker.legs[i].extendX(output[i][1] * x[2])
            vrep.simxSynchronousTrigger(CLIENTID)
            cpg.update()

        errorCode, end = vrep.simxGetObjectPosition(CLIENTID, \
            walker.base_handle, -1, vrep.simx_opmode_blocking)

        vrep.simxStopSimulation(CLIENTID, vrep.simx_opmode_blocking)
        vrep.simxGetPingTime(CLIENTID)
        vrep.simxClearIntegerSignal(CLIENTID, '', vrep.simx_opmode_blocking)
        vrep.simxClearStringSignal(CLIENTID, '', vrep.simx_opmode_blocking)
        vrep.simxClearFloatSignal(CLIENTID, '', vrep.simx_opmode_blocking)
        # to be maximized
        print('Objective: ' + str(start[0] - end[0]))
        return np.array([start[0] - end[0]])
    except Exception:
        print('Encountered an exception, disconnecting from remote API server')
        vrep.simxStopSimulation(CLIENTID, vrep.simx_opmode_oneshot)
        vrep.simxGetPingTime(CLIENTID)
        vrep.simxFinish(CLIENTID)
        traceback.print_exc()
        exit()

def turning_obj_f(x):
    print('\nParameters: ' + str(x))
    try:
        CLIENTID = ENV_VAR['client_id']
        walker = ENV_VAR['walker']
        errorCode,baseCollectionHandle=\
            vrep.simxGetCollectionHandle(CLIENTID, 'Base', \
            vrep.simx_opmode_blocking)
        vrep.simxSynchronous(CLIENTID, 1)
        vrep.simxStartSimulation(CLIENTID, vrep.simx_opmode_blocking)

        walker.reset()

        errorCode, start = vrep.simxGetObjectPosition(CLIENTID, \
            walker.base_handle, -1, vrep.simx_opmode_blocking)
        assert errorCode in VALID_ERROR_CODES, 'Error getting object position'

        q1 = vrep.simxGetObjectGroupData(CLIENTID, baseCollectionHandle, \
            7, vrep.simx_opmode_blocking)
        o1 = quatToDirection(q1[3]) + np.pi

        x[2] = 15 + (3 * x[2])
        x[3] = x[3] * 2 * np.pi / 10

        gait = DualTripod
        gait.f = x[2]
        gait.phase_offset = x[3]
        gait.coupling_phase_biases=gait.generate_coupling(gait.phase_groupings)

        walker.set_left_bias(x[0] / 10)
        walker.set_right_bias(x[1] / 10)

        cpg = CpgController(gait)
        for _ in range(1000):
            cpg.update(plot=False)

        print('Running trial...')
        for _ in range(100):
            output = cpg.output()
            walker.update_energy()
            for i in range(6):
                walker.legs[i].extendZ(output[i][0])
                walker.legs[i].extendX(output[i][1])
            vrep.simxSynchronousTrigger(CLIENTID)
            cpg.update()

        total_energy = walker.calculate_energy() / 5

        errorCode, end = vrep.simxGetObjectPosition(CLIENTID, \
            walker.base_handle, -1, vrep.simx_opmode_blocking)
        q2 = vrep.simxGetObjectGroupData(CLIENTID, baseCollectionHandle, \
            7, vrep.simx_opmode_blocking)
        o2 = quatToDirection(q2[3]) + np.pi

        vrep.simxStopSimulation(CLIENTID, vrep.simx_opmode_blocking)
        vrep.simxGetPingTime(CLIENTID)
        vrep.simxClearIntegerSignal(CLIENTID, '', vrep.simx_opmode_blocking)
        vrep.simxClearStringSignal(CLIENTID, '', vrep.simx_opmode_blocking)
        vrep.simxClearFloatSignal(CLIENTID, '', vrep.simx_opmode_blocking)
        displacement = np.array(end) - np.array(start)
        orientation = o2 - o1

        info = np.array([total_energy, displacement[0], displacement[1], \
            orientation])

        objective = distance(displacement, (x[4], x[5]))

        print('Orientation change: ' + str(orientation))
        print('Total power: ' + str(total_energy) + 'W')
        print('Target offset: ' + str(objective))
        return np.array([objective]), info

    except Exception:
        print('Encountered an exception, disconnecting from remote API server')
        vrep.simxStopSimulation(CLIENTID, vrep.simx_opmode_oneshot)
        vrep.simxGetPingTime(CLIENTID)
        vrep.simxFinish(CLIENTID)
        traceback.print_exc()
        exit()

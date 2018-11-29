from __future__ import division
import numpy as np

pi = np.pi

class Gait:
    def __init__(self, phase_groupings, f, R_l, R_f, X_l, X_f, phase_offset):
        '''
        Contains default parameters for the various CPG gaits to be fed into CpgController
        Note that 'leg' corresponds to the vertical motor and 'foot' corresponds to horizontal motor

        phase_groupings:    Dictionary scheme used to indicate groupings
        f:                  Frequency (can also be overridden when initializing CpgController)
        R_l:                Sequence of leg extension lengths
        R_f:                Sequence of foot extension lengths
        X_l:                Sequence of leg offset values
        X_f:                Sequence of foot offset values
        phase_offset:       Phase difference between leg and foot
        '''
        self.phase_groupings = phase_groupings
        self.phase_offset = phase_offset
        self.f = f
        self.R_l = R_l
        self.R_f = R_f
        self.X_l = X_l
        self.X_f = X_f

        self.coupling_phase_biases = self.generate_coupling(phase_groupings)

    def generate_coupling(self, phase_groupings):
        '''
        Generates the coupling bias matrix from a phase_groupings array

        phase_groupings:    Scheme used to generate matrix
        '''
        coupling_phase_biases = np.zeros((12, 12))
        for i in range(6):
            coupling_phase_biases[i][i + 6] = -self.phase_offset
            coupling_phase_biases[i + 6][i] = self.phase_offset

        for i in range(6):
            for j in range(6):
                coupling_phase_biases[i][j] = phase_groupings[j] - phase_groupings[i]
        return coupling_phase_biases

'''
GAITS
'''
DualTripod = Gait(
    phase_groupings={
        0: 0,
        1: pi,
        2: 0,
        3: pi,
        4: 0,
        5: pi
    },
    f=30,
    R_l=[.04 for _ in range(6)],
    R_f=[.04 for _ in range(6)],
    X_l=[.04 for _ in range(6)],
    X_f=[.04 for _ in range(6)],
    phase_offset=pi / 2
)

Ripple = Gait(
    phase_groupings={
        0: 0,
        1: 3 * pi / 2,
        2: pi,
        5: pi,
        4: pi / 2,
        3: 0
    },
    f=25,
    R_l=[.04 for _ in range(6)],
    R_f=[.04 for _ in range(6)],
    X_l=[.04 for _ in range(6)],
    X_f=[.04 for _ in range(6)],
    phase_offset=pi / 2
)

Wave = Gait(
    phase_groupings={
        0: 2 * pi / 3,
        1: pi / 3,
        2: 0,
        3: 3 * pi / 3,
        4: 4 * pi / 3,
        5: 5 * pi / 3
    },
    f=30,
    R_l=[.04 for _ in range(6)],
    R_f=[.04 for _ in range(6)],
    X_l=[.04 for _ in range(6)],
    X_f=[.04 for _ in range(6)],
    phase_offset=pi / 2
)

FourTwo = Gait(
    phase_groupings={
        0: 0,
        1: pi,
        2: 4 * pi / 3,
        3: 4 * pi / 3,
        4: pi,
        5: 0
    },
    f=35,
    R_l=[.04 for _ in range(6)],
    R_f=[.04 for _ in range(6)],
    X_l=[.04 for _ in range(6)],
    X_f=[.04 for _ in range(6)],
    phase_offset=1.902525
)

## JUNKYARD / NURSERY

TurnLeft = Gait(
    phase_groupings={
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    },
    f=20,
    R_l=[0 for _ in range(3)] + [.04 for _ in range(3)],
    R_f=[0 for _ in range(3)] + [.04 for _ in range(3)],
    X_l=[0, .04, 0] + [.04 for _ in range(3)],
    X_f=[0, 0, 0] + [.04 for _ in range(3)],
    phase_offset=pi / 2
)

TurnRight = Gait(
    phase_groupings={
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    },
    f=20,
    R_l=[.04 for _ in range(3)] + [0 for _ in range(3)],
    R_f=[.04 for _ in range(3)] + [0 for _ in range(3)],
    X_l=[.04 for _ in range(3)] + [0, .04, 0],
    X_f=[.04 for _ in range(3)] + [0, 0, 0],
    phase_offset=pi / 2
)

Discovered1 = Gait(
    phase_groupings = {
        0: 3.8713592,
        1: 1.674,
        2: 0,
        3: 0,
        4: 3.8713592,
        5: 0
    },
    f=33,
    R_l=[.04 for _ in range(6)],
    R_f=[.04 for _ in range(6)],
    X_l=[.04 for _ in range(6)],
    X_f=[.04 for _ in range(6)],
    phase_offset=2.5735733
)

Dummy = Gait(
    phase_groupings={
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    },
    f=15,
    R_l=[.04 for _ in range(6)],
    R_f=[.04 for _ in range(6)],
    X_l=[.04 for _ in range(6)],
    X_f=[.04 for _ in range(6)],
    phase_offset=pi / 2
)

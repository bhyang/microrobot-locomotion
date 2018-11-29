import numpy as np
import matplotlib.pyplot as plt
from cpg_gaits import *

pi = np.pi

STEP_SIZE = .05
R_GAINS_CONSTANT = 20
X_GAINS_CONSTANT = 20
COUPLING_WEIGHT = 20

class Oscillator:
    def __init__(self, f, R, X):
        self.f = f
        self.R = R
        self.X = X

        self.phase = 0
        self.r = self.R
        self.d_r = 0

        self.x = self.X
        self.d_x = 0

    def output(self):
        return self.x + (self.r * np.cos(self.phase))

    def d_output(self):
        return -self.r * np.sin(self.phase)

    def max(self):
        return max(self.r + self.x, self.x - self.r)

    def min(self):
        return min(self.r + self.x, self.x - self.r)

class OscillatorNetwork:
    def __init__(self, oscillators, coupling_weights, phase_biases):
        '''
        Implementation of the Kuramoto coupling network, composed of oscillators

        oscillators:        Sequence of oscillators that compose the network
        coupling_weights:   Matrix specifying the coupling between oscillators
        phase_biases:       Matrix specifying the desired phase biases
        '''
        self.oscillators = oscillators
        self.coupling_weights = coupling_weights
        self.phase_biases = phase_biases

        self.N = len(oscillators)

    def updateAll(self):
        # Calculating phase changes using Euler's method
        phase_changes = []
        for i in range(self.N):
            current = self.oscillators[i]
            phase_change = current.f
            for j in range(self.N):
                other = self.oscillators[j]
                phase_change += self.coupling_weights[i][j] * other.r * \
                    np.sin(other.phase - current.phase - self.phase_biases[i][j])
            phase_changes.append(phase_change * STEP_SIZE)

        # Updating phase changes
        for i in range(self.N):
            self.oscillators[i].phase += phase_changes[i]

        # Updating amplitudes and offsets
        for osc in self.oscillators:
            osc.d_r += STEP_SIZE * R_GAINS_CONSTANT * ((R_GAINS_CONSTANT * \
                (osc.R - osc.r) / 4) - osc.d_r)
            osc.r += STEP_SIZE * osc.d_r

            osc.d_x += STEP_SIZE * X_GAINS_CONSTANT * ((X_GAINS_CONSTANT * \
                (osc.X - osc.x) / 4) - osc.d_x)
            osc.x += STEP_SIZE * osc.d_x

    def modulate(self, new_R=None, new_X=None):
        for i in range(12):
            if new_R:
                self.oscillators[i].R = new_R[i]
            if new_X:
                self.oscillators[i].X = new_X[i]


class CpgController:
    def __init__(self, gait, f=None):
        '''
        CPG-based controller for the six-legged walker robot
        Uses OscillatorNetwork to construct a CPG network, with some added
        convenience functions

        gait:       User-specified Gait object (see cpg_gaits for more)
        frequency:  Allows the user to override the default gait frequency
        '''
        self.gait = gait
        if not f:
            f = gait.f

        # Initializing oscillators with gait parameters
        legs = [Oscillator(f=f, R=gait.R_l[_], X=gait.X_l[_]) for _ in range(6)]
        feet = [Oscillator(f=f, R=gait.R_f[_], X=gait.X_f[_]) for _ in range(6)]
        self.oscillators = legs + feet
        self.oscillators[0].phase = .01

        # Creating coupling matrix
        self.coupling_weights = np.zeros((12, 12))
        for i in range(6):
            self.coupling_weights[i][i + 6] = COUPLING_WEIGHT
            self.coupling_weights[i + 6][i] = COUPLING_WEIGHT
            for j in range(6):
                self.coupling_weights[i][j] = COUPLING_WEIGHT

        self.phase_biases = gait.coupling_phase_biases
        self.network = OscillatorNetwork(self.oscillators, \
            self.coupling_weights, self.phase_biases)

        # Plotting variables
        self.time = 0
        self.x_data = []
        self.y_data = [[] for _ in range(12)]

    def update(self, plot=True):
        self.network.updateAll()
        if plot:
            self.x_data.append(self.time)
            self.time += STEP_SIZE

            output = self.output()
            for i in range(6):
                self.y_data[i].append(output[i][0])
                self.y_data[i + 6].append(output[i][1])

    def encode(self, vert_osc, horiz_osc):
        '''
        Turns CPG outputs into useable values to be fed into the motors
        Also builds in the physical constraints with the micro
        sized spring actuators using piece-wise functions

        vert_osc:   Oscillator corresponding to the vertical motor
        horiz_osc:  Oscillator corresponding to the horizontal motor
        '''
        vert_increasing = np.sign(vert_osc.d_output()) == 1
        horiz_increasing = np.sign(horiz_osc.d_output()) == 1

        if vert_increasing and horiz_increasing:
            return vert_osc.output(), horiz_osc.output(), 0
        elif horiz_increasing and not vert_increasing:
            return vert_osc.max(), horiz_osc.output(), 0
        elif not vert_increasing and not horiz_increasing:
            return vert_osc.max(), horiz_osc.max(), 0
        else:
            return vert_osc.output(), horiz_osc.min(), 1

    def output(self):
        return [self.encode(self.oscillators[i], self.oscillators[i + 6]) \
            for i in range(6)]

    def plot(self):
        f = plt.figure(figsize=(20/2, 9/2))
        for y in self.y_data:
            plt.plot(self.x_data, y, linewidth=2.5)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Actuation distance [cm]', fontsize=12)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.xticks((0,.2,.4,.6, .8))
        plt.yticks((0, .04, .08))
        plt.show()

    def modulate(self, new_R=None, new_X=None):
        self.network.modulate(new_R, new_X)

def main():
    cpg = CpgController(gait=Dummy)
    for _ in range(1021):
        cpg.update(plot=False)
    for _ in range(84):
        cpg.update()
    cpg.plot()

# main()

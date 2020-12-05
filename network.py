import numpy as np
from scipy.stats import vonmises
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, time_to_phase, make_delay_positive
from stochastic_neurons.stochastic_neuron import stochastic_neuron
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Network:
    def __init__(self, N, cycleTime, dt,W, scope='hopfield'):
        self._magnitude = np.abs(W)
        self._neurons = []
        self.last_spikes = []
        self.cycleTime = cycleTime
        w_angles = np.angle(W)
        w_abs = np.abs(np.reshape(W, (N ** 2)))
        positive_delays = make_delay_positive(w_angles)
        delays = phase_to_time(positive_delays, cycleTime)
        self._delays = delays
        for _ in range(N):
            self._neurons.append(stochastic_neuron(N, cycleTime=cycleTime, dt = dt))



    @property
    def magnitude(self):
        """
        Get the weight 2-D Tensor for the network.
        Rows correspond to inputs and columns correspond
        to outputs.
        """
        return self._magnitude
    @property
    def delay(self):
        return self._delays

    """step should go for one cycle. The first time you call this input a pattern, 
    and then use the state of the network as the next input pattern.(stored in last_spikes) """
    def step(self, input_pattern):
        next_spikes = []
        for i in range(len(input_pattern)):
            next_spikes.append(self._neurons[i].forward(self.last_spikes, self._magnitude[i], self._delays[i]))

        self.last_spikes = next_spikes


    def run_simulation(self, input_pattern, num_cycles):
        #feed in pattern
        self.last_spikes = phase_to_time(input_pattern,self.cycleTime)
        #recur
        for i in range(num_cycles):
            print(i)
            self.step(self.last_spikes)

        difference = np.abs(input_pattern - time_to_phase(self.last_spikes))
        print(np.cos(difference).sum()/len(input_pattern))



































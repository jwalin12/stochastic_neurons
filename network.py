import numpy as np
from scipy.stats import vonmises
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, time_to_phase, form_matrix, gen_random_spikes
from stochastic_neurons.stochastic_neuron import stochastic_neuron
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Network:
    def __init__(self, N, cycleTime, dt, scope='hopfield'):
        self._magnitude = tf.get_variable('weights',
                                        shape=(N, N),
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
        self._delays = tf.get_variable('weights',
                                        shape=(N, N),
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

        self._neurons = []
        self.last_spikes = []
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
        for i in range(len(input_pattern)):































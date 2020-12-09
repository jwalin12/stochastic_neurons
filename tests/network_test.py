import numpy as np
from scipy.stats import vonmises
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, phase_noise, find_weights,storkey_learning_weights, find_weights_TPAM_learning
from stochastic_neurons.stochastic_neuron import stochastic_neuron

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from stochastic_neurons.network import Network
from pprint import pprint


num_patterns = 1
N = 128
cycleTime = 10
dt = 0.0001
num_cycles = 8

# No sparsity enforced --> Are we going to ignore that aspect of this model? - just texting basic network function right now
phases = np.zeros((num_patterns, N)) # Generates num_patterns random vectors (M x N)
phases[0][1] = 1
phases[0][20] = -1
patterns = np.exp(1j * phases * 2 * np.pi) # Converts the vectors into phasors
# print(f'patterns: {patterns}')
corrupted_phase = phase_noise(phases, kappa = 3) # Adds some corruption --> phasor notation
# print(f'corrupted: {corrupted}')
W  = find_weights(patterns) # Storing patterns
network = Network(N,cycleTime,dt, W) # Initialize network based on weight matrix
network.run_simulation(corrupted_phase[0],num_cycles, np.angle(patterns[0]))


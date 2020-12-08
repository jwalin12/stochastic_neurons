import numpy as np
from scipy.stats import vonmises
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, phase_noise, find_weights,storkey_learning_weights, find_weights_TPAM_learning
from stochastic_neurons.stochastic_neuron import stochastic_neuron

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from stochastic_neurons.network import Network
from pprint import pprint


num_patterns = 8
N = 128
cycleTime = 10
dt = 0.0001
num_cycles = 8

# No sparsity enforced --> Are we going to ignore that aspect of this model?
phases = np.random.uniform(size=(num_patterns, N)) # Generates num_patterns random vectors (M x N)
# print(f'phases: {phases}')
patterns = np.exp(1j * phases * 2 * np.pi) # Converts the vectors into phasors
# print(f'patterns: {patterns}')
corrupted = phase_noise(patterns, kappa = 3) # Adds some corruption --> phasor notation
# print(f'corrupted: {corrupted}')
corrupted_phase = np.angle(corrupted) # Converts from complex number to real valued representation from -pi to pi
# print(f'corrupted_phase: {corrupted_phase}')


W = storkey_learning_weights(phases.T) # Storing patterns
network = Network(N,cycleTime,dt, W) # Initialize network based on weight matrix
network.run_simulation(corrupted_phase[0],num_cycles, np.angle(patterns[0]))


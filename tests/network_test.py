import numpy as np
from scipy.stats import vonmises
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, phase_noise, find_weights
from stochastic_neurons.stochastic_neuron import stochastic_neuron
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from stochastic_neurons.network import Network


num_patterns = 4
N = 128
cycleTime = 10
dt = 0.0001
num_cycles = 5

phases = np.random.uniform(size=(num_patterns, N))
patterns = np.exp(1j * phases * 2 * np.pi)
corrupted = phase_noise(patterns, kappa = 3)
corrupted_phase = np.angle(corrupted)


W = find_weights(patterns,4, 2**9)
network = Network(N,cycleTime,dt, W)
network.run_simulation(corrupted_phase[0],num_cycles)







import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, phase_noise, find_weights
from stochastic_neurons.stochastic_neuron import stochastic_neuron
from stochastic_neurons.data_indexing import random_tpam_indexing

import tensorflow as tf
from stochastic_neurons.network import Network
from pprint import pprint




num_patterns = 4
N = 128
cycleTime = 10
dt = 0.0001
num_cycles = 5

# Load and unpack mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Import MSNIST from tf
x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize values

# Gather 10 unique images to store in network
unique_digits = []
for i in range(10):
    temp = x_train[np.where(y_train == i)]
    temp = temp.reshape((len(temp), 28*28))[:5000]
    unique_digits.append(temp[0])
    plt.imshow(temp.reshape((len(temp),28,28))[0])
    plt.show()

# Encoding matrix W^I
encoding_matrix = random_tpam_indexing(unique_digits)
patterns = np.exp(1j * unique_digits * 2 * np.pi) # Converts the vectors into phasors
# print(f'patterns: {patterns}')
corrupted = phase_noise(patterns, kappa = 3) # Adds some corruption --> phasor notation
# print(f'corrupted: {corrupted}')
corrupted_phase = np.angle(corrupted) # Converts from complex number to real valued representation from -pi to pi
# print(f'corrupted_phase: {corrupted_phase}')


W = find_weights(patterns,4, 2**9) # Storing patterns
network = Network(N,cycleTime,dt, W) # Initialize network based on weight matrix
network.run_simulation(corrupted_phase[0],num_cycles) # 


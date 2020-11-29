import numpy as np
from scipy.stats import vonmises
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, time_to_phase, form_matrix, gen_random_spikes
from stochastic_neurons.stochastic_neuron import stochastic_neuron



N = 128
cycleTime = 10
dt = 0.0001
simLen = 10*cycleTime
num_patterns = 16
patterns = []

for _ in range(num_patterns):
    patterns.append(gen_random_spikes(cycleTime, dt, shape = (N,)))

W = form_matrix(patterns)
















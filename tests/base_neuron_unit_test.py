from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, time_to_phase, gen_random_spikes
from stochastic_neurons.stochastic_neuron import stochastic_neuron
import numpy as np
from matplotlib import pyplot


cycleTime = 10
dt = 0.001
nDim = 5
time = 0
sim_len = 5*cycleTime

weights = np.ones(nDim)
neuron = stochastic_neuron(nDim, cycleTime, dt)
output_spikes = []
input_spikes = []
while time < sim_len:
    spike_times = gen_random_spikes(cycleTime, dt, shape = (nDim,1))
    input_spikes.append(spike_times)
    output_spikes.append(neuron.forward(spike_times,weights))
    time += cycleTime


print(input_spikes)
print(output_spikes)






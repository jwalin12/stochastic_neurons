import numpy as np
from scipy.stats import vonmises
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, time_to_phase

"""stochastci neuron. Takes in spikes at a cycle and outputs spikes based on sampling form a vonmises distribution."""

class stochastic_neuron:

    def __init__(self, nDim, cycleTime, dt):
        self.nDim = nDim
        self.cycleTime = cycleTime
        self.dt = dt


    """neuron takes in input spike times. In order to deal with weights effectively, what happens is that the neuron will
    sample the spikes coming in from a normal distribution (int) (10*magnitude) times and use those samples to create the von-miss that it uses for the output."""
    def forward(self, input_spikes_times, input_spike_magnitudes):
        inp_spike_phases = time_to_phase(input_spikes_times, self.cycleTime)
        input_fit_params = vonmises.fit(list(inp_spike_phases)* 10*np.array(input_spike_magnitudes)[:self.nDim])
        processed_spikes = vonmises.rvs(kappa = np.array(input_fit_params[0]), loc = np.array(input_fit_params[1]), size = (self.nDim,))
        output_params = vonmises.fit(processed_spikes)
        return fit_time_to_dt(vonmises.rvs(kappa = output_params[0], loc = output_params[1]), self.dt, self.cycleTime)
















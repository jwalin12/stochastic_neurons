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
    def forward(self, input_spikes_times, input_spike_magnitudes, input_spike_delays):
        inp_spike_phases = time_to_phase(np.array(input_spikes_times + input_spike_delays), self.cycleTime)
        processed_spikes = []
        for i in range(len(input_spike_magnitudes)):
            processed_spikes.extend([inp_spike_phases[i]] * (int(10*input_spike_magnitudes[i])))
        if(len(processed_spikes) == 0):
            return 0 #if neuron is disconnected the output doesnt matter
        input_fit_params = vonmises.fit(processed_spikes)
        return fit_time_to_dt(vonmises.rvs(kappa = input_fit_params[0], loc = input_fit_params[1]), self.dt, self.cycleTime)
















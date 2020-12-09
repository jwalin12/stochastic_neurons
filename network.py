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


    def run_simulation(self, input_pattern, num_cycles, orig_pattern):
        print("running simulation")
        #feed in pattern
        self.last_spikes = phase_to_time(input_pattern,self.cycleTime)
        #recur
        for i in range(num_cycles):
            print("cycle no: ", i)
            difference = np.abs(orig_pattern - time_to_phase(self.last_spikes, self.cycleTime))
            similarity = np.abs(np.exp(1j*difference).sum())/len(input_pattern)
            print("similarity: ",similarity)
            self.step(self.last_spikes)
        # print(self.last_spikes)
        last_phase = time_to_phase(self.last_spikes, self.cycleTime)
        difference = np.abs(orig_pattern - last_phase)
        similarity =  np.abs(np.exp(1j*difference).sum())/len(input_pattern)
        print("similarity : ", similarity)
        return last_phase, similarity



class HeteroAssociativeNetwork:
    """
    simple hetero associative network for returning outputs
    """
    def __init__(self, N, D, W):
        self._neurons = []
        self._W
    
    def run(v_in):
        """
        v_in: phasor vector output from the
        """
        pass

class HopfieldNetwork:

  def __init__(self, n, num_iterations):
    self.n = n
    self.num_iterations = num_iterations

  def store_patterns(self, patterns):
    V = np.array(patterns).T
    self.T = (1 / (self.n ** 2)) * V.dot(V.T)
    np.fill_diagonal(self.T, 0)
    print("T: ", self.T)

  def recover_pattern(self, V):
    i = 0
    converged = False
    old_energy = np.inf

    while i < self.num_iterations and not converged:
      new_energy = self._compute_energy(V)
      V = np.sign(self.T.dot(V))
      converged = new_energy >= old_energy
      i += 1
      old_energy = new_energy

    return V

  def _compute_energy(self, V):
    return -0.5 * V.T.dot(self.T).dot(V)
        




















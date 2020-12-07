import numpy as np
from scipy.sparse import random
from pprint import pprint

from stochastic_neurons.utils import get_phasor_vector


"""
Storing Stage:
Transform a real-valued RGB image into a complex vector from the encoding matrix.
Map the complex vector into a timed sequence of input spikes, which feeds into the 
spiking TPAM network.

Readout Stage:
Decode using a Hebbian heteroassociative memory. Use the same neurons and synaptic 
mechanisms to implement the complex dot product for the readout matrix.
The readout neurons respond proporionally to the input magnitude (no refacotry period).
"""

################# Indexing stage ####################
def random_tpam_indexing(P, N, K):
    """
    Encodes data in phasor patterns using a sparse phasor matrix.
    P: Real valued data matrix (D x M) where D is the dimensionality of data and M is # of data points
    N: Number of neurons in the TPAM network
    K: The sparsity constraint (% sparsity of the stored patterns)
    @returns the synaptic encoding matrix W^I (N x D)
    """
    D, M = P.shape 
    
    # Generate random sparse phasor matrix of size N x M
    S = []
    for _ in range(M):
        vec = get_phasor_vector(N, K)
        S.append(vec)
    S = np.array(S).T
    print(f'M: {M}, D: {D}, N: {N}')
    # print(f'S: {S.shape}, P: {P.T.shape}')
    # print('S: ', S)
    # Multiply Sparse phasor matrix by P.T
    W_tilde_i = S@P.T

    return W_tilde_i



P = np.random.randint(0,255, (3,4))
print(P)
W = random_tpam_indexing(P, 5, 0.4)
print(W)
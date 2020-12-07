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
def random_tpam_s_matrix(M, N, K):
    """
    Returns a matrix (N x M) of random sparse phasor vectors.
    M: # of data points
    N: # of neurons in the TPAM network
    K: The sparsity constraint (% sparsity of the stored patterns)
    @returns matrix of random sparse phasors
    """
    
    # Generate random sparse phasor matrix of size N x M
    S = []
    for _ in range(M):
        vec = get_phasor_vector(N, K)
        S.append(vec)
    S = np.array(S).T

    return S

def random_tpam_w_encode(S, P):
    """
    Encodes data in phasor patterns using a sparse phasor matrix.
    S: Matrix of sparse phasor vectors
    P: Real valued data matrix (D x M) where D is the dimensionality of data and M is # of data points
    @returns the synaptic encoding matrix W^I (N x D)
    """

    # Multiply Sparse phasor matrix by P.T
    return S@P.T

def random_tpam_w_decode(S, P, K):
    """
    Decodes data in phasor patterns using a sparse phasor matrix.
    S: Matrix of sparse phasor vectors
    P: Real valued data matrix (D x M) where D is the dimensionality of data and M is # of data points
    @returns the synaptic encoding matrix W^I (D x N)
    """

    # Multiply the pattern matrix by the complex conjugate transpose of the sparse phasor matrix
    S = np.matrix(S)
    return 1/K * P@S.H

# M, N, K = 
# P = np.random.randint(0,255, (3,4))
# print(P)
# W = random_tpam_s_matrix(, 5, 0.4)
# print(W)
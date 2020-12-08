import numpy as np
import math
from scipy.sparse import random
import scipy.linalg as linalg
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

def orthogonal_tpam_s_matrix(M, N, phot):
    """
    Returns a matrix (N x M) of orthogonal sparse phasor vectors.
    M: # of data points
    N: # of neurons in the TPAM network
    K: The sparsity constraint (% sparsity of the stored patterns)
    @returns matrix of random sparse phasors
    """
    # Phase vector with evenly spaced phasors
    phases = np.linspace(-np.pi, np.pi, num=N)
    patterns = np.exp(1j * phases ) # Converts the vectors into phasors

    # Sparse phasor matrix of size N x M
    S = []

    # Find K
    K = math.floor(phot*N)

    # Available indices in N
    A = np.array(range(N))

    for i in range(M):

        # Take a random sample of size K from available positions
        indices = np.random.choice(A, K)

        # Remove indices just used
        A = np.setdiff1d(A, indices)

        # Phasor value at index according to vec
        vec = np.array([patterns[i] if i in indices else 0 for i in range(N)])
        S.append(vec)

    return np.array(S).T

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

def pinv_tpam_w_encode(S, P):
    """
    Encodes data in phasor patterns using a sparse phasor matrix.
    S: Matrix of sparse phasor vectors
    P: Real valued data matrix (D x M) where D is the dimensionality of data and M is # of data points
    @returns the synaptic encoding matrix W^I (N x D)
    """

    # Multiply Sparse phasor matrix by P.T
    return S@linalg.pinv2(P)

def pinv_tpam_w_decode(S, P, K):
    """
    Decodes data in phasor patterns using a sparse phasor matrix.
    S: Matrix of sparse phasor vectors
    P: Real valued data matrix (D x M) where D is the dimensionality of data and M is # of data points
    @returns the synaptic encoding matrix W^I (D x N)
    """

    # Multiply the pattern matrix by the complex conjugate transpose of the sparse phasor matrix
    S = np.matrix(S)
    return 1/K * linalg.pinv2(P).T@S.H

def decode_phase_encoded_vector(W_H, vec, phot, N, D):
    """
    Decode a phase encoded vector according to the decoding matrix and
    sparsity constraints.
    W_H: decoding matrix
    vec: phase encoded vector
    phot: percent sparsity
    N: number of neurons
    D: dimensionality of data
    """
    full_decoded = W_H@vec # Final decoded vector (D x 1)
    print("full_decoded: ", full_decoded)

    # Find K
    K = math.floor(phot*N)
    
    # Take K values with highest magnitude
    k_highest = full_decoded.argsort()[-K:][::-1]
    for i in range(D):
        if i not in k_highest:
            full_decoded[i] = 0
            
    print("k_decoded: ", full_decoded)
    return full_decoded
import numpy as np
import math
from scipy.stats import vonmises
from scipy.sparse import random


def gen_random_spikes(cycleTime, dt, shape = (1,)):
    return fit_time_to_dt(np.random.random(shape)*cycleTime, dt, cycleTime)

def form_matrix(S):
    W = S.dot(S.conj().T) / S.shape[1]
    W = W - np.diag(np.diag(W))
    return W

def make_delay_positive(a):
    return np.where(a >= 0, a, 2. * np.pi + a)

def time_to_phase(time, cycleTime):
    cycleTime = float(cycleTime)
    return ((np.array(time)%cycleTime) / cycleTime)*(np.pi*2)

def phase_to_time(phase, cycleTime):
    return phase%(np.pi*2)*cycleTime

def phase_noise(z, kappa=0.1, tol = 10):
    noise = np.random.vonmises(mu=0, kappa=kappa, size=z.shape)
    z_orig = z
    z_new = np.exp(1j*(np.angle(z) + noise))
    # if complex_difference(z_orig, z_new) < tol:
    #     return phase_noise(z_orig, kappa=0.1, tol = tol)
    return z_new

"arg max learning rule"




def vonmises_similarity(phase, input_phase, kappa=1):
    return np.exp(kappa * (np.cos(phase - input_phase) - 1))


def find_weights(patterns, k, resolution=2 ** 12):
    P, N = patterns.shape
    phis = []
    A = []
    for i in range(N):
        for j in range(N):
            if i == j:
                phis.append(0)
                A.append(0)
            else:
                phi = np.linspace(-np.pi, np.pi, num=resolution)
                if(patterns[:, j:j + 1].any(0)):
                    W_ij = 0
                    phis.append(0)
                else:
                    dphi = np.angle(patterns[:, i:i + 1] / patterns[:, j:j + 1])
                    obj = np.sum(vonmises_similarity(phase=np.angle(patterns[:, i:i + 1]),
                                                     input_phase=np.angle(patterns[:, j:j + 1]) + phi[None, :],
                                                     kappa=k), axis=0)
                    # obj = np.sum(np.cos(np.angle(patterns[:, i:i + 1]) - (np.angle(patterns[:, j:j + 1]) + phi[None, :])),
                    #              axis=0)
                    phi_max = np.argmax(obj)
                    phi_max = phi[phi_max]
                    phis.append(phi_max)
                    W_ij = np.sum(vonmises_similarity(phase=np.angle(patterns[:, i:i + 1]),
                                           input_phase=np.angle(patterns[:, j:j + 1]) + phi_max,
                                           kappa=k))
                A.append(W_ij)
    phis = np.array(phis).reshape((N, N))
    A = np.array(A).reshape((N, N))
    return A * np.exp(1j * phis)

def get_phasor_vector(N, K):
    """
    Return a vector of size N with K % sparsity. The phasors are evenly spaced and sequential over N positions.
    N: Size of vector
    K: sparsity constraint [0,1)
    """
    # Phase vector with evenly spaced phasors
    phases = np.linspace(-np.pi, np.pi, num=N)
    patterns = np.exp(1j * phases ) # Converts the vectors into phasors

    # Binary vector with K% sparsity
    vec = random(1, N, density=K)
    vec.data[:] = 1
    vec = np.array(vec.todense().flat)
    
    # Phasor value at index according to vec
    out = np.array([patterns[i] if vec[i] else 0 for i in range(N)])
    return out

def merge_rgb_vectors(vec1, vec2, percent1):
    return percent1*vec1 + (1 - percent1)*vec2
    
def get_rgb_from_phasor(decoded_vector, num_vals):
    """
    Return a vector of size N with K % sparsity. The phasors are evenly spaced and sequential over N positions.
    N: Size of vector
    K: sparsity constraint [0,1)
    """
    # Get angle from decoded
    decoded_phase = np.angle(decoded_vector).tolist()[0]
    print("decoded_phase: ", decoded_phase)

    # Angle vector with evenly spaced values
    phases = np.linspace(-np.pi, np.pi, num=num_vals)
    # print("phases: ", phases)

    out = []
    # Map from angle to RGB value
    for x in decoded_phase:
        i = 0
        curr = phases[0]
        while curr < x:
            i += 1
            curr = phases[i]
        # print("curr: ", curr)
        # print("x: ", x)
        # print("x<curr? ", x < curr)
        # print("i: ", i)
        # if i == 128:
        #     out.append(0)
        # else:
        # invert output
        out.append(255-i)

    return np.array(out)

def random_phasors(N, M, K):
    phases = np.random.uniform(0, 2 * np.pi, size=(N, M))
    idicies = np.stack([np.random.choice(range(N), replace=False, size=K) for i in range(M)], axis=0)
    M = np.zeros((N, M))

    for i, idx in enumerate(idicies):
        M[idx, i] = 1
    S = np.exp(1j * phases) * M
    S = S / (np.abs(S) + 10 ** -6)
    return S

"""makes time fitted to granularity of steps"""
def fit_time_to_dt(time, dt, cycleTime):

    time += (dt - (time%cycleTime)%dt)
    return time





def time_to_block(time, block_len, cycleTime):
    time = time%cycleTime
    time_increment = cycleTime/block_len
    time_counter = 0
    idx = 0
    for i in range(block_len):
        if(time_counter < time and time < time_counter+time_increment):
            return idx
        idx+=1
        time_counter += time_increment




def phase_to_block(phase, block_len, cycleTime):
    return time_to_block(phase_to_time(phase, cycleTime),block_len,cycleTime)


import numpy as np
from scipy.stats import vonmises


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
                dphi = np.angle(patterns[:, i:i + 1] / patterns[:, j:j + 1])
                obj = np.sum(vonmises_similarity(phase=np.angle(patterns[:, i:i + 1]),
                                                 input_phase=np.angle(patterns[:, j:j + 1]) + phi[None, :],
                                                 kappa=k), axis=0)
                # obj = np.sum(np.cos(np.angle(patterns[:, i:i + 1]) - (np.angle(patterns[:, j:j + 1]) + phi[None, :])),
                #              axis=0)
                phi_max = np.argmax(obj)
                phi_max = phi[phi_max]
                phis.append(phi_max)
                A.append(np.sum(vonmises_similarity(phase=np.angle(patterns[:, i:i + 1]),
                                           input_phase=np.angle(patterns[:, j:j + 1]) + phi_max,
                                           kappa=k)))
    phis = np.array(phis).reshape((N, N))
    A = np.array(A).reshape((N, N))
    return A * np.exp(1j * phis)


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


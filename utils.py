import numpy as np
from scipy.stats import vonmises


def gen_random_spikes(cycleTime, dt, shape = (1,)):
    return fit_time_to_dt(np.random.random(shape)*cycleTime, dt, cycleTime)

def form_matrix(S):
    W = S.dot(S.conj().T) / S.shape[1]
    W = W - np.diag(np.diag(W))
    return W

def time_to_phase(time, cycleTime):
    return ((time%cycleTime) / cycleTime)*(np.pi*2)

def phase_to_time(phase, cycleTime):
    return phase%(np.pi*2)*cycleTime


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


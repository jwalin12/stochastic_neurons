import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, phase_noise, find_weights, get_rgb_from_phasor, merge_rgb_vectors, find_weights_TPAM_learning
from stochastic_neurons.stochastic_neuron import stochastic_neuron
from stochastic_neurons.data_indexing import *

import tensorflow as tf
from stochastic_neurons.network import Network
from pprint import pprint

def test_rgb_merge():
    # Load and unpack mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Import MSNIST from tf
    # x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize values

    # Gather 10 unique images to store in network
    unique_digits = []
    for i in range(10):
        temp = x_train[np.where(y_train == i)]
        temp = temp.reshape((len(temp), 28*28))[:5000]
        unique_digits.append(temp[0])
    unique_digits = np.array(unique_digits).T # Data matrix (D x M)
    
    for i in range(0, 10, 1):
        merged = merge_rgb_vectors(unique_digits[:,2], unique_digits[:,3], i/10)
        print(merged)
        plt.imshow(merged.reshape((28,28)))
        plt.show()


def test_random_tpam_network(K = 0.1, N=128, cycleTime=10, dt=0.0001, num_cycles=3, percent1=0.6, arg_max = True, tol = 0.9):
    """
    Test random tpam network on 10 mnist images.
    """
    # Load and unpack mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Import MSNIST from tf
    x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize values

    # Gather 10 unique images to store in network
    unique_digits = []
    for i in range(10):
        temp = x_train[np.where(y_train == i)]
        temp = temp.reshape((len(temp), 28*28))[:5000]
        unique_digits.append(temp[0])
    unique_digits = np.array(unique_digits).T # Data matrix (D x M)
    
    D, M = unique_digits.shape
    print(f'N: {N}, M: {M}, D: {D}')

    # Encode values into sparse phasor representation
    S = orthogonal_tpam_s_matrix(M, N, K) # Random codebook of orthogonal sparse phasors (N x M)
    print(f'S: {S.shape}')
    encoding_matrix = pinv_tpam_w_encode(S, unique_digits) # Generate encoding matrix (N x D)
    print(f'encoding_matrix: {encoding_matrix.shape}')

    # Corrupt original data
    # corrupted = phase_noise(unique_digits, kappa = 3) # Adds some corruption --> phasor notation (D x M)
    # print(f'corrupted: {corrupted.shape}')
    # # corrupted_phase = np.angle(corrupted) # Converts from complex number to real valued representation from -pi to pi
    # # print(f'corrupted_phase: {corrupted_phase.shape}')
    # encoded_corrupted_matrix = encoding_matrix@corrupted # corrupted encoded matrix (N, M)
    # print(f'encoded_corrupted_matrix: {encoded_corrupted_matrix.shape}')
    # phase_encoded_corrupted_vec = np.angle(encoded_corrupted_matrix[:,0]) # corrupted pattern to be decoded

    # Test a merged 2 and 3 (D x 1)

    merged = merge_rgb_vectors(unique_digits[:,2], unique_digits[:,3], percent1)
    plt.imshow(merged.reshape((28,28)))
    plt.show()

    # Encode the merged vector 
    encoded_vector = encoding_matrix@merged # (N x 1)
    phase_encoded_vector = np.angle(encoded_vector) # (N x 1)
    phase_encoded_orig_vector = np.angle(encoding_matrix@unique_digits[:,2])

    # Find weights, create network, and run simulation
    # TODO: Is S the correct thing to input  here ? - Think so, Jwalin
    if(arg_max):
        W = find_weights(S.T, 100, 2**9) # Storing patterns
    else:
        W = find_weights_TPAM_learning(S)
    network = Network(N, cycleTime, dt, W) # Initialize network based on weight matrix
    orig_difference = np.abs(phase_encoded_orig_vector - phase_encoded_vector)
    orig_similarity = np.abs(np.exp(1j*orig_difference).sum())/N
    print(orig_similarity)
    result_vec, similarity = network.run_simulation(phase_encoded_vector, num_cycles, orig_pattern= phase_encoded_orig_vector) # Find result based on corrupted phase, result_vec is a vector of phases

    # result_phase_vec = np.angle(result_vec)
    # print(f'result_phase_vec: {result_phase_vec.shape}')

    # Decode final phasor 
    decoding_matrix = random_tpam_w_decode(S, unique_digits, K) # Generate decoding matrix (D x N)
    print(f'decoding_matrix: {decoding_matrix.shape}')
    decoded_vector = decoding_matrix@result_vec # Final decoded vector (D x 1)
    print(f'decoded_vector: {decoded_vector}')
    # print(f'decoded_vector: {decoded_vector}')
    rgb_decoded_vector = get_rgb_from_phasor(decoded_vector, 256)
    # print(f'rgb_decoded_vector: {rgb_decoded_vector}')
    scaled_merging = percent1*(similarity/orig_similarity) #proportional merging of image
    print(f'scaled merging: " {scaled_merging}')
    output_img = merge_rgb_vectors(unique_digits[:, 2], unique_digits[:, 3], scaled_merging)
    plt.imshow(output_img)
    plt.show()

def test_encode_decode(K = 0.1, N=128, pinv=True, ortho=True):
    # Load and unpack mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Import MSNIST from tf
    x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize values

    # Gather 10 unique images to store in network
    unique_digits = []
    for i in range(10):
        temp = x_train[np.where(y_train == i)]
        temp = temp.reshape((len(temp), 28*28))[:5000]
        unique_digits.append(temp[0])
        # plt.imshow(temp.reshape((len(temp),28,28))[0])
        # plt.show()
    unique_digits = np.array(unique_digits).T # Data matrix (D x M)
    print(f'unique_digits: {unique_digits[:,0].shape}')

    D, M = unique_digits.shape
    print(f'N: {N}, M: {M}, D: {D}')

    # Encode values into sparse phasor representation
    if ortho:
        S = orthogonal_tpam_s_matrix(M, N, K) # Random codebook of sparse phasors (N x M)
    else:
        S = random_tpam_s_matrix(M, N, K) # Random codebook of sparse phasors (N x M)
    print(f'S: {S}')
    if pinv:
        encoding_matrix = pinv_tpam_w_encode(S, unique_digits) # Generate encoding matrix (N x D)
    else:
        encoding_matrix = random_tpam_w_encode(S, unique_digits) # Generate encoding matrix (N x D)
    print(f'encoding_matrix: {encoding_matrix.shape}')

    # Decode vectors
    # if pinv:
    #     decoding_matrix = pinv_tpam_w_decode(S, unique_digits, K) # Generate decoding matrix (D x N)
    # else:
    decoding_matrix = random_tpam_w_decode(S, unique_digits, K) # Generate decoding matrix (D x N)
    print(f'decoding_matrix: {decoding_matrix.shape}')
    for i in range(10):
        encoded_vector = S[:,i]

        print(f'encoded_vector: {encoded_vector}')
        phase_encoded_vector = np.angle(encoded_vector)
        # Offset angle to (0, 2pi]
        # phase_encoded_vector += np.pi
        print(f'phase_encoded_vector: {phase_encoded_vector}')
        decoded_vector = decoding_matrix@phase_encoded_vector # Final decoded vector (D x 1)
        print(f'decoded_vector: {decoded_vector.shape}')
        print(f'decoded_vector: {decoded_vector}')
        rgb_decoded_vector = get_rgb_from_phasor(decoded_vector, 256)
        print(f'rgb_decoded_vector: {rgb_decoded_vector}')
        

        plt.imshow(rgb_decoded_vector.reshape((28,28)))
        plt.show()
    
# test_encode_decode()
# test_encode_decode(ortho=False)
test_random_tpam_network()
# test_rgb_merge()
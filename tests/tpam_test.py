import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt
from stochastic_neurons.utils import fit_time_to_dt, phase_to_time, phase_noise, find_weights, get_rgb_from_phasor, merge_rgb_vectors,storkey_learning_weights, cosine_similarity, make_delay_positive
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


def test_random_tpam_network(K = 0.1, N=128, cycleTime=10, dt=0.0001, num_cycles=3, percent1=0.7, arg_max = True, tol = 0.9):
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

    target = unique_digits[:,2]

    # Encode the merged vector 
    encoded_vector = encoding_matrix@merged # (N x 1) phasor notation
    print(f'encoded_vector:  {encoded_vector.shape}')
    phase_encoded_vector = np.angle(encoded_vector) # (N x 1) phase notation
    print(f'phase_encoded_vector: {phase_encoded_vector.shape}')
    phase_encoded_orig_vector = np.angle(encoding_matrix@target) # phase notation

    # Find weights, create network, and run simulation
    if(arg_max):
        W = find_weights(S.T, 40, 2**9) # Storing patterns
    else:
        W = storkey_learning_weights(S)
    network = Network(N, cycleTime, dt, W) # Initialize network based on weight matrix
    # result_vec = network.run_simulation(phase_encoded_vector, num_cycles) # Find result based on corrupted phase
    # result_phasor_vec = np.angle(result_vec)
    # print(f'result_phasor_vec: {result_vec}')
    orig_difference = np.abs(phase_encoded_orig_vector - phase_encoded_vector)
    orig_similarity = np.abs(np.exp(1j * orig_difference).sum()) / N
    print("phase_encoded_orig to phase encoded similarity", orig_similarity)
    result_vec, similarity = network.run_simulation(phase_encoded_vector, num_cycles, orig_pattern=phase_encoded_orig_vector) # Find result based on corrupted phase, result_vec is a vector of phases

    print("result_vec to phase_encoded_orig similiarity: ", similarity)

    print("similarity to merge: {}".format(similarity*percent1/(orig_similarity)))

    # Convert output phase vector to phasor
    result_phasor_vec = np.exp(1j*result_vec)
    print(f'result_phasor_vec: {result_phasor_vec.shape}')

    # Decode final phasor 
    decoding_matrix = random_tpam_w_decode(S, unique_digits, K) # Generate decoding matrix (D x N)
    print(f'decoding_matrix: {decoding_matrix.shape}')
    decoded_vector = decode_phase_encoded_vector(decoding_matrix, encoded_vector, K, N, D)[0] # Final decoded vector (D x 1)
    # decoded_vector = decoding_matrix@result_vec # Final decoded vector (D x 1)
    print(f'decoded_vector: {decoded_vector.shape}')
    # print(f'decoded_vector: {decoded_vector}')
    rgb_decoded_vector = get_rgb_from_phasor(decoded_vector, 256)
    # print(f'rgb_decoded_vector: {rgb_decoded_vector}')

    print('showing decoded image: ')
    plt.imshow(rgb_decoded_vector.reshape((28,28)))
    plt.show()

    print('showing target image:')
    plt.imshow(target.reshape((28,28)))
    plt.show()

#####################################################################################

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

#####################################################################################

def test_two_three_decode(K = 0.1, N=128, pinv=True, ortho=True, percent1=0.7):
    """
    Test encode decode schema for a merged 2 and 3
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
        # plt.imshow(temp.reshape((len(temp),28,28))[0])
        # plt.show()
    unique_digits = np.array(unique_digits).T # Data matrix (D x M)
    print(f'unique_digits: {unique_digits[:,0].shape}')

    D, M = unique_digits.shape
    print(f'N: {N}, M: {M}, D: {D}')

    # Test a merged 2 and 3 (D x 1)
    merged = merge_rgb_vectors(unique_digits[:,2], unique_digits[:,3], percent1)
    target = unique_digits[:,2]

    # Encode values into sparse phasor representation
    S = orthogonal_tpam_s_matrix(M, N, K) # Random codebook of sparse phasors (N x M)
    print(f'S: {S}')
    encoding_matrix = pinv_tpam_w_encode(S, unique_digits) # Generate encoding matrix (N x D)
    print(f'encoding_matrix: {encoding_matrix.shape}')
    target_index = S[:,2]

    # Decoding network
    # network = HopfieldNetwork(N, 2)
    # network.store_patterns(S)

    # ref_encoded_vec = encoding_matrix@unique_digits[:,5]
    # ref_encoded_vec_2 = encoding_matrix@unique_digits[:,6]
    # for i in range(10):
    #     encoded_vec = S[:,i] #encoding_matrix@unique_digits[:,i]
    #     max_cos = -np.inf
    #     index = 0
    #     for j in range(10):
    #         curr = cosine_similarity(np.angle(encoded_vec), np.angle(S[:,j]))
    #         print(f'cos similarity encoded {i} target {j}: {curr}')
    #         if curr > max_cos:
    #             max_cos = curr
    #             index = j
    #     print(f'    max similarity: {max_cos}, {index}')

    # encode and decode merged and target vector
    for vec in [merged, target]:
        plt.imshow(vec.reshape((28,28)))
        plt.show()
        encoded_vector = encoding_matrix@vec
        # print(f'encoded_vector: {encoded_vector}')
        # print(f'target_index: {target_index}')
        # print(f'cos_similarity: {cosine_similarity(np.angle(encoded_vector), np.angle(target_index))}')
        # print(f'cos_similarity reference: {cosine_similarity(np.angle(encoded_vector), np.angle(ref_encoded_vec))}')
        # print(f'cos_similarity reference2: {cosine_similarity(np.angle(encoded_vector), np.angle(ref_encoded_vec_2))}')
        # orig_difference = np.abs(encoded_vector - target_index)
        # print(f'original difference: {orig_difference}')
        # orig_similarity = np.abs(np.exp(1j*orig_difference).sum())/N
        # print(f'original simlarity: {orig_similarity}')
        # phase_encoded_vector = np.angle(encoded_vector)
        # print(f'phase_encoded_vector: {phase_encoded_vector}')

        # recovered = network.recover_pattern(phase_encoded_vector)

        # Decode vectors
        decoding_matrix = random_tpam_w_decode(S, unique_digits, K) # Generate decoding matrix (D x N)
        print(f'decoding_matrix: {decoding_matrix.shape}')
        # decoded_vector = decoding_matrix@encoded_vector # Final decoded vector (D x 1)
        # decoded_vector = decoding_matrix@recovered # Final decoded vector (D x 1)

        decoded_vector = decode_phase_encoded_vector(decoding_matrix, encoded_vector, K, N, D)[0]
        # print(f'decoded_vector: {decoded_vector.shape}')
        print(f'decoded_vector: {decoded_vector}')
        rgb_decoded_vector = get_rgb_from_phasor(decoded_vector, 256)
        print(f'rgb_decoded_vector: {rgb_decoded_vector}')
        

        plt.imshow(rgb_decoded_vector.reshape((28,28)))
        plt.show()

    # plt.imshow(target.reshape((28,28)))
    # plt.show()

    # plt.imshow(merged.reshape((28,28)))
    # plt.show()
    
# test_encode_decode()
# test_encode_decode(ortho=False)
# test_random_tpam_network()
# test_rgb_merge()
#test_two_three_decode()
test_random_tpam_network(num_cycles= 8, arg_max=True)

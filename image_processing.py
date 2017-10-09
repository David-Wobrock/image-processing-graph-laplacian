import argparse
import sampling
import affinity_methods
import numpy as np
from utils import rgb2ycc, ycc2rgb
from scipy import misc
from scipy.linalg import eigh, sqrtm, svd
from scipy.spatial import distance
import logging
import time
import sys

logger = logging.getLogger(__name__)


def Permutation(phi, sample_indices):
    start = time.time()
    k1 = 0  # Sample pixels
    k2 = len(sample_indices)  # Remaining pixels

    correct_indices = np.ones(phi.shape[0], dtype=np.uint32)
    # TODO loop can be optimised (less conditions)
    for i in range(phi.shape[0]):
        if k1 < len(sample_indices) and i == sample_indices[k1]:
            correct_indices[i] = k1
            k1 += 1
        else:
            correct_indices[i] = k2
            k2 += 1
    print('Permutation done in {}s'.format(time.time() - start))
    return phi[correct_indices, :]


def Nystroem(y, sample_indices, affinity_function):
    start = time.time()
    M, N = y.shape[:2]

    AB = affinity_function(y, sample_indices)

    K_A = AB[:, sample_indices]
    v = np.asarray(range(M*N))
    v = np.delete(v, sample_indices)
    K_AB = AB[:, v]

    phi_A, Pi, _ = svd(K_A)
    phi_A[:, ::2] *= -1
    phi = np.concatenate((
        phi_A,
        np.dot(
            np.dot(K_AB.T, phi_A),
            np.linalg.inv(np.diag(Pi)))))

    print('Nystrom done in {}s'.format(time.time() - start))
    return phi, Pi


def Sinkhorn(phi, Pi):
    start = time.time()
    n, m = phi.shape[:2]
    r = np.ones(n)
    for i in range(100):
        c = 1./(np.dot(phi, (Pi * (np.dot(phi.T, r).T)).T))
        r = 1./(np.dot(phi, (Pi * (np.dot(phi.T, c).T)).T))
    v = np.repeat(c, m).reshape(n, m) * phi
    ABw = np.empty([m, n])
    for i in range(m):  # Can be parallelised
        ABw[i, :] = np.dot((r[i] * (Pi.T * phi[i, :])), v.T)
    W_A = ABw[:, :m]
    W_AB = ABw[:, m:n]
    print('Sinkhorn done in {}s'.format(time.time() - start))
    return W_A, W_AB


def Orthogonalization(W_A, W_AB):
    start = time.time()
    W_Ah = np.zeros(W_A.shape)
    np.fill_diagonal(W_Ah, 1./ (W_A.diagonal() ** 0.5))
    Q = W_A + np.dot(W_Ah, W_AB).dot(W_AB.T).dot(W_Ah)

    U, L, _ = svd(Q)
    Lh = np.zeros(np.diag(L).shape)
    np.fill_diagonal(Lh, 1./(L ** 0.5))
    V = np.concatenate((W_A, W_AB.T)).dot(W_Ah).dot(U).dot(Lh)

    Lambda = L
    Lambda[Lambda>1] = 1
    print('Orthogonalization done in {}s'.format(time.time() - start))
    return V, Lambda


def image_processing(y, **kwargs):
    start = time.time()
    M, N = y.shape[:2]

    # Sampling
    #sampling_code = kwargs.get('sampling', 'random')
    sampling_code = kwargs.get('sampling', 'spatially_uniform')
    sampling_function = sampling.get_sample_function(sampling_code)
    sample_size = int(M*N*0.01)
    #sample_size = 100
    sample_indices = sampling_function(M, N, sample_size)
    logger.info('Number of sample pixels: Theory {0} / Real {1}'.format(sample_size, len(sample_indices)))

    # Nystroem
    affinity_code = kwargs.get('affinity', 'NLM')
    #affinity_code = affinity_methods.get('affinity', 'bilateral')
    affinity_function = affinity_methods.get_affinity_function(affinity_code)
    phi, Pi = Nystroem(y, sample_indices, affinity_function)

    # Permute pixel order in eigenvectors of affinity matrix
    phi = Permutation(phi, sample_indices)
    # Display affinity vector of a pixel
    K = np.dot(phi, Pi).dot(phi.T)
    pass

    W_A, W_AB = Sinkhorn(phi, Pi)
    V, Lambda = Orthogonalization(W_A, W_AB)
    
    # Display eigenvalues
    #plt.figure(5)
    #plt.plot(Lambda[:10])
    np.savetxt('results/eigenvalues.txt', Lambda[:10])
    V = Permutation(V, sample_indices)
    #plt.figure(6)
    #plt.imshow(V[:, 0].reshape(N, M).T, cmap='gray')
    misc.imsave('results/eigenvector1.jpg', V[:, 0].reshape(N, M).T)
    #plt.figure(7)
    #plt.imshow(V[:, 1].reshape(N, M).T, cmap='gray')
    misc.imsave('results/eigenvector2.jpg', V[:, 1].reshape(N, M).T)
    #plt.figure(8)
    #plt.imshow(V[:, 2].reshape(N, M).T, cmap='gray')
    misc.imsave('results/eigenvector3.jpg', V[:, 2].reshape(N, M).T)
    #plt.figure(9)
    #plt.imshow(V[:, 3].reshape(N, M).T, cmap='gray')
    misc.imsave('results/eigenvector4.jpg', V[:, 3].reshape(N, M).T)

    # TODO Display filter eigenvector
    Z = np.dot(
        V,
        ((Lambda.T) * np.dot(V.T, y.reshape(M*N)))).reshape(N, M).T
    print('Program done in {}s'.format(time.time() - start))
    return Z


def set_up_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    root.addHandler(ch)


if __name__ == '__main__':
    # TODO command line tool for selecting different
    # variations
    # + name of the input picture
    # + display result or save it to file
    parser = argparse.ArgumentParser(
        description='Image Processing using Graph Laplacian Operator')
    parser.add_argument(
        '-save',
        action='store_true')

    args = parser.parse_args()
    if not args.save:  # If I don't save, I display
        import matplotlib.pyplot as plt

    set_up_logging()

    #img_name = 'flower_noisy.jpg'
    img_name = 'mountain_noisy.jpg'
    #img_name = 'Lena.png'
    #img_name = 'house.jpg'

    y = misc.imread(img_name)
    y = rgb2ycc(y)
    print("Image shape {}".format(y.shape))
    if len(y.shape) == 2:  # Detect gray scale
        plt.gray()

    y_ycc = y[:, :, 0]
    z_ycc = image_processing(y_ycc)
    z = y.copy()
    z[:, :, 0] = z_ycc.copy()

    # Grayscale of both pictures
    #plt.figure(3)
    #plt.imshow(y_ycc, cmap='gray')
    #plt.figure(4)
    #plt.imshow(z_ycc, cmap='gray')

    if args.save:
        z = ycc2rgb(z)
        misc.imsave('results/output.jpg', z.astype(np.uint8, copy=False))
    else:
        plt.figure(1)
        y = ycc2rgb(y)
        plt.imshow(y.astype(np.uint8, copy=False))

        plt.figure(2)
        z = ycc2rgb(z)
        plt.imshow(z.astype(np.uint8, copy=False))
        plt.show()

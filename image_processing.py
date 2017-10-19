import sampling
import affinity_methods
import numpy as np
import matplotlib.pyplot as plt
from utils import rgb2ycc, ycc2rgb
from scipy import misc
from scipy.linalg import eigh, sqrtm, svd
from scipy.spatial import distance
import logging
import time
import sys

from utils import xy2num

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
    phi = np.concatenate((
        phi_A,
        np.dot(
            np.dot(K_AB.T, phi_A),
            np.linalg.inv(np.diag(Pi)))))

    print('Nystrom done in {0}s'.format(time.time() - start))
    return phi, Pi


def Sinkhorn(phi, Pi):
    start = time.time()
    M, N = phi.shape[:2]
    r = np.ones(M)
    for i in range(100):
        c = 1./(np.dot(phi, (Pi * (np.dot(phi.T, r).T)).T))
        r = 1./(np.dot(phi, (Pi * (np.dot(phi.T, c).T)).T))
    v = np.repeat(c, N).reshape(M, N) * phi
    ABw = np.empty([N, M])
    for i in range(N):  # Can be parallelised
        ABw[i, :] = np.dot((r[i] * (Pi.T * phi[i, :])), v.T)
    W_A = ABw[:, :N]
    W_AB = ABw[:, N:M]
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

def display_affinity_matrix(M, N, phi, Pi, sample_indices, pixel_x, pixel_y):
    # Permute pixel order in eigenvectors of affinity matrix
    phi = Permutation(phi, sample_indices)

    affinity_pixel = xy2num(pixel_x, pixel_y, M, N)
    K_pixel = np.dot((phi[affinity_pixel, :] * Pi), phi.T).reshape(M, N)
    # TODO if save
    plt.imsave(
        'results/affinity_{0}x{1}.jpg'.format(pixel_x, pixel_y),
        K_pixel,
        vmin=0,
        vmax=1,
        cmap='RdBu_r')
    logger.info('Displayed affinity matrix of pixel {0}x{1}'.format(pixel_x, pixel_y))


def adaptive_sharpening(y, cr, cb, phi, Pi):
    start = time.time()
    M, N = y.shape[:2]
    num_pixels = M * N
    alpha = 1./num_pixels

    beta = 1.5
    beta_crcb = 0.5

    y_vector = y.reshape(num_pixels)
    cr_vector_init = cr.reshape(num_pixels)
    cb_vector_init = cb.reshape(num_pixels)

    z_vector = np.empty(num_pixels)
    cr_vector = np.empty(num_pixels)
    cb_vector = np.empty(num_pixels)
    for i in range(num_pixels): # TODO can be parallelised
        k_i = np.dot((phi[i, :] * Pi), phi.T)
        d_i = sum(k_i)
        w_i = 1 - alpha * (d_i - k_i)
        w_i2 = w_i * w_i
        w_i3 = w_i2 * w_i

        f_i = (1 + beta)*w_i2 - beta*w_i3
        z_vector[i] = np.dot(f_i, y_vector)

        f_i_crcb = (1 + beta_crcb)*w_i2 - beta_crcb*w_i3
        cr_vector[i] = np.dot(f_i_crcb, cr_vector_init)
        cb_vector[i] = np.dot(f_i_crcb, cb_vector_init)
        if i%1000 == 0:
            print(i)

    z = z_vector.reshape(M, N)
    cr = cr_vector.reshape(M, N)
    cb = cb_vector.reshape(M, N)
    print('Adaptive sharpening done in {0}s'.format(time.time() - start))
    return z, cr, cb


def image_processing(y, cr, cb, **kwargs):
    start = time.time()
    M, N = y.shape[:2]

    # Sampling
    sampling_code = kwargs.get('sampling', sampling.SPATIALLY_UNIFORM)
    sampling_function = sampling.methods[sampling_code]
    sample_size = int(M*N*0.01)
    #sample_size = 20
    sample_indices = sampling_function(M, N, sample_size)
    logger.info('Number of sample pixels: Theory {0} / Real {1}'.format(sample_size, len(sample_indices)))

    # Nystroem
    affinity_code = kwargs.get('affinity', affinity_methods.PHOTOMETRIC)
    affinity_function = affinity_methods.methods[affinity_code]
    phi, Pi = Nystroem(y, sample_indices, affinity_function)

    # Display affinity vector of a pixel
    #display_affinity_matrix(M, N, phi, Pi, sample_indices, 80, 120)
    #display_affinity_matrix(M, N, phi, Pi, sample_indices, 165, 65)
    display_affinity_matrix(M, N, phi, Pi, sample_indices, 30, 30)
    display_affinity_matrix(M, N, phi, Pi, sample_indices, 125, 125)

    #V = Permutation(V, sample_indices)
    #z = np.dot(
    #    V,
    #    ((Lambda.T) * np.dot(V.T, y.reshape(M*N)))).reshape(M, N)
    
    # Display eigenvalues
    # TODO
    #plt.figure(5)
    #plt.plot(Lambda[:10])
    #np.savetxt('results/eigenvalues.txt', Lambda[:50])
    #V = Permutation(V, sample_indices)
    #plt.figure(6)
    #plt.imshow(V[:, 0].reshape(N, M).T, cmap='gray')
    #plt.imsave('results/eigenvector1.jpg', V[:, 0].reshape(M, N), cmap='gray')
    #plt.figure(7)
    #plt.imshow(V[:, 1].reshape(N, M).T, cmap='gray')
    #plt.imsave('results/eigenvector2.jpg', V[:, 1].reshape(M, N), cmap='gray')
    #plt.figure(8)
    #plt.imshow(V[:, 2].reshape(N, M).T, cmap='gray')
    #plt.imsave('results/eigenvector3.jpg', V[:, 2].reshape(M, N), cmap='gray')
    #plt.figure(9)
    #plt.imshow(V[:, 3].reshape(N, M).T, cmap='gray')
    #plt.imsave('results/eigenvector4.jpg', V[:, 3].reshape(M, N), cmap='gray')

    # TODO Display filter eigenvector
    #z = np.dot(
    #    V,
    #    ((Lambda.T) * np.dot(V.T, y.reshape(M*N)))).reshape(M, N)
    phi = Permutation(phi, sample_indices)

    # Denoising
    #num_pixels = M*N
    #alpha = 1./num_pixels
    #y_vector = y.reshape(num_pixels)
    #z_vector = np.empty(num_pixels)
    #for i in range(num_pixels):
    #    k_i = np.dot((phi[i, :] * Pi), phi.T)
    #    d_i = sum(k_i)
    #    z_vector[i] = np.dot((1 - alpha*(d_i - k_i)), y_vector)
    #    print(i)
    #z = z_vector.reshape(M, N)  # To matrix/image
    alpha = 1./ (M*N)
    beta = 1.5
    beta_crcb = 0.5

    k = np.dot((phi * Pi), phi.T)
    d = np.diag(np.sum(k, axis=0))
    alpha = 1./(M*N)
    i = np.identity(k.shape[0])
    W = i - alpha*(d-k)
    W2 = W ** 2
    W3 = W2 * W
    F = (1+beta)*W2 - beta*W3
    F_crcb = (1+beta_crcb)*W2 - beta_crcb*W3
    z = np.dot(F, y.reshape(M*N)).reshape(M, N)
    cr = np.dot(F_crcb, cr.reshape(M*N)).reshape(M, N)
    cb = np.dot(F_crcb, cb.reshape(M*N)).reshape(M, N)

    # Sharpening
    #z, cr, cb = adaptive_sharpening(y, cr, cb, phi, Pi)
    print('Program done in {}s'.format(time.time() - start))
    return z, cr, cb


def set_up_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    root.addHandler(ch)


if __name__ == '__main__':
    import argparse
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

    set_up_logging()

    #img_name = 'input/flower_noisy.jpg'
    #img_name = 'input/flower_blurry.jpg'
    #img_name = 'input/mountain_noisy.jpg'
    #img_name = 'input/mountain.jpg'
    #img_name = 'input/mountain_noisy_hand2.jpg'
    #img_name = 'input/Lena.png'
    #img_name = 'input/house.jpg'
    #img_name = 'input/small_house.jpg'
    img_name = 'input/beach.jpg'
    #img_name = 'input/theatre_noisy.jpg'
    #img_name = 'input/guy.png'

    y = misc.imread(img_name)
    print("Image '{0}' has shape {1} => {2} pixels".format(img_name, y.shape, y.shape[0]*y.shape[1]))
    if len(y.shape) == 2:  # Detect gray scale
        z = image_processing(y)
        plt.imsave('results/output.jpg', z.astype(np.uint8, copy=False), cmap='gray')
    else:
        y = rgb2ycc(y)
        y_ycc = y[:, :, 0]
        y_cr = y[:, :, 1]
        y_cb = y[:, :, 2]

        z_ycc, z_cr, z_cb = image_processing(y_ycc, y_cr, y_cb)

        z = np.empty(y.shape)
        z[:, :, 0] = z_ycc
        z[:, :, 1] = z_cr
        z[:, :, 2] = z_cb

        y = ycc2rgb(y)
        z = ycc2rgb(z)

        plt.imsave('results/y_ycc.jpg', y_ycc, cmap='gray')
        plt.imsave('results/y_cr.jpg', y_cr, cmap='gray')
        plt.imsave('results/y_cb.jpg', y_cb, cmap='gray')
        plt.imsave('results/z_ycc.jpg', z_ycc, cmap='gray')
        plt.imsave('results/z_cr.jpg', z_cr, cmap='gray')
        plt.imsave('results/z_cb.jpg', z_cb, cmap='gray')

        if args.save:
            plt.imsave('results/input.jpg', y.astype(np.uint8, copy=False))
            plt.imsave('results/output.jpg', z.astype(np.uint8, copy=False))
            z_test = z.copy()
            z_test[:, :, 1] = z_ycc
            z_test[:, :, 1] = y_cr
            z_test[:, :, 2] = y_cb
            z_test = ycc2rgb(z_test)
            plt.imsave('results/output2.jpg', z_test.astype(np.uint8, copy=False))
        else:
            plt.figure(1)
            Nlt.imshow(y.astype(np.uint8, copy=False))

            plt.figure(2)
            plt.imshow(z.astype(np.uint8, copy=False))
            plt.show()

    # Residuals
    r = np.abs(y_ycc - z_ycc)
    plt.imsave('results/residuals.jpg', r, cmap='gray')

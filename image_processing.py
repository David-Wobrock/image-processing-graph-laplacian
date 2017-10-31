import sampling
import affinity_methods
import numpy as np
import matplotlib
matplotlib.use('Agg')
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


def permutation(phi, sample_indices):
    start = time.time()
    k1 = 0  # Current index in sample pixels
    k2 = len(sample_indices)  # Current index in remaining pixels
    num_pixels = phi.shape[0]

    correct_indices = np.ones(num_pixels, dtype=np.uint32)
    for i in range(num_pixels):
        if k1 < len(sample_indices) and i == sample_indices[k1]:
            correct_indices[i] = k1
            k1 += 1
        else:
            correct_indices[i] = k2
            k2 += 1
    logger.info('Permutation done in {0}s'.format(time.time() - start))
    return phi[correct_indices, :]


def nystroem(y, sample_indices, affinity_function):
    start = time.time()
    M, N = y.shape[:2]

    K_AB = affinity_function(y, sample_indices)

    K_A = K_AB[:, sample_indices]
    v = np.asarray(range(M*N))
    v = np.delete(v, sample_indices)
    K_B = K_AB[:, v]

    phi_A, Pi, _ = svd(K_A)
    phi = np.concatenate((
        phi_A,
        np.dot(
            K_B.T,
            np.dot(
                phi_A,
                np.linalg.inv(np.diag(Pi))))))

    logger.info('Nystrom done in {0}s'.format(time.time() - start))
    return phi, Pi


def Sinkhorn(phi, Pi):
    # Adapted from GLIDE, by Milanfar
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
    logger.info('Sinkhorn done in {0}s'.format(time.time() - start))
    return W_A, W_AB


def Orthogonalization(W_A, W_AB):
    # Adapted from GLIDE, by Milanfar
    start = time.time()
    W_Ah = sqrtm(np.linalg.pinv(W_A))
    Q = W_A + np.dot(W_Ah, W_AB).dot(W_AB.T).dot(W_Ah)

    U, L, _ = svd(Q)
    Lh = np.linalg.pinv(np.sqrt(np.diag(L)))
    V = np.concatenate((W_A, W_AB.T)).dot(W_Ah).dot(U).dot(Lh)

    Lambda = L
    Lambda[Lambda>1] = 1
    logger.info('Orthogonalization done in {0}s'.format(time.time() - start))
    return V, Lambda

def orthogonalisation(A, B):
    start = time.time()
    A_sqrt_inv = np.diag(1. / np.sqrt(A.diagonal()))
    #A_sqrt_inv = 1. / np.sqrt(A)
    #A_sqrt_inv = sqrtm(np.linalg.pinv(A))

    S = A + A_sqrt_inv.dot(B).dot(B.T).dot(A_sqrt_inv)
    phi_S, Pi_S, _ = svd(S)

    #Pi_S_sqrt_inv = np.linalg.pinv(np.sqrt(np.diag(Pi_S)))
    Pi_S_sqrt_inv = np.diag(1./(np.sqrt(Pi_S)))

    V = np.concatenate((A, B.T)).dot(A_sqrt_inv).dot(phi_S).dot(Pi_S_sqrt_inv)
    Pi = Pi_S
    #Pi[Pi>1] = 1

    logger.info('Orthogonalisation done in {0}s'.format(time.time() - start))
    return V, Pi


def compute_and_display_affinity_matrix(M, N, phi, Pi, pixel_x, pixel_y):
    concerned_pixel = xy2num(pixel_x, pixel_y, M, N)

    K_pixel = np.dot((phi[concerned_pixel, :] * Pi), phi.T)  # One line of K
    K_pixel = K_pixel.reshape(M, N)

    display_or_save('affinity_{0}x{1}.png'.format(pixel_x, pixel_y), K_pixel, vmin=0, vmax=1, cmap='RdBu_r')
    logger.info('Displayed affinity matrix of pixel {0}x{1}'.format(pixel_x, pixel_y))


def compute_and_display_pixel_degree(M, N, phi, Pi):
    start = time.time()
    D = np.empty(M*N)
    it_len = 1000
    for i in range(0, M*N, it_len):
        D[i:i+it_len] = np.sum(np.dot((phi[i:i+it_len, :] * Pi), phi.T), axis=1)
    D = D.reshape(M, N)
    display_or_save('pixel_degrees.png', D, cmap='RdBu_r')
    logger.info('Displayed degrees of pixels matrix in {0}s'.format(time.time() - start))


def adaptive_sharpening(y, cr, cb, phi, Pi, beta, beta_crcb):
    start = time.time()
    M, N = y.shape[:2]
    num_pixels = M * N
    alpha = 1./num_pixels

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

    z = z_vector.reshape(M, N)
    cr = cr_vector.reshape(M, N)
    cb = cb_vector.reshape(M, N)
    logger.info('Adaptive sharpening done in {0}s'.format(time.time() - start))
    return z, cr, cb


def sharpening_full_matrix(y, cr, cb, phi, Pi, beta, beta_crcb):
    start = time.time()
    M, N = y.shape[:2]
    alpha = 1./ (M*N)

    k = np.dot((phi * Pi), phi.T)
    d = np.diag(np.sum(k, axis=0))
    i = np.identity(k.shape[0])

    W = i - alpha*(d-k)
    W2 = W ** 2
    W3 = W2 * W

    F = (1+beta)*W2 - beta*W3
    #F_crcb = (1+beta_crcb)*W2 - beta_crcb*W3

    z = np.dot(F, y.reshape(M*N)).reshape(M, N)
    #cr = np.dot(F_crcb, cr.reshape(M*N)).reshape(M, N)
    #cb = np.dot(F_crcb, cb.reshape(M*N)).reshape(M, N)
    logger.info('Sharpening full matrix done in {0}s'.format(time.time() - start))
    return z, cr, cb


def smoothing(y, sample_indices, phi, Pi):
    start = time.time()
    M, N = y.shape[:2]
    num_pixels = M*N
    sample_size = len(sample_indices)

    alpha = 1./num_pixels
    y_vector = y.reshape(num_pixels)
    z_vector = np.empty(num_pixels)

    #K = np.dot((phi * Pi), phi.T)
    #D = np.diag(np.sum(K, axis=0))
    #Lapl = alpha * (D-K)
    #W = np.identity(num_pixels) - Lapl
    #z_vector = W.dot(y_vector)
    #return z_vector.reshape(M, N)

    W_AB = np.empty((sample_size, num_pixels))
    # *** Fill W_AB
    for i, sample_idx in enumerate(sample_indices):
        k_i = np.dot((phi[sample_idx, :] * Pi), phi.T)
        d_i = np.zeros(num_pixels)
        d_i[sample_idx] = sum(k_i)
        ident_i = np.zeros(num_pixels)
        ident_i[sample_idx] = 1.
        W_AB[i, :] = ident_i - alpha * (d_i - k_i)

    #W_AB[W_AB<0] = 0
    W_A = W_AB[:, sample_indices]
    v = np.asarray(range(num_pixels))
    v = np.delete(v, sample_indices)
    W_B = W_AB[:, v]

    phi, Pi = orthogonalisation(W_A, W_B)
    #phi_A, Pi, _ = svd(W_A)
    #phi = np.concatenate((
    #    phi_A,
    #    np.dot(
    #        np.dot(W_B.T, phi_A),
    #        np.linalg.inv(np.diag(Pi)))))

    # TODO phi orthonormal ?????? why?
    phi = permutation(phi, sample_indices)

    plt.figure()
    plt.plot(range(1, Pi.shape[0]+1), Pi)
    plt.savefig('results/eigenvalues.png')
    display_or_save('eigvec1.png', phi[:, 0].reshape(M, N), cmap='gray')
    display_or_save('eigvec2.png', phi[:, 1].reshape(M, N), cmap='gray')
    display_or_save('eigvec3.png', phi[:, 2].reshape(M, N), cmap='gray')
    z_vector = np.dot(
        phi,
        ((Pi.T) * np.dot(phi.T, y_vector)))

    z = z_vector.reshape(M, N)
    logger.info('Smoothing done in {0}s'.format(time.time() - start))
    return z


def denoising(y, phi, Pi):
    start = time.time()
    M, N = y.shape[:2]
    num_pixels = M*N
    alpha = 1./num_pixels
    y_vector = y.reshape(num_pixels)
    z_vector = np.empty(num_pixels)

    k = np.dot((phi * Pi), phi.T)
    d = np.diag(np.sum(k, axis=0))

    z_vector = np.dot(alpha*(d-k), y_vector)

    z = z_vector.reshape(M, N)
    logger.info('Denoising full matrix done in {0}s'.format(time.time() - start))
    return z


def image_processing(y, cr=None, cb=None, **kwargs):
    start = time.time()
    M, N = y.shape[:2]

    # Sampling
    sampling_code = kwargs.get('sampling', sampling.SPATIALLY_UNIFORM)
    sampling_function = sampling.methods[sampling_code]
    #sample_size = int(M*N*0.001)
    sample_size = 200
    sample_indices = sampling_function(M, N, sample_size)
    logger.info('Number of sample pixels: Theory {0} / Real {1}'.format(sample_size, len(sample_indices)))

    # Nystroem
    affinity_code = kwargs.get('affinity', affinity_methods.PHOTOMETRIC)
    affinity_function = affinity_methods.methods[affinity_code]
    phi, Pi = nystroem(y, sample_indices, affinity_function)
    
    # Display affinity vector of a pixel
    phi_perm = permutation(phi, sample_indices)
    compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 65, 65)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 165, 65)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 80, 120)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 430, 55)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 92, 357)

    #compute_and_display_pixel_degree(M, N, phi, Pi)
    #z = y

    # Smoothing
    z = smoothing(y, sample_indices, phi_perm, Pi)

    # Sinkhorn
    #W_A, W_B = Sinkhorn(phi, Pi)
    #V, L = Orthogonalization(W_A, W_B)
    #V = permutation(V, sample_indices)
    #z = np.dot(V, L * np.dot(V.T, y.reshape(M*N))).reshape(M, N)

    # Denoising
    #z = denoising(y, phi, Pi)

    # Sharpening full matrix
    beta = 1.6
    beta_crcb = 0.6
    #z, cr, cb = sharpening_full_matrix(y, cr, cb, phi_perm, Pi, beta, beta_crcb)

    # Sharpening
    #z, cr, cb = adaptive_sharpening(y, cr, cb, phi, Pi, beta, beta_crcb)

    logger.info('Program done in {0}s'.format(time.time() - start))
    return z, cr, cb


def set_up_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    root.addHandler(ch)


def display_or_save(name, img, cmap=None, vmin=None, vmax=None):
    global save_image
    if save_image:
        plt.imsave('results/{0}'.format(name), img, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        fig = plt.figure()
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.suptitle(name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Image Processing using Graph Laplacian Operator')
    parser.add_argument(
        'image_name',
        type=str,
        nargs=1)
    parser.add_argument(
        '-save',
        action='store_true')

    args = parser.parse_args()
    global save_image
    save_image = args.save
    img_name = args.image_name[0]

    set_up_logging()

    y = misc.imread(img_name)
    #y = y[:, :, 0]
    logger.info("Image '{0}' has shape {1} => {2} pixels".format(img_name, y.shape, y.shape[0]*y.shape[1]))
    if len(y.shape) == 2:  # Detect gray scale
        z = image_processing(y)[0]
        display_or_save('input.png', y, cmap='gray')
        display_or_save('output.png', z.astype(np.uint8, copy=False), cmap='gray')
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

        display_or_save('y_ycc.png', y_ycc, cmap='gray')
        display_or_save('y_cr.png', y_cr, cmap='gray')
        display_or_save('y_cb.png', y_cb, cmap='gray')
        display_or_save('z_ycc.png', z_ycc, cmap='gray')
        display_or_save('z_cr.png', z_cr, cmap='gray')
        display_or_save('z_cb.png', z_cb, cmap='gray')

        display_or_save('input.png', y.astype(np.uint8, copy=False))
        display_or_save('output.png', z.astype(np.uint8, copy=False))

        # Residuals
        r = np.abs(y_ycc - z_ycc)
        display_or_save('residuals.png', r, cmap='gray')

    if not save_image:
        plt.show()

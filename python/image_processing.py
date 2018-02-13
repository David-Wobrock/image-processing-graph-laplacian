import sampling
import affinity_methods
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import rgb2ycc, ycc2rgb, num2xy
from scipy import misc
from scipy.linalg import eigh, sqrtm, svd, orth
from scipy.spatial import distance
import logging
import time
import sys

from utils import xy2num

logger = logging.getLogger(__name__)

def display_sample_pixels(y, sample_indices):
    M, N = y.shape
    y_samples = np.empty((M, N, 3), dtype=np.uint8)  # RGB
    y_samples[:, :, 0] = y
    y_samples[:, :, 1] = y
    y_samples[:, :, 2] = y
    red_pixel = (255, 0, 0)
    for index in sample_indices:
        x, y = num2xy(index, M, N)
        for i in range(-5, 6):
            for j in range(-5, 6):
                y_samples[x+i, y+j] = red_pixel

    display_or_save('sample_pixels.png', y_samples)


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


def affinity(y, sample_indices, affinity_function):
    start = time.time()

    M, N = y.shape[:2]

    K_AB = affinity_function(y, sample_indices)  # Can be parallelised

    start_split = time.time()
    K_A = K_AB[:, sample_indices]
    v = np.asarray(range(M*N))
    v = np.delete(v, sample_indices)
    K_B = K_AB[:, v]
    logger.info('Split K_AB into K_A and K_B in {0}s'.format(time.time() - start_split))
    return K_A, K_B


def nystroem(K_A, K_B):
    start = time.time()
    start_svd = time.time()
    phi_A, Pi, _ = svd(K_A)
    #Pi, phi_A = np.linalg.eigh(K_A)
    #Pi = Pi[::-1]
    #phi_A = phi_A[:, ::-1]
    logger.info('SVD size {0} took {1}s'.format(K_A.shape, time.time() - start_svd))

    start_phi_approx = time.time()
    phi = np.concatenate((
        phi_A,
        np.dot(
            K_B.T,
            phi_A * (1./Pi))))
    logger.info('Compute approx phi of K took {0}s'.format(time.time() - start_phi_approx))

    logger.info('Nystrom done in {0}s'.format(time.time() - start))
    return phi, Pi


def sinkhorn(phi, Pi):
    # Adapted from GLIDE, by Milanfar
    start = time.time()
    M, N = phi.shape[:2]
    r = np.ones(M)
    for i in range(100):
        c = 1. / phi.dot(Pi * phi.T.dot(r))
        c = np.nan_to_num(c)
        r = 1. / phi.dot(Pi * phi.T.dot(c))
        r = np.nan_to_num(r)
    phi_c = phi * np.repeat(c, N).reshape(M, N)
    W_AB = np.empty([N, M])
    for i in range(N):  # Can be parallelised
        W_AB[i, :] = (r[i]*phi[i, :]*Pi).dot(phi_c.T)
    W_A = W_AB[:, :N]
    W_B = W_AB[:, N:M]
    logger.info('Sinkhorn done in {0}s'.format(time.time() - start))
    return W_A, W_B


def orthogonalisation(A, B):
    start = time.time()

    phi, Pi, _ = svd(A)
    Pi_sqrt_inv = 1./np.sqrt(Pi)
    A_sqrt_inv = (phi*Pi_sqrt_inv).dot(phi.T)

    Q = A + A_sqrt_inv.dot(B).dot(B.T).dot(A_sqrt_inv)
    phi_Q, Pi_Q, _ = svd(Q)

    Pi_Q_sqrt_inv = np.diag(1./(np.sqrt(Pi_Q)))

    V = np.concatenate((A, B.T)).dot(A_sqrt_inv).dot(phi_Q).dot(Pi_Q_sqrt_inv)
    Pi = Pi_Q
    Pi[Pi>1] = 1

    logger.info('Orthogonalisation done in {0}s'.format(time.time() - start))
    return V, Pi


def compute_and_display_affinity_matrix(M, N, phi, Pi, pixel_x, pixel_y):
    concerned_pixel = xy2num(pixel_x, pixel_y, M, N)

    K_pixel = np.dot((phi[concerned_pixel, :] * Pi), phi.T)  # One line of K
    K_pixel = K_pixel.reshape(M, N)

    display_or_save('affinity_{0}x{1}.png'.format(pixel_x, pixel_y), K_pixel, vmin=0, vmax=1, cmap='gray')
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


def smoothing_matrix(sample_indices, phi, Pi):
    start = time.time()

    K = (phi * Pi).dot(phi.T)
    size = K.shape[0]
    D = np.sum(K, axis=1)
    ##return np.diag(1./D).dot(K)  # with Random Walk Laplacian
    alpha = 1./np.mean(D)
    Lapl = alpha*(np.diag(D)-K)
    #return Lapl
    W = np.identity(size) + alpha*(K - np.diag(D))
    W_A = W[:len(sample_indices), :len(sample_indices)]
    W_B = W[:len(sample_indices), len(sample_indices):]
    #return W  # Renormalised Laplacian

    #W_A, W_B = sinkhorn(phi, Pi)
    #W_A[W_A<0] = 0
    #K_f = (phi[:len(sample_indices)] * Pi).dot(phi.T)
    #D_f = np.sum(K_f, axis=1)
    #alpha_f = 1./np.mean(D_f)
    #ident = np.concatenate(
    #    (np.identity(len(D_f)),
    #     np.zeros((K_f.shape[0], K_f.shape[1]-len(sample_indices)))),
    #    axis=1)
    #d_full = np.concatenate(
    #    (np.diag(D_f), np.zeros((K_f.shape[0], K_f.shape[1]-len(sample_indices)))), axis=1)
    #WAWB = ident + alpha_f*(K_f - d_full)
    #W_A = WAWB[:, :len(sample_indices)]
    #W_B = WAWB[:, len(sample_indices):]
    #V, L = orthogonalisation(W_A, W_B)
    #phi_A, L, _ = svd(W_A)
    L, phi_A = np.linalg.eigh(W_A)
    L = L[::-1]
    phi_A = phi_A[:, ::-1]
    V = np.concatenate((
        phi_A,
        np.dot(
            W_B.T,
            phi_A * (1./L))))
    V = permutation(V, sample_indices)
    #V = GS(V)

    logger.info('Returned smoothing matrix in {0}s'.format(time.time() - start))
    return V, L


def smoothing(y, sample_indices, phi, Pi):
    start = time.time()
    M, N = y.shape[:2]
    num_pixels = M*N

    y_vector = y.reshape(num_pixels)
    z_vector = np.empty(num_pixels)

    V, L = smoothing_matrix(sample_indices, phi, Pi)

    plt.figure()
    plt.plot(range(1, L.shape[0]+1), L)
    plt.savefig('results/eigenvalues.png')
    display_or_save('eigvec1.png', V[:, 0].reshape(M, N), cmap='gray')
    display_or_save('eigvec2.png', V[:, 1].reshape(M, N), cmap='gray')
    display_or_save('eigvec3.png', V[:, 2].reshape(M, N), cmap='gray')

    z_vector = np.dot(V, L * np.dot(V.T, y_vector))
    #W = smoothing_matrix(sample_indices, permutation(phi, sample_indices), Pi)
    #z_vector = W.dot(y_vector)
    z = z_vector.reshape(M, N)
    logger.info('Smoothing done in {0}s'.format(time.time() - start))
    return z


def sharpening(y, sample_indices, phi, Pi):
    start = time.time()
    M, N = y.shape[:2]
    num_pixels = M*N

    y_vector = y.reshape(num_pixels)
    z_vector = np.empty(num_pixels)

    beta = 1.5
    phi, Pi = smoothing_matrix(sample_indices, phi, Pi)

    start_sharpening_filter = time.time()
    w_square_y = phi.dot(Pi * phi.T.dot(phi.dot(Pi * phi.T.dot(y_vector))))
    w_cube_y = phi.dot(Pi * phi.T.dot(w_square_y))
    z_vector = (1+beta)*w_square_y - beta*w_cube_y
    logger.info('Sharpening filter computation in {0}s'.format(time.time() - start_sharpening_filter))
    z = z_vector.reshape(M, N)

    logger.info('Sharpening done in {0}s'.format(time.time() - start))
    return z


def image_processing(y, cr=None, cb=None, **kwargs):
    start = time.time()
    M, N = y.shape[:2]

    # Sampling
    sampling_code = kwargs.get('sampling', sampling.SPATIALLY_UNIFORM)
    sampling_function = sampling.methods[sampling_code]
    sample_size = int(M*N*0.01)
    #sample_size = 200
    sample_indices = sampling_function(M, N, sample_size)
    logger.info('Number of sample pixels: Theory {0} / Real {1} (or {2:.2f}% of the all pixels)'.format(sample_size, len(sample_indices), (len(sample_indices)*100.)/(M*N)))
    #display_sample_pixels(y, sample_indices)

    B_size = (len(sample_indices), (M*N) - len(sample_indices))
    # Nystroem
    affinity_code = kwargs.get('affinity', affinity_methods.BILATERAL)
    affinity_function = affinity_methods.methods[affinity_code]
    K_A, K_B = affinity(y, sample_indices, affinity_function)

    phi, Pi = nystroem(K_A, K_B)
    Pi = Pi[::-1]
    phi = phi[:, ::-1]
    phi = permutation(phi, sample_indices)
    K = (phi*Pi).dot(phi.T)
    D = np.sum(K, axis=1)
    alphaF = 1./np.mean(D)
    D = np.diag(D)
    Lapl = alphaF*(D-K)
    W = np.identity(M*N) - Lapl

    D_A = np.sum(K_A, axis=1) + np.sum(K_B, axis=1)
    alpha = 1./np.mean(D_A)
    D_A = np.diag(D_A)
    L_A = alpha * (D_A - K_A)
    L_B = -alpha*K_B
    #L_B = -alpha*K_B
    #D_inv_sqrt = np.diag(1./np.sqrt(D_A))
    #mean = [np.mean(1./np.sqrt(D_A))] * (M*N - len(sample_indices))
    #D_inv_sqrtB = np.diag(mean)
    #L_A = np.identity(len(sample_indices)) - np.dot(D_inv_sqrt, K_A).dot(D_inv_sqrt)
    #L_B = -(np.dot(D_inv_sqrt, K_B).dot(D_inv_sqrtB))
    #phi, Pi = nystroem(L_A, L_B)

    mu, phi_A = np.linalg.eigh(L_A)
    phi = np.concatenate((
        phi_A,
        np.dot(
            L_B.T,
            phi_A * (1./mu))))
    Pi = 1-mu
    #Pi = np.linalg.eigvalsh(Lapl)

    plt.figure()
    plt.plot(range(1, mu.shape[0]+1), mu)
    plt.savefig('results/eigenvalues_Lapl.png')
    plt.figure()
    plt.plot(range(1, Pi.shape[0]+1), Pi)
    plt.savefig('results/eigenvalues_W.png')
    phi = permutation(phi, sample_indices)

    y_vector = y.reshape(M*N)
    z_vector = y_vector - np.dot(phi*(mu+5), phi.T.dot(y_vector))
    #z_vector = y_vector - Lapl.dot(y_vector)
    #z_vector = Lapl.dot(y_vector)
    #z_vector = np.dot(phi*Pi, phi.T.dot(y_vector))
    #z_vector = (np.identity(M*N) - np.dot(phi*Pi, phi.T)).dot(y_vector)
    #z_vector = W.dot(y_vector)
    z = z_vector.reshape(M, N)

    """
    phi, Pi = nystroem(K_A, K_B)
    phi = permutation(phi, sample_indices)
    K = (phi*Pi).dot(phi.T)
    #D = np.sum(K_A, axis=1) + np.sum(K_B, axis=1)
    D = np.sum(K, axis=1)
    alpha = 1./np.mean(D)
    D_diag = np.diag(D)
    Lapl = alpha*(D_diag-K)
    W = np.identity(M*N) - Lapl
    z = W.dot(y.reshape(M*N)).reshape(M, N)
    """

    # Display affinity vector of a pixel
    #phi_perm = permutation(phi, sample_indices)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 5, 5)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 165, 65)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 80, 120)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 2373, 3441)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 2169, 3729)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 1185, 2270)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 2632, 188)
    #compute_and_display_affinity_matrix(M, N, phi_perm, Pi, 1085, 2270)

    #compute_and_display_pixel_degree(M, N, phi, Pi)

    # Smoothing
    #z = smoothing(y, sample_indices, phi, Pi)
    #z = np.empty(M*N)
    #y_vector = y.reshape(M*N)
    #for i in range(M*N):
    #    print(i)
    #    dirac = np.zeros(M*N)
    #    dirac[i] = 1.
    #    k_vec = (phi_perm[i] * Pi).dot(phi_perm.T)
    #    d_i = np.sum(k_vec)
    #    alpha = 0.0006
    #    z[i] = (dirac + alpha*(k_vec - (d_i*dirac))).dot(y_vector)
    #z = z.reshape(M, N)

    # Sharpening
    #z = sharpening(y, sample_indices, phi, Pi)

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


def residuals(y, z):
    r = np.abs(y - z)
    display_or_save('residuals.png', r, cmap='gray')


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
    #y = y[:,:,0]
    logger.info("Image '{0}' has shape {1} => {2} pixels".format(img_name, y.shape, y.shape[0]*y.shape[1]))
    if len(y.shape) == 2:  # Detect gray scale
        z = image_processing(y)[0]
        display_or_save('input.png', y, cmap='gray')
        display_or_save('output.png', z.astype(np.uint8, copy=False), cmap='gray')
        residuals(y, z)
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

        residuals(y_ycc, z_ycc)

    if not save_image:
        plt.show()

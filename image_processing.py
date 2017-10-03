import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import misc
from scipy.linalg import eigh, sqrtm, svd
from scipy.spatial import distance


def sample_img_random(img):
    """Random sampling"""
    nx, ny = img.shape
    N = nx*ny
    n = round(N*0.002)  # Number of sample pixels
    all_pixels = set()
    for i in range(nx):
        for j in range(ny):
            #data_r, data_g, data_b = img[i, j].tolist()
            #all_pixels.add((i, j, data_r, data_g, data_b))
            gray_scale = float(img[i, j])
            all_pixels.add((i, j, gray_scale))
    sample_pixels = random.sample(all_pixels, n)
    remaining_pixels = all_pixels.difference(sample_pixels)
    return sample_pixels, remaining_pixels


def sample_img(img):
    """Spatially uniform sampling"""
    nx, ny = img.shape
    sample_distance = 20

    all_pixels = set()
    for i in range(nx):
        for j in range(ny):
            gray_scale = float(img[i, j])
            all_pixels.add((i, j, gray_scale))

    sample_pixels = set()
    for i in range(2, nx, sample_distance):
        for j in range(2, ny, sample_distance):
            sample_pixels.add((i, j, float(img[i, j])))

    remaining_pixels = all_pixels.difference(sample_pixels)
    print("Number of sample pixels: {}".format(len(sample_pixels)))
    return list(sample_pixels), list(remaining_pixels)


def euclidean_dist(i, j):
    euclidean_dist_location = np.sqrt(np.power(j[0] - i[0], 2) + np.power(j[1] - i[1], 2))
    euclidean_dist_colors = np.sqrt(np.power(j[2] - i[2], 2) + np.power(j[3] - i[3], 2) + np.power(j[4] - i[4], 2))
    # 0.8 and 0.2 are just arbitrary
    return euclidean_dist_location*0.8 + euclidean_dist_colors*0.2

def euclidean_dist_gray(i, j):
    euclidean_dist_location = np.sqrt(np.power(j[0] - i[0], 2) + np.power(j[1] - i[1], 2))
    euclidean_dist_colors = np.sqrt(np.power(j[2] - i[2], 2))
    # 0.8 and 0.2 are just arbitrary
    return euclidean_dist_location*0.8 + euclidean_dist_colors*0.2

def bilateral(i, j):
    sigma_d = 10
    sigma_r = 15
    location_diff = np.power(
        np.power(i[0] - j[0], 2) + np.power(i[1] - j[1], 2),
        2)
    intensity_diff = np.power(np.abs(i[2] - j[2]), 2)
    return np.exp(
        -(location_diff / (2*sigma_d*sigma_d)) - (intensity_diff / (2*sigma_r*sigma_d)))


def gaussian(i, j):
    sigma = 10
    norm = np.linalg.norm(np.asarray(i) - np.asarray(j))
    return np.exp(- (norm ** 2) / (2*(sigma**2)))


def build_affinity_matrices_from_sample(sample_pixels, remaining_pixels):
    """Affinity matrix with euclidean distance"""
    n = len(sample_pixels)
    m = len(remaining_pixels)
    # Compute A and B
    A = np.empty([n, n], dtype=np.float64)
    for idx_i, i in enumerate(sample_pixels):
        idx_j = idx_i
        for j in sample_pixels[idx_i:]:
            #A[idx_i, idx_j] = distance.euclidean(i, j)
            A[idx_i, idx_j] = ((i[0] - j[0])**2) + ((i[1] - j[1])**2)
            A[idx_j, idx_i] = A[idx_i, idx_j]
            idx_j += 1
    B = np.empty([n, m], dtype=np.float64)
    for idx_i, i in enumerate(sample_pixels):
        print(idx_i)
        for idx_j, j in enumerate(remaining_pixels):
            #B[idx_i, idx_j] = distance.euclidean(i, j)
            B[idx_i, idx_j] = ((i[0] - j[0])**2) + ((i[1] - j[1])**2)

    return A, B


def sinkhorn(approx_eigvecs, diag_eigvals_A):
    n = approx_eigvecs.shape[1]
    N = approx_eigvecs.shape[0]
    r = np.ones(N)
    for i in range(100):
        c = 1. / (np.dot(approx_eigvecs, (diag_eigvals_A * np.dot(approx_eigvecs.T, r))))
        r = 1. / (np.dot(approx_eigvecs, (diag_eigvals_A * np.dot(approx_eigvecs.T, c))))

    v = np.tile(c, [1, n]) * approx_eigvecs.T
    W_AB = np.empty([n, N])
    for i in range(n):  # Can be parallelised 
        W_AB[i, :] = np.dot(np.dot(r[i], (diag_eigvals_A.T * approx_eigvecs[i, :])), v)
    
    return W_AB[:, :n], W_AB[:, n:N]


def orthogonalization(W_A, W_B):
    sqrtW_A = sqrtm(np.inv(W_A))
    Q = W_A + np.dot(
        np.dot(
            np.dot(sqrtW_A, W_B),
            W_B.T),
        sqrtW_A) 
    eigvals_Q, eigvecs_Q = eigh(Q)
    approx_eigvecs = np.dot(
            np.dot(
                np.dot(
                    np.concatenate((W_A, W_B.T)),
                    sqrtW_A),
                eigvecs_Q),
            np.inv(sqrt(eigvals_Q)))

    return eigvals_Q, approx_eigvecs


def global_image_denoising(img):
    sample_pixels, remaining_pixels = sample_img(img)
    K_A, K_B = build_affinity_matrices_from_sample(
        sample_pixels, remaining_pixels)
    assert np.all(K_A == K_A.T)

    #eigvals_K_A, eigvecs_K_A = eigh(K_A)
    eigvecs_K_A, eigvals_K_A, _ = svd(K_A)
    diag_eigvals_K_A = np.diag(eigvals_K_A)

    approx_eigvecs_K = np.concatenate((
        eigvecs_K_A,
        np.dot(
            np.dot(K_B.T, eigvecs_K_A),
            np.linalg.inv(diag_eigvals_K_A))))

    print(approx_eigvecs_K)
    print(approx_eigvecs_K.shape)
    '''W_A, W_B = sinkhorn(approx_eigvecs_K, diag_eigvals_K_A)

    approx_eigvals_W, approx_eigvecs_W = orthogonalization(W_A, W_B)

    approx_filter = approx_eigvecs_W * approx_eigvals_W * approx_eigvecs_W.T
    return approx_filter * img'''
    return img

#img_name = 'Lena.png'
#img_name = 'mountain.jpg'
img_name = 'flower.jpg'

img = misc.imread(img_name)
print("Image shape {}".format(img.shape))
if len(img.shape) == 2:  # Detect gray scale
    plt.gray()

#plt.figure(1)
#plt.imshow(img)

# Create noisy image
sigma = 50
noisy_img = img + np.dot(np.random.random_sample(img.shape), sigma)
plt.figure(2)
plt.imshow(noisy_img)

denoised_img = global_image_denoising(noisy_img)
plt.figure(3)
plt.imshow(denoised_img)

#misc.imsave('test.out.png', approx_affinity_mat)
#plt.show()

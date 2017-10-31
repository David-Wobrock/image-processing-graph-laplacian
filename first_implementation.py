# -*- encoding: utf8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy.linalg import svd, sqrtm, pinv2
from scipy.signal import gaussian
import time


# https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
# https://stackoverflow.com/q/42474491
def im2col(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape[:2]
    s0, s1 = A.strides[:2]
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def Nyst(y, sample_indices):
    start = time.time()
    M, N = y.shape[:2]

    h = 55.
    ksz = 7
    kRad = int((ksz - 1)/2)

    img = np.pad(y, (kRad, kRad), 'symmetric')
    G = matlab_style_gauss2D((ksz,ksz), 1.2)
    G = G.reshape(ksz**2, 1)
    G = G / np.sum(G)

    Z = im2col(img.T, [ksz, ksz])
    Z = Z * np.repeat(G, M*N).reshape(ksz**2, M*N)

    AB = np.empty((len(sample_indices), M*N))
    for i in range(len(sample_indices)):
        ind_y = int(np.floor(sample_indices[i] / M))
        ind_x = int(sample_indices[i] - (ind_y)*M)
        loc_p = [ind_x+kRad, ind_y+kRad]
        yp = img[loc_p[0]-kRad:loc_p[0]+kRad+1, loc_p[1]-kRad:loc_p[1]+kRad+1]
        Zc = yp.T.reshape(ksz**2) * G.reshape(ksz**2)
        Zc = np.repeat(Zc, M*N).reshape(ksz**2, M*N)
        Ker = np.exp(-np.sum((Zc - Z) ** 2, axis=0) / h**2)
        AB[i, :] = Ker

    K_A = AB[:, sample_indices]
    v = np.asarray(range(M*N))
    v[sample_indices] = 0
    v = np.nonzero(v)[0]
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


def Sampling(M, N, sample_dist):
    start = time.time()
    spx = range(2, M, sample_dist)
    spy = range(2, N, sample_dist)
    sample_indices = np.empty(len(spx)*len(spy), dtype=np.uint32)
    c = 0
    for j in spy:
        for i in spx:
            sample_indices[c] = i + M*(j-1)
            c += 1
        
    print('Sampling done in {}s'.format(time.time() - start))
    return np.concatenate(([1], sample_indices)) - 1

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


def ComputeImage(y, phi, Pi):
    start = time.time()
    M, N = y.shape[:2]
    diag_Pi = np.diag(Pi)
    alpha = 1./(M*N)
    Z = np.empty([M, N])
    for i in range(M):
        K_i = np.dot(np.dot(phi, diag_Pi)[i, :], phi.T)
        d_i = sum(K_i)

        # Re-normalised Laplacian
        first_part = (1 - alpha*d_i) * y[i, :]
        second_part = alpha * sum((K_i[j*N:(j+1)*N] * y[:, j]) for j in range(N))
        Z[i, :] = first_part + second_part
    print('Filter done in {}s'.format(time.time() - start))
    return Z


def disp_eigvecs(phi, M, N):
    for i in range(10):
        plt.figure(i+3)
        plt.imshow(phi[:, i].reshape(M, N).T, cmap='gray')


def Sinkhorn(phi, Pi):
    start = time.time()
    n, m = phi.shape[:2]
    r = np.ones(n)
    for i in range(100):
        c = 1./(np.dot(phi, (Pi * (np.dot(phi.T, r).T)).T))
        r = 1./(np.dot(phi, (Pi * (np.dot(phi.T, c).T)).T))
    v = np.repeat(c, m).reshape(n, m) * phi
    ABw = np.empty([m, n])
    for i in range(m):
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


def GLIDE(y):
    start = time.time()
    M, N = y.shape[:2]
    sample_dist = 10

    # Sampling
    sample_indices = Sampling(M, N, sample_dist)

    # Nystroem
    phi, Pi = Nyst(y, sample_indices)

    # Permute pixel order in eigenvectors of affinity matrix
    #phi = Permutation(phi, sample_indices)

    # Display eigenvectors
    #disp_eigvecs(phi, M, N)
    #return

    # Get filter by re-normalised Laplacian
    #Z = ComputeImage(y, phi, Pi)


    W_A, W_AB = Sinkhorn(phi, Pi)
    V, Lambda = Orthogonalization(W_A, W_AB)
    V = Permutation(V, sample_indices)
    Z = np.dot(
        V,
        ((Lambda.T) * np.dot(V.T, y.reshape(M*N)))).reshape(M, N).T
    print('GLIDE done in {}s'.format(time.time() - start))
    return Z


if __name__ == '__main__':
    img_name = 'input/flower.jpg'
    x = misc.imread(img_name).astype(np.double)
    print(x.shape)

    # Noising
    sigma = 50
    y = x + np.dot(
        np.random.random_sample(x.shape),
        sigma)

    z = GLIDE(y)

    # Display
    plt.figure(1)
    plt.imshow(x, cmap='gray')

    plt.figure(2)
    plt.imshow(y, cmap='gray')

    plt.figure(3)
    plt.imshow(z, cmap='gray')

    plt.show()

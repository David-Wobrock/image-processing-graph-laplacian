# -*- encoding: utf8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy.linalg import svd, sqrtm, pinv2
from scipy.signal import gaussian

# https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
# https://stackoverflow.com/q/42474491
def im2col(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
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


def Perm(V, indx):
    c = 0
    Vnew = V
    if indx[-1] == V.shape[0]:
        q = len(indx)-1
        indxx = indx
    else:
        q = len(indx)
        indxx = np.concatenate((indx, [V.shape[0]-1]))
    In = len(indx)
    for i in range(q):
        Vnew[indxx[c], :] = V[c, :]
        l = indxx[c+1] - indxx[c] - 1
        t1 = indxx[c]+1
        t2 = indxx[c+1]-1
        Vnew[t1:t2+1, :] = V[In:In+l, :]
        In += 1
        c += 1
    if indx[-1] == V.shape[0]:
        Vnew[-1, :] = V[c, :]
    return Vnew


def Orth(W_A, W_AB, m_max):
    #W_Ah = np.sqrt(pinv2(W_A))
    W_Ah = np.empty(W_A.shape)
    with open('W_Ah.txt', 'r') as f:
        for i, line in enumerate(f):
            W_Ah[i, :] = np.asarray(line.split(','), dtype=np.double)
    Q = W_A + np.dot(
        np.dot(
            np.dot(W_Ah, W_AB),
            W_AB.T),
        W_Ah)
    U, L, _ = svd(Q)
    U *= -1
    V = np.dot(
            np.dot(
                np.dot(
                    np.concatenate((W_A, W_AB.T)),
                    W_Ah),
                U),
            np.linalg.inv(np.sqrt(np.diag(L))))
    L[L>1] = 1
    if V.shape[1] > m_max:
        V = V[:, :m_max]
        L = L[:m_max]
    return V, L

def Sink(phi, Pi):
    n, m = phi.shape  # n, nb of pixels / m, nb of leading eigenvectors
    r = np.ones([n, 1])
    for k in range(100):
        c = 1./(np.dot(phi, (Pi * (np.dot(phi.T, r).T)).T))
        r = 1./(np.dot(phi, (Pi * (np.dot(phi.T, c).T)).T))
    v = np.repeat(c, m).reshape(n, m) * phi
    ABw = np.empty([m, n])
    for i in range(m):
        ABw[i, :] = np.dot((r[i] * (Pi.T * phi[i, :])), v.T)
    W_A = ABw[:, :m]
    W_AB = ABw[:, m:n]
    return W_A, W_AB

def Nyst(zt, h, sample_indices):
    M, N = zt.shape

    ksz = 7
    kRad = int((ksz - 1)/2)

    img = np.pad(zt, (kRad, kRad), 'symmetric')
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

    return phi, Pi


def Adapted_h(sigma):
    if sigma < 5:
        return 0.8*sigma
    if sigma < 15:
        return 0.6*sigm
    if sigma < 35:
        return 0.3*sigma
    if sigma < 45:
        return 0.2*sigma
    return 0.15*sigma


def Sampling(M, N, sample_dist):
    spx = range(2, M, sample_dist)
    spy = range(2, N, sample_dist)
    sample_indices = np.empty(len(spx)*len(spy), dtype=np.uint32)
    c = 0
    for j in spy:
        for i in spx:
            sample_indices[c] = i + M*(j-1)
            c += 1
        
    return np.concatenate(([1], sample_indices)) - 1

def NLM(y1, y2, sigma):
    M, N = y1.shape

    if sigma < 5:
        h = 0.9*sigma
        psz = 3
    elif sigma < 15:
        h = 0.85*sigma
        psz = 3
    elif sigma < 25:
        g = 0.75*sigma
        psz = 5
    elif sigma < 35:
        g = 0.65*sigma
        psz = 5
    else:
        h = 0.55*sigma
        psz = 5

    R = 10
    wsz = 2*R+1
    pRad = int((psz-1)/2)

    images = []
    images.append(
        np.pad(y1, (R+pRad, R+pRad), 'symmetric'))
    images.append(
        np.pad(y2, (R+pRad, R+pRad), 'symmetric'))
    zt = [np.empty((M, N)), np.empty((M, N))]
    for k, img in enumerate(images):
        for i in range(M):
            for j in range(N):
                loc_p = [i+R+pRad, j+R+pRad]
                yp = img[loc_p[0]-R:loc_p[0]+R+1, loc_p[1]-R:loc_p[1]+R+1]

                row_min = loc_p[0] - R - pRad
                row_max = loc_p[0] + R + pRad
                col_min = loc_p[1] - R - pRad
                col_max = loc_p[1] + R + pRad
    
                z = img[row_min:row_max+1, col_min:col_max+1]
                Z = im2col(z.T, [psz, psz])

                Zc = Z[:, int((wsz ** 2 - 1)/2)]
                Zc = np.tile(Zc, [wsz**2, 1]).T
                d2 = np.mean((Zc - Z) ** 2, axis=0)
                Ker = np.exp(-np.maximum(d2-2*sigma**2, 0)/h**2)
                Ker = Ker / np.sum(Ker)
                zt[k][i, j] = np.dot(Ker, yp.T.reshape(wsz**2))

    zt1 = zt[0]
    zt1[zt1>255] = 255
    zt1[zt1<0] = 0
    zt2 = zt[1]
    zt2[zt2>255] = 255
    zt2[zt2<0] = 0
    return zt1, zt2


def GLIDE(y, z, sigma):
    M, N = z.shape
    sample_dist = 10
    m0 = 5
    k0 = 0
    ms = 5
    ks = 0.01
    m_max = 200
    k_max = 0.5

    # Monte-Carlo
    eps = 1
    #a = eps*np.random.random_sample([M*N, 1])
    a = np.empty([M*N, 1])
    with open('a.txt') as f:
        for i, line in enumerate(f):
            a[i] = float(line.strip())
    y2 = y + a.reshape(M, N)

    # NLM Prefiltering
    zt, zt2 = NLM(y, y2, sigma)
    sample_indices = Sampling(M, N, sample_dist)

    # Adapted smoothing param h
    h = Adapted_h(sigma)

    ztt = np.empty([2, M, N])
    ztt[0] = zt
    ztt[1] = zt2

    V12 = np.empty([2, M*N, m_max])
    lambda12 = np.empty([2, m_max])
    for k in range(2):
        zz = ztt[k].squeeze()
        # Nystroem
        phi, Pi = Nyst(zz, h, sample_indices)
        # Sinkhorn
        W_A, W_AB = Sink(phi, Pi)
        # Orthogonalising
        V12[k, :, :], lambda12[k, :] = Orth(W_A, W_AB, m_max)
        # Permutation
        V12[k, :, :] = Perm(V12[k].squeeze(), sample_indices)

    lambda1 = lambda12[0].squeeze()
    lambda2 = lambda12[1].squeeze()
    V1 = V12[0].squeeze()
    V2 = V12[1].squeeze()

    # MSE minimisation
    bd1 = np.dot(V1.T, y.T.reshape(M*N))
    bd2 = np.dot(V2.T, y2.T.reshape(M*N))
    Mp = range(m0, m_max, ms)
    Kp = [x / 100.0 for x in range(int(k0*100), int(k_max*100), int(ks*100))]
    n = M*N

    MSE_est = np.zeros([len(Mp), len(Kp)])
    for i, mp in enumerate(Mp):
        for j, k in enumerate(Kp):
            # Diffusions
            lambdaK1 = lambda1[0:mp].T ** k
            lambdaK2 = lambda2[0:mp].T ** k
            zh1 = np.dot(V1[:, 0:mp], lambdaK1 * bd1[0:mp]) 
            zh2 = np.dot(V2[:, 0:mp], lambdaK2 * bd2[0:mp]) 
            div = np.dot(np.dot((1./(n*eps)), a.T), (zh2-zh1))

            # Sure MSE
            MSE_est[i, j] = (1./n) * np.sum(y[:] ** 2) + (1./n) * \
                np.sum((lambdaK1 ** 2 - 2*lambdaK1) * bd1[0:mp] ** 2) + \
                2*(sigma**2) * div - sigma ** 2

    len_m = len(Mp)
    len_k = len(Kp)
    MSE_estim, I = np.min(MSE_est[:])
    kh = k0+ks*np.floor(I/len_m)
    mh = m0+ms*(np.mod(I, len_m))

    zh = np.dot(
        V1[:, 1:mh],
        (lambda1[0:mh].T ** kh) * np.dot(V[:, 0:mh].T, y[:]))
    zh = zh.reshape(M, N)
    for i, row in enumerate(zh):
        for j, elem in enumerate(row):
            if elem > 255:
                zh[i, j] = 255
            elif elem < 0:
                zh[i, j] = 0

    return zh, zt


if __name__ == '__main__':
    z = misc.imread('flower.jpg').astype(np.double)

    # Noising
    sigma = 50
    #y = z + np.dot(
    #    np.random.random_sample(z.shape),
    #    sigma)
    y = misc.imread('flower_noisy.jpg').astype(np.double)

    zh, zt = GLIDE(y, z, sigma)

    # Display
    plt.figure(1)
    plt.imshow(y, cmap='gray')

    plt.figure(2)
    plt.imshow(zt, cmap='gray')

    plt.figure(3)
    plt.imshow(zh, cmap='gray')

    plt.show()


from utils import im2col, matlab_style_gauss2D
import numpy as np


def affinity(y, sample_indices):
    M, N = y.shape[:2]
    h = 5.
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
    return AB 

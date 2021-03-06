from utils import im2col, matlab_style_gauss2D, num2xy
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


def NLM_affinity(y, sample_indices):
    start = time.time()
    M, N = y.shape[:2]
    h = 3.
    ksz = 7
    kRad = int((ksz - 1)/2)

    img = np.pad(y, (kRad, kRad), 'symmetric')
    G = matlab_style_gauss2D((ksz,ksz), 1.2)
    G = G.reshape(ksz**2, 1)
    G = G / np.sum(G)

    Z = im2col(img.T, [ksz, ksz])
    Z = Z * np.repeat(G, M*N).reshape(ksz**2, M*N)

    AB = np.empty((len(sample_indices), M*N), dtype=np.float64)
    for i, idx in enumerate(sample_indices):
        ind_x, ind_y = num2xy(idx, M, N)
        loc_p = [ind_x+kRad, ind_y+kRad]
        yp = img[loc_p[0]-kRad:loc_p[0]+kRad+1, loc_p[1]-kRad:loc_p[1]+kRad+1]
        Zc = yp.T.reshape(ksz**2) * G.reshape(ksz**2)
        Zc = np.repeat(Zc, M*N).reshape(ksz**2, M*N)
        Ker = np.exp(-np.sum((Zc - Z) ** 2, axis=0) / h**2)
        AB[i, :] = Ker
    logger.info('Affinity function NLM done in {0}s'.format(time.time() - start))
    return AB

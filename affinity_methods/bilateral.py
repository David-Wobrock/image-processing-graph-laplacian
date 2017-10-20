import logging
import numpy as np
from utils import num2xy
import time

logger = logging.getLogger(__name__)


def bilateral_affinity(y, sample_indices):
    start = time.time()
    M, N = y.shape[:2]
    h_photo = 20.
    h_spatial = 20.

    all_coords = np.asarray(
        [[a, b] for a in range(M) for b in range(N)])
    all_values = np.asarray(y.reshape(M*N), dtype=np.float64)

    AB = np.empty((len(sample_indices), M*N), dtype=np.float64)
    for i, idx in enumerate(sample_indices):
        ind_x, ind_y = num2xy(idx, M, N)

        sample_pos = np.asarray((ind_x, ind_y))
        sample_value = y[ind_x, ind_y]
        photometric = np.exp(-(
            np.abs(sample_value-all_values)**2)/(h_photo**2))
        spatial = np.exp(-(
            np.linalg.norm(sample_pos-all_coords, axis=1)**2)/(h_spatial**2))
        AB[i, :] = photometric * spatial
    
    logger.info('Affinity function bilateral done in {0}s'.format(time.time() - start))
    return AB

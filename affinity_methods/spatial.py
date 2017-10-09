import logging
import numpy as np
from utils import num2xy
import time

logger = logging.getLogger(__name__)

def spatial_affinity(y, sample_indices):
    start = time.time()
    M, N = y.shape[:2]
    h = 5.

    all_coords = np.asarray(
        [[x,y] for x in range(M) for y in range(N)])

    AB = np.empty((len(sample_indices), M*N))
    for i in range(len(sample_indices)):
        ind_x, ind_y = num2xy(sample_indices[i], M, N)

        sample_pos = np.asarray([ind_x, ind_y])
        AB[i, :] = np.asarray(
            np.exp(-(
                np.linalg.norm(sample_pos-all_coords, axis=1)**2)/25.))
    
    logger.info('Affinity function spatial done in {0}s'.format(time.time() - start))
    return AB

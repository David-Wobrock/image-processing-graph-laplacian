import logging
import numpy as np
from utils import num2xy
import time

logger = logging.getLogger(__name__)

def spatial_affinity(y, sample_indices):
    start = time.time()
    M, N = y.shape[:2]
    h = 25.

    all_coords = np.asarray(
        [[a, b] for a in range(M) for b in range(N)])

    AB = np.empty((len(sample_indices), M*N))
    for i, idx in enumerate(sample_indices):
        sample_pos = np.asarray(num2xy(idx, M, N))

        AB[i, :] = np.exp(-(
            np.linalg.norm(sample_pos-all_coords, axis=1)**2)/(h**2))

    logger.info('Affinity function spatial done in {0}s'.format(time.time() - start))
    return AB

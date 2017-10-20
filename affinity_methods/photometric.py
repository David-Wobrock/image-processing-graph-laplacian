import logging
import numpy as np
from utils import num2xy
import time

logger = logging.getLogger(__name__)


def photometric_affinity(y, sample_indices):
    start = time.time()
    M, N = y.shape[:2]
    h = 3.

    all_values = np.asarray(y.reshape(M*N), dtype=np.float64)

    AB = np.empty((len(sample_indices), M*N), dtype=np.float64)
    for i, idx in enumerate(sample_indices):
        ind_x, ind_y = num2xy(idx, M, N)

        sample_value = y[ind_x, ind_y]
        AB[i, :] = np.exp(-(
            np.abs(sample_value-all_values)**2)/(h**2))

    logger.info('Affinity function photometric done in {0}s'.format(time.time() - start))
    return AB

import logging
import numpy as np
import time

logger = logging.getLogger(__name__)


def random_sample(M, N, num_samples):
    start = time.time()
    sample_indices = set(np.random.randint(0, M*N, num_samples))
    while len(sample_indices) < num_samples:
        sample_indices.add(np.random.randint(0, int(M*N)))
    sample_indices = np.sort(list(sample_indices))

    logger.info('Random sampling done in {0}s'.format(time.time() - start))
    return sample_indices

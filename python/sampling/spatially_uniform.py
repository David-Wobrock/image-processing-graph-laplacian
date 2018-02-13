import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def spatially_uniform_sample(M, N, num_samples):
    start = time.time()
    sample_dist = int(np.sqrt((M*N) / num_samples))

    xy0 = int(sample_dist / 2)
    spx = range(xy0, M-1, sample_dist)
    spy = range(xy0, N-1, sample_dist)
    sample_indices = np.empty(len(spx)*len(spy), dtype=np.uint32)
    c = 0
    for i in spx:
        for j in spy:
            sample_indices[c] = N*i + j
            c += 1
        
    logger.info('Spatially uniform sampling (with distance {0}) done in {1}s'.format(sample_dist, time.time() - start))
    return sample_indices

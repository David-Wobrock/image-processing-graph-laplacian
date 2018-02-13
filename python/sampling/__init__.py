from .random import random_sample
from .spatially_uniform import spatially_uniform_sample

RANDOM, SPATIALLY_UNIFORM = 'random', 'spatially_uniform'

methods = {
    RANDOM: random_sample,
    SPATIALLY_UNIFORM: spatially_uniform_sample
}

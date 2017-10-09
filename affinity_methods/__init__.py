from .NLM import NLM_affinity
from .bilateral import bilateral_affinity
from .spatial import spatial_affinity
from .photometric import photometric_affinity

SPATIAL, PHOTOMETRIC, BILATERAL, NLM = 'spatial', 'photometric', 'bilateral', 'NLM'

methods = {
    SPATIAL: spatial_affinity,
    NLM: NLM_affinity,
    BILATERAL: bilateral_affinity,
    PHOTOMETRIC: photometric_affinity
}

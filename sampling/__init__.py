def get_sample_function(code):
    if code == 'random':
        from .random import sample
    elif code == 'spatially_uniform':
        from .spatially_uniform import sample
    else:
        raise Exception('Unknown sample function code')
    return sample

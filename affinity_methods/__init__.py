def get_affinity_function(code):
    if code == 'NLM':
        from .NLM import affinity
    elif code == 'bilateral':
        from .bilateral import affinity
    else:
        raise Exception('Unknown sample function code')
    return affinity

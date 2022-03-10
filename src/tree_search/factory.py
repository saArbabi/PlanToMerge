import copy
def safe_deepcopy_env(obj):
    """
        Perform a deep copy of an environment.
    """
    print('i die here')
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        if k not in ['env_initializor']:
            setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result

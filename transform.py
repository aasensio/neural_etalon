import torch
import numpy as np

def logit(x):
    """
    Logit function
    
    Parameters
    ----------
    x : float
        Any array

    Returns
    -------
    logit : float
        Logit transform of the input
    """
    if isinstance(x, torch.Tensor):
        return torch.log(x / (1.0 - x))
    else:
        return np.log(x / (1.0 - x))


def inv_logit(x):
    """
    Inverse logit function
    
    Parameters
    ----------
    x : float
        Any array

    Returns
    -------
    inv_logit : float
        Inverse logit transform of the input
    """
    if isinstance(x, torch.Tensor):
        return 1.0 / (1.0 + torch.exp(-x))
    else:
        return 1.0 / (1.0 + np.exp(-x))

def physical_to_transformed(x, lower, upper):
    """
    Transform from physical parameters to unconstrained physical parameters
    
    Parameters
    ----------
    x : float
        Any array
    lower : float
        Lower limit of the parameter
    upper : float
        Upper limit of the parameter

    Returns
    -------
    out : float
        Transformed parameters
    """
    return logit( (x-lower) / (upper - lower))

def transformed_to_physical(x, lower, upper):
    """
    Transform from unconstrained physical parameters to physical parameters
    
    Parameters
    ----------
    x : float
        Any array
    lower : float
        Lower limit of the parameter
    upper : float
        Upper limit of the parameter

    Returns
    -------
    out : float
        Transformed parameters
    """
    return lower + (upper - lower) * inv_logit(x)
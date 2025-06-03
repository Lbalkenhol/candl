"""
Handles imports and deals with optional packages such as JAX.
You can manually set ``USE_JAX = False`` to force numpy/scipy to be used instead of JAX even if it is installed (may be helpful for debugging).
"""

# --------------------------------------#
# NECESSARY
# --------------------------------------#

import os

import inspect

from functools import partial

import importlib

import yaml
import logging

import numpy as np
import scipy as sp

from copy import deepcopy

import sigfig
from textwrap import wrap
from tqdm.notebook import tqdm

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# --------------------------------------#
# OPTIONAL
# --------------------------------------#

# Global flag that can override use of JAX even if it is installed
# Useful for debugging to print out numpy arrays
USE_JAX = True

# Optional JAX import
JAX_IMPORT_SUCCEEDED = False
try:
    if USE_JAX:
        import jax.numpy as jnp
        import jax.scipy as jsp
        from jax import jit, jacfwd, custom_jvp
        from jax import config as jax_config

        # Unify syntax for setting array elements
        def jax_optional_set_element(arr, ix, el):
            return arr.at[ix].set(el)

        # Need JAX to run with 64bit for dynamic range in covmats
        try:
            jax_config.update("jax_enable_x64", True)
        except:
            raise Exception(
                "candl: could not configure JAX to run in 64 bit mode - this will likely lead to wrong results!"
            )
        JAX_IMPORT_SUCCEEDED = True
    else:
        raise ImportError()
except:
    import numpy as jnp
    import scipy as jsp

    # Unify syntax for setting array elements
    def jax_optional_set_element(arr, ix, el):
        arr[ix] = el
        return arr

    # define jit decorator to do nothing
    def jit(func, **kwargs):
        return func

    # define custom_jvp decorator to do nothing
    def custom_jvp(func, **kwargs):
        func.defjvp = lambda *args: None
        return func


# Optional pycapse import
try:
    import pycapse as pc
    import json
except:
    pass

# Optional CAMB import
try:
    import camb
except:
    pass

# Optional CLASS import
try:
    import classy
except:
    pass

# Optional Cobaya import
try:
    from cobaya.theory import Theory as cobaya_theory_Theory
    from cobaya.likelihood import Likelihood as cobaya_likelihood_Likelihood
except:

    # Empty class definitions in case Cobaya is not installed
    class cobaya_theory_Theory:
        pass

    class cobaya_likelihood_Likelihood:
        pass


# Optional CosmoPower import
try:
    import cosmopower as cp
except:
    pass

# Optional CosmoPower-JAX import
try:
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
except:
    pass

# Optional clipy import
try:
    import clipy
except:
    pass

# Optional candl_data import
try:
    import candl_data
except:
    print(
        "Failed to import candl_data, if you want to use the candl data sets follow the instructions on: https://github.com/Lbalkenhol/candl_data."
    )

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from os import listdir
import yaml
import candl
import numpy as np
from copy import deepcopy
import sigfig

try:
    import candl_data
except:
    print(
        "Failed to import candl_data, some tests may not find their respective likelihoods."
    )

# --------------------------------------#
# TEST FUNCTIONS
# --------------------------------------#


def run_all_tests():
    """
    Runs all tests in the candl/tests directory.
    See run_test() for details.
    """

    # Grab all test yaml files
    all_test_yaml_files = [
        f for f in listdir(f"{candl.__path__[0]}/tests/") if f.endswith(".yaml")
    ]

    # Loop over all tests
    for test_file in all_test_yaml_files:
        run_test(f"{candl.__path__[0]}/tests/{test_file}")


def run_test(test_file):
    """
    Runs a test based on the provided test file.
    Uses the relative difference between the official likelihood and the candl implementation.
    Relative tolerance is 1e-3.
    Prints results.


    Parameters
    ------------
        test_file : str
            The path to the test .yaml file.

    Returns
    ------------
        None
    """

    # Load info from yaml file
    with open(test_file, "r") as f:
        test_dict = yaml.load(f, Loader=yaml.loader.SafeLoader)

    # Initialise likelihood
    try:
        data_set_str = test_dict["data_set_file"]
        if not ".yaml" in data_set_str:
            data_set_str = eval(data_set_str)
        if test_dict["lensing"]:
            candl_like = candl.LensLike(data_set_str, feedback=False)
        else:
            candl_like = candl.Like(data_set_str, feedback=False)
    except:
        print(
            f"!!! -> Failed to initialise likelihood for {test_file} pointing to {data_set_str}"
        )
        return

    # Load test spectrum, juggle into dictionary, grab test parameter values
    test_spec = np.loadtxt(f"{candl.__path__[0]}/{test_dict['test_spectrum']}")
    pars_for_like = deepcopy(test_dict["param_values"])
    pars_for_like["Dl"] = {}
    test_spec_save_order = ["ell", "TT", "TE", "EE", "BB", "pp", "kk"]
    for i, spec in enumerate(test_spec_save_order):
        pars_for_like["Dl"][spec] = test_spec[:, i]

    # Evaluate likelihood
    rel_diff = 1
    if "test_chisq" in test_dict:
        # Test chisq
        candl_chisq = candl_like.chi_square(pars_for_like)
        rel_diff = (candl_chisq - test_dict["test_chisq"]) / test_dict["test_chisq"]
    elif "test_logl" in test_dict:
        # Test logl
        candl_logl = candl_like.log_like(pars_for_like)
        # Remove prior contribution if desired
        if "remove_prior" in test_dict:
            if test_dict["remove_prior"]:
                candl_logl += candl_like.prior_logl(pars_for_like)
        rel_diff = (candl_logl - test_dict["test_logl"]) / test_dict["test_logl"]

    # Print results
    if abs(rel_diff) < 1e-3:
        print(f"Test passed for {candl_like.name}:")
        print(
            f"  (relative difference = {sigfig.round(float(rel_diff), sigfigs = 2)})!"
        )
    else:
        print(f"!!! -> Test failed for {candl_like.name}:")
        print(
            f"  !!! -> (relative difference = {sigfig.round(float(rel_diff), sigfigs = 2)})!"
        )

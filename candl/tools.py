"""
Various tools that interact with the likelihood and its products.

Overview:
--------------

Fisher matrix calculation:

* :func:`get_fisher_matrix`

Simple minimiser:

* :func:`newton_raphson_minimiser`
* :func:`newton_raphson_minimiser_bdp`
* :func:`add_uniform_scatter`

Mock data generation:

* :func:`generate_mock_data`

Multi-frequency analysis:

* :func:`get_foreground_contributions`
* :func:`make_MV_combination`
* :func:`test_statistic_MV_consistency`
* :func:`make_frequency_conditional`
* :func:`conditional_prediction`
* :func:`test_statistic_conditional`
* :func:`make_difference_spectra`
* :func:`test_statistic_difference`

Bundling likelihood and theory code together:

* :func:`get_params_to_logl_func`
* :func:`get_params_to_chi_square_func`

Helpers to transform model spectra:

* :func:`pars_to_model_specs`
* :func:`pars_to_model_specs_partial_transformation`
* :func:`undo_transformations`
"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
import candl
import candl.transformations.abstract_base
import candl.io

# --------------------------------------#
# FISHER MATRIX
# --------------------------------------#


def get_fisher_matrix(
    pars_to_theory_specs,
    like,
    pars,
    par_order=None,
    return_par_order=False,
    return_par_cov=True,
):
    """
    Calculate the Parameter covariance matrix based on the Fisher matrix using a differentiable theory code and the covariance of the data set.
    Applies priors on nuisance parameters.

    Parameters
    --------------
    pars_to_theory_specs : func
        Differentiable function moving dictionary of parameters to CMB spectra.
    like : candl.Like
        Likelihood
    pars : dict
        Dictionary of parameter values.
    par_order : list (optional)
        Order of parameters requested in returned matrix.
    return_par_order : bool (optional)
        return a list of parameters to indicate the order in the matrix.
    return_par_cov : bool (optional)
        Return the parameter covariance matrix, setting to False will return the Fisher matrix (i.e. not inverted).

    Returns
    --------------
    array (float) :
        Parameter covariance matrix.
    list (str) :
        List of parameter names.
    """

    # Define parameter order
    if par_order is None:
        par_order = list(pars.keys())

    # Get derivate function
    model_specs = lambda p: pars_to_model_specs(like, p, pars_to_theory_specs)[1]
    model_specs_deriv = jacfwd(
        model_specs
    )  # Forward autodiff should be better here (going from few cosmological parameters to many band power bins)

    # Calculate derivatives
    Dl_derivs_dict = model_specs_deriv(pars)
    Dl_derivs_arr = []
    for j, p in enumerate(par_order):
        Dl_derivs_arr.append(Dl_derivs_dict[p])
    Dl_derivs_arr = np.asarray(Dl_derivs_arr)

    # Calculate Fisher matrix
    fisher_mat = np.dot(
        Dl_derivs_arr, np.dot(np.linalg.inv(like.covariance), Dl_derivs_arr.T)
    )

    # Apply priors
    for prior in like.priors:
        prior_cov_inv = np.linalg.inv(prior.prior_covariance)
        for i, p1 in enumerate(prior.par_names):
            for j, p2 in enumerate(prior.par_names):
                try:
                    fisher_mat[
                        par_order.index(p1), par_order.index(p2)
                    ] += prior_cov_inv[i, j]
                except:
                    pass

    # Invert and return
    if return_par_cov:
        par_cov = np.linalg.inv(fisher_mat)

        if return_par_order:
            return par_cov, par_order
        return par_cov
    else:
        if return_par_order:
            return fisher_mat, par_order
        return fisher_mat


# --------------------------------------#
# MINIMISATION
# --------------------------------------#


def newton_raphson_minimiser(
    like_deriv,
    like_hess,
    starting_pars,
    pars_for_min,
    N_newton=10,
    step_size=0.2,
    show_progress=False,
):
    """
    Take Newton-Raphson steps towards the minimum.

    Warning
    --------------
    This is not a stress-tested minimiser, but rather a simple implementation of the algorithm. Use at your own risk.

    Parameters
    --------------
    like_deriv : func
        Derivative of the likelihood.
    like_hess : func
        Hessian of the likelihood.
    starting_pars : dict
        Start parameters.
    pars_for_min : list
        List of parameters to minimise.
    N_newton : int
        Number of steps to take.
    step_size : float
        Scaling for step size.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    --------------
    list :
        List of all evaluation points
    list :
        List of parameter covariance matrix at all evaluation points.
    """

    # Progress bar
    if show_progress:
        pbar = tqdm(total=N_newton)

    # Prepare for Newton steps
    eval_points = [starting_pars]
    eval_par_cov = []
    has_failed = False
    for i_newton in range(N_newton + 1):
        # Calculate Hessian
        this_hess_dict = like_hess(eval_points[-1])

        # Turn into a nice array
        this_hess_arr = np.zeros((len(pars_for_min), len(pars_for_min)))
        for i, p1 in enumerate(pars_for_min):
            for j, p2 in enumerate(pars_for_min):
                this_hess_arr[i, j] = this_hess_dict[p1][p2]

        # Get the inverse
        this_par_cov = np.linalg.inv(-this_hess_arr)
        eval_par_cov.append(this_par_cov)

        # Exit here if desired number of steps have been reached
        if i_newton == N_newton:
            break

        # Juggle dict to arr
        par_dict = like_deriv(eval_points[i_newton])
        par_arr = np.zeros(len(pars_for_min))
        for j, p in enumerate(pars_for_min):
            par_arr[j] = par_dict[p]

        # Calculate Newton step
        step = jnp.dot(this_par_cov, par_arr)

        # Back into dict form for next step
        new_pars = deepcopy(eval_points[i_newton])
        for j, p in enumerate(pars_for_min):
            if step[j] is jnp.nan:  # stop if nan is encountered
                has_failed = True
                break
            new_pars[p] += step_size * step[j]

        # If step failed break out of loop
        if has_failed:
            break

        # Add any fixed parameter values
        for p in list(starting_pars.keys()):
            if not p in pars_for_min:
                new_pars[p] = starting_pars[p]

        eval_points.append(new_pars)

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    # If minimiser has failed let the user know
    if has_failed:
        raise Exception(
            f"Minimiser failed (encountered nan in proposed step). Try a better initial point and smaller step size for better stability."
        )

    return eval_points, eval_par_cov


def newton_raphson_minimiser_bdp(
    like_deriv,
    like_hess,
    starting_pars,
    pars_for_min,
    bdp,
    N_newton=10,
    step_size=1,
    show_progress=False,
):
    """
    Same as newton_raphson_minimiser() but allows for the input of data band powers.
    This can be useful if dealing with the same likelihood but lots of mock data realisations to avoid the jitting
    penalty of new likelihoods for each realisation.

    Warning
    --------------
    This is not a stress-tested minimiser, but rather a simple implementation of the algorithm. Use at your own risk.
    """

    # Progress bar
    if show_progress:
        pbar = tqdm(total=N_newton)

    # Prepare for Newton steps
    eval_points = [starting_pars]
    eval_par_cov = []
    has_failed = False
    for i_newton in range(N_newton + 1):
        # Calculate Hessian
        this_hess_dict = like_hess(eval_points[-1], bdp)

        # Turn into a nice array
        this_hess_arr = jnp.zeros((len(pars_for_min), len(pars_for_min)))
        for i, p1 in enumerate(pars_for_min):
            for j, p2 in enumerate(pars_for_min):
                this_hess_arr = jax_optional_set_element(
                    this_hess_arr, (i, j), this_hess_dict[p1][p2]
                )
                # this_hess_arr[i, j] = this_hess_dict[p1][p2]

        # Get the inverse
        this_par_cov = jnp.linalg.inv(-this_hess_arr)
        eval_par_cov.append(this_par_cov)

        # Exit here if desired number of steps have been reached
        if i_newton == N_newton:
            break

        # Juggle dict to arr
        par_dict = like_deriv(eval_points[i_newton], bdp)
        par_arr = jnp.zeros(len(pars_for_min))
        for j, p in enumerate(pars_for_min):
            par_arr = jax_optional_set_element(par_arr, j, par_dict[p])
            # par_arr[j] = par_dict[p]

        # Calculate Newton step
        step = jnp.dot(this_par_cov, par_arr)

        # Back into dict form for next step
        new_pars = deepcopy(eval_points[i_newton])
        for j, p in enumerate(pars_for_min):
            if step[j] is jnp.nan:  # stop if nan is encountered
                has_failed = True
                break
            new_pars[p] += step_size * step[j]

        # If step failed break out of loop
        if has_failed:
            break

        # Add any fixed parameter values
        for p in list(starting_pars.keys()):
            if not p in pars_for_min:
                new_pars[p] = starting_pars[p]

        eval_points.append(new_pars)

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    # If minimiser has failed let the user know
    if has_failed:
        raise Exception(
            f"Minimiser failed (encountered nan in proposed step). Try a better initial point and smaller step size for better stability."
        )

    return eval_points, eval_par_cov


def add_uniform_scatter(start_params, box_width, par_errs, seed=None):
    """
    Add uniform scatter to a point, helpful to offset starting points for minimisers.

    Parameters
    --------------
    start_params : dict
        Central parameter values.
    box_width : float
        Width of the sampled region in units of par_errs.
    par_errs : dict
        Parameter errors.
    seed : int
        RNG seed.

    Returns
    --------------
    dict :
        Offset parameters.
    dict :
        Parameter values delimiting the sampled region.
    """

    # Set RNG seed
    rng = np.random.default_rng(seed)

    # Add scatter and retain size of sampled region
    start_box = {}
    for par in par_errs:
        scatter_width = box_width * par_errs[par]
        par_scatter_range = [
            start_params[par] - scatter_width / 2,
            start_params[par] + scatter_width / 2,
        ]
        start_params[par] = rng.uniform(par_scatter_range[0], par_scatter_range[1])
        start_box[par] = par_scatter_range
    return start_params, start_box


# --------------------------------------#
# MOCK DATA GENERATION
# --------------------------------------#


def generate_mock_data(pars, pars_to_theory_specs, like, N_real, seed=None):
    """
    Generate mock band powers based on the covariance matrix.

    Parameters
    --------------
    pars : dict
        Fiducial parameter values.
    pars_to_theory_specs: func
        function moving dictionary of cosmological parameters to dictionary of CMB Dls
    like : Like
        Likelihood to take care of transformation and binning.
    N_real : int
        Number of realisations requested
    seed : int
        RNG seed.

    Returns
    --------------
    array :
        Binned model Dls (CMB and all transformation of the likelihood)
    list :
        List of mock band power realisations.
    """

    # Grab model spectrum
    unbinned_modified_theory_Dls, binned_modified_theory_Dls = pars_to_model_specs(
        like, pars, pars_to_theory_specs
    )

    # Add scatter
    rng = np.random.default_rng(seed)
    mock_bdp = []
    for i in range(N_real):
        this_scatter = jnp.dot(
            like.covariance_chol_dec,
            rng.standard_normal(np.shape(binned_modified_theory_Dls)[0]),
        )
        this_mock_bdp = binned_modified_theory_Dls + this_scatter
        mock_bdp.append(this_mock_bdp)

    return binned_modified_theory_Dls, mock_bdp


# --------------------------------------#
# FOREGROUND CONTRIBUTIONS
# --------------------------------------#


def get_foreground_contributions(
    like, params, return_type="arr", remove_zeros=True, binning="unbinned"
):
    """
    Return the foreground contributions.
    This method is intended to be accessed by the user for visualisation purposes to show the contributions of
    the different foreground components.

    Parameters
    --------------
    params : dict
        Nuisance parameter values.
    return_type : str
        One of ["arr", "dict", "dict by spec"]. Determines the output format.
    remove_zeros : bool
        Remove foreground components that are zero for all ells.
    binning : str
        One of ["unbinned", "binned", "both"]. Whether to return the foreground components for individual ells,
        binned, or both.

    Returns
    --------------
    Depending on return_type:
    "arr" : array, float
        A (N_ell_bins_theory*N_spectra_total or N_bins_total x N_foregrounds) array containing all the foreground
        contributions.
    "dict" : dict
        A dictionary with keys corresponding to the different foreground components and the values are
        (N_ell_bins_theory*N_spectra_total or N_bins_total) long vectors containing each contribution.
    "dict by spec" : dict
        A dictionary with keys corresponding to different spectra, each holding another dictionary with
        foreground names as keys and contributions as values.
    if binning is "both" then the output is wrapped in a dictionary containing "binned" and "unbinned" contributions
    at the lowest level.
    """

    # Check return_type argument
    if not return_type in ["arr", "dict", "dict by spec"]:
        raise Exception(
            f"Did not understand return_type '{return_type}'; must be one of (arr, dict, dict by spec). Returning as array"
        )
        return_type = "arr"

    # Check binning argument
    if not binning in ["unbinned", "binned", "both"]:
        raise Exception(
            f"Did not understand binning '{binning}'; must be one of (unbinned, binned, both). Returning unbinned"
        )
        binning = "unbinned"

    # Get foreground contributions
    foreground_contributions = []
    foreground_names = []
    for transformation in like.data_model:
        # Check if the transformation is a foreground - if so grab contribution and short name
        if isinstance(transformation, candl.transformations.abstract_base.Foreground):
            foreground_contributions.append(transformation.output(params))
            foreground_names.append(transformation.descriptor)

    foreground_contributions = jnp.asarray(foreground_contributions)

    # Return requested type
    if return_type == "dict":
        # Return as dictionary {label:contribution}
        if binning == "unbinned":
            return {
                foreground_names[i_fg]: foreground_contributions[i_fg, :]
                for i_fg in range(len(foreground_names))
            }
        elif binning == "binned":
            return {
                foreground_names[i_fg]: like.bin_model_specs(
                    foreground_contributions[i_fg, :]
                )
                for i_fg in range(len(foreground_names))
            }
        elif binning == "both":
            return {
                foreground_names[i_fg]: {
                    "unbinned": like.bin_model_specs(foreground_contributions[i_fg, :]),
                    "binned": like.bin_model_specs(foreground_contributions[i_fg, :]),
                }
                for i_fg in range(len(foreground_names))
            }
        return {
            foreground_names[i_fg]: foreground_contributions[i_fg, :]
            for i_fg in range(len(foreground_names))
        }

    elif return_type == "dict by spec":
        # Return as individual dictionaries for spectra {spec: {label:contribution}}
        dict_by_spec = {}
        for i_spec, spec in enumerate(like.spec_order):
            dict_by_spec[spec] = {}
            for i_fg, fg_label in enumerate(foreground_names):
                fg_arr = foreground_contributions[
                    i_fg,
                    like.N_ell_bins_theory
                    * i_spec : like.N_ell_bins_theory
                    * (i_spec + 1),
                ]
                if all(fg_arr == 0) and remove_zeros:
                    pass
                else:
                    if binning == "unbinned":
                        dict_by_spec[spec][fg_label] = fg_arr
                    else:
                        binned_fg_arr = like.bin_model_specs(
                            foreground_contributions[i_fg, :]
                        )[like.bins_start_ix[i_spec] : like.bins_stop_ix[i_spec]]
                        if binning == "binned":
                            dict_by_spec[spec][fg_label] = binned_fg_arr
                        elif binning == "both":
                            dict_by_spec[spec][fg_label] = {
                                "unbinned": fg_arr,
                                "binned": binned_fg_arr,
                            }
        return dict_by_spec

    # Return as array
    if binning == "unbinned":
        return foreground_contributions, foreground_names
    else:
        binned_fg_arr = []
        for i_fg in range(len(foreground_names)):
            binned_fg_arr.append(
                like.bin_model_specs(foreground_contributions[i_fg, :])
            )
        binned_fg_arr = jnp.asarray(binned_fg_arr)
        if binning == "binned":
            return binned_fg_arr, foreground_names
        elif binning == "both":
            return {
                "unbinned": foreground_contributions,
                "binned": binned_fg_arr,
            }, foreground_names


# --------------------------------------#
# MINIMUM-VARIANCE COMBINATION
# --------------------------------------#


def make_MV_combination(like, data_CMB_only, design_matrix, blinded=False):
    """
    Combine multifrequency band powers into MV estimate.
    See Appendix C4 here https://arxiv.org/pdf/1507.02704.pdf for details.

    Parameters
    --------------
    like : candl.Like
        The likelihood.
    data_CMB_only : array (float)
        The data bandpowers with all transformations undone.
    design_matrix : array (int)
        The design matrix.
    blinded : bool
        Whether operations are carried out in blinded or unblinded space (applies additional factor of the blinding function to the covariance).

    Returns
    --------------
    dict :
        "MV spec" : MV spectrum (array (float))
        "MV cov" : MV covariance (array (float))
        "MV ell" : effetive multipoles (array (float))
        "mix mat" : mixing matrix (array (float))
    """

    # Assemble MV band powers and covariance
    if blinded:
        cov_inv = np.linalg.inv(
            like.covariance * np.outer(like._blinding_function, like._blinding_function)
        )  # adjust covariance for blinding on the fly
    else:
        cov_inv = np.linalg.inv(like.covariance)

    weighting_fac = np.matmul(design_matrix.T, np.matmul(cov_inv, design_matrix))
    bdp_fac = np.matmul(design_matrix.T, np.matmul(cov_inv, data_CMB_only))

    MV_spec = np.matmul(np.linalg.inv(weighting_fac), bdp_fac)
    MV_cov = np.linalg.inv(
        np.matmul(design_matrix.T, np.matmul(cov_inv, design_matrix))
    )

    ell_eff_fac = np.matmul(design_matrix.T, np.matmul(cov_inv, like.effective_ells))
    MV_ell = np.matmul(np.linalg.inv(weighting_fac), ell_eff_fac)

    mixing_mat = np.linalg.inv(weighting_fac) @ design_matrix.T @ cov_inv

    return {
        "MV spec": MV_spec,
        "MV cov": MV_cov,
        "MV ell": MV_ell,
        "mix mat": mixing_mat,
    }


def test_statistic_MV_consistency(
    like, MV_dict, data_CMB_only, design_matrix, blinded=False
):
    """
    Test consistency by comparing multifrequency band powers to MV combination

    Parameters
    --------------
    like : candl.Like
        The likelihood.
    MV_dict : dict
        Dictionary containing information about MV combination (intended is output from make_MV_combination).
    data_CMB_only : array (float)
        The data bandpowers with all transformations undone.
    design_matrix : array (int)
        The design matrix.
    blinded : bool
        Whether operations are carried out in blinded or unblinded space (applies additional factor of the blinding function to the covariance).

    Returns
    --------------
    float:
        chisq
    float:
        PTE
    """

    delta = (design_matrix @ MV_dict["MV spec"]) - data_CMB_only

    if blinded:
        chol_fac = np.linalg.solve(
            np.linalg.cholesky(
                like.covariance
                * np.outer(like._blinding_function, like._blinding_function)
            ),
            delta,
        )  # adjust cov for blinding function
    else:
        chol_fac = np.linalg.solve(like.covariance_chol_dec, delta)
    chisq = np.dot(chol_fac.T, chol_fac)

    ndof = like.N_bins_total - len(MV_dict["MV spec"])
    PTE = 1 - sp.stats.chi2.cdf(chisq, ndof)

    return chisq, PTE


# --------------------------------------#
# FREQUENCY CONDITIONAL
# --------------------------------------#


def make_frequency_conditional(spec_str, like, best_fit_model_binned, blinded=False):
    """
    Generate frequency conditional prediction.
    See Section 3.6.3 here https://arxiv.org/pdf/1907.12875.pdf for details.

    Parameters
    --------------
    spec_str : str
        String identifying the spectrum. Intended format is (type) (freq1)x(freq2), e.g. "TT 150x150"
    like : candl.Like
        The likelihood.
    best_fit_model_binned : array (float)
        The binned best fit spectrum.
    blinded : bool
        Whether to operate in blinded space or not.

    Returns
    --------------
    dict :
        "cond spec" : Conditional spectrum (array (float))
        "cond cov" : Conditional covariance (array (float))
    """

    fld, freq_combo = spec_str.split(" ")

    # Make data vectors and covariance of all multifrequency spectra of this type
    fld_N_bins_list = [
        like.N_bins[i] for i in range(len(like.spec_order)) if like.spec_types[i] == fld
    ]
    fld_N_bins = sum(fld_N_bins_list)
    fld_start_ix, fld_stop_ix = candl.get_start_stop_ix(fld_N_bins_list)
    fld_spec_order = [
        like.spec_order[i]
        for i in range(len(like.spec_order))
        if like.spec_types[i] == fld
    ]

    fld_data_vec = np.zeros(fld_N_bins)
    fld_bf_vec = np.zeros(fld_N_bins)
    fld_cov = np.zeros((fld_N_bins, fld_N_bins))

    spec_msk = np.zeros(fld_N_bins)

    # Loop over all spectra and populate data structures
    for i, spec_1 in enumerate(fld_spec_order):
        full_vec_i = like.spec_order.index(spec_1)

        # data band powers
        if blinded:
            fld_data_vec[fld_start_ix[i] : fld_stop_ix[i]] = like.data_bandpowers[
                like.bins_start_ix[full_vec_i] : like.bins_stop_ix[full_vec_i]
            ]
        else:
            fld_data_vec[fld_start_ix[i] : fld_stop_ix[i]] = like._data_bandpowers[
                like.bins_start_ix[full_vec_i] : like.bins_stop_ix[full_vec_i]
            ]

        # model
        fld_bf_vec[fld_start_ix[i] : fld_stop_ix[i]] = best_fit_model_binned[
            like.bins_start_ix[full_vec_i] : like.bins_stop_ix[full_vec_i]
        ]

        if freq_combo == "x".join(like.spec_freqs[full_vec_i]):
            spec_msk[fld_start_ix[i] : fld_stop_ix[i]] = 1

        # covariance
        for j, spec_2 in enumerate(fld_spec_order):
            full_vec_j = like.spec_order.index(spec_2)

            this_cov_slice = like.covariance[
                like.bins_start_ix[full_vec_i] : like.bins_stop_ix[full_vec_i],
                like.bins_start_ix[full_vec_j] : like.bins_stop_ix[full_vec_j],
            ]

            if blinded:
                this_cov_slice *= np.outer(
                    like._blinding_function, like._blinding_function
                )[
                    like.bins_start_ix[full_vec_i] : like.bins_stop_ix[full_vec_i],
                    like.bins_start_ix[full_vec_j] : like.bins_stop_ix[full_vec_j],
                ]

            fld_cov[
                fld_start_ix[i] : fld_stop_ix[i], fld_start_ix[j] : fld_stop_ix[j]
            ] = this_cov_slice

    spec_msk = spec_msk == 1

    # Hand off to conditional function
    cond_spec, cond_cov = conditional_prediction(
        fld_data_vec, fld_bf_vec, fld_cov, spec_msk
    )

    return {"cond spec": cond_spec, "cond cov": cond_cov}


def conditional_prediction(x, x_bar, sigma, mask=None):
    """
    Linear algebra behind the conditional prediction.
    Code supplied by Karim Benabed.

    Parameters
    --------------
    x : array (float)
        The data vector.
    x_bar : array (float)
        The best-fit model.
    sigma : array (float)
        the covariance matrix
    mask : array (bool)
        Mask specifying which parts to excise in generating the conditional prediction.

    Returns
    --------------
    float:
        Conditional prediction
    float:
        Conditional covariance

    Notes
    --------------
    Original doc-string:
    This computes the conditional.
    x is the data vector, x_bar is the theoretical prediction.
    Both are 1D vectors and the ordering is whatever you want, provided that it's the same than the sigma covariance (a 2d array obviously).
    Mask is an optional 1D array of boolean, same size as x, to select a particular part of the data you want to ignore in the conditional implements eq. 353 of https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf.
    Returns, x_tilde and sigma_tilde.
    """

    if mask is None:
        mask = np.ones(len(x)) == 1
    xa = x_bar[mask]
    nmask = mask == False
    xb = x_bar[nmask]
    sigma_aa = sigma[mask, :][:, mask]
    sigma_bb = sigma[nmask, :][:, nmask]
    sigma_ab = sigma[mask, :][:, nmask]
    sigma_bb_inv = np.linalg.inv(sigma_bb)
    sigma_ab_DOT_sigma_bb_inv = np.dot(sigma_ab, sigma_bb_inv)
    x_tilde = xa + np.dot(sigma_ab_DOT_sigma_bb_inv, (x[nmask] - xb))
    sigma_tilde = sigma_aa - np.dot(sigma_ab_DOT_sigma_bb_inv, sigma_ab.T)

    return x_tilde, sigma_tilde


def test_statistic_conditional(spec_str, cond_dict, like, blinded=False):
    """
    Test conditional prediction against measured band powers.

    Parameters
    --------------
    spec_str : str
        String identifying the spectrum. Intended format is (type) (freq1)x(freq2), e.g. "TT 150x150"
    cond_dict : dict
        Dictionary containing information about the conditional prediction (intended is output from make_frequency_conditional).
    like : candl.Like
        The likelihood.
    blinded : bool
        Whether to operate on blinded band powers or not.

    Returns
    --------------
    float:
        chisq
    float:
        PTE
    """
    ix = like.spec_order.index(spec_str)
    if blinded:
        delta_cond = (
            cond_dict["cond spec"]
            - like.data_bandpowers[like.bins_start_ix[ix] : like.bins_stop_ix[ix]]
        )
    else:
        delta_cond = (
            cond_dict["cond spec"]
            - like._data_bandpowers[like.bins_start_ix[ix] : like.bins_stop_ix[ix]]
        )

    chisq = delta_cond.T @ np.linalg.inv(cond_dict["cond cov"]) @ delta_cond

    ndof = len(delta_cond)
    PTE = 1 - sp.stats.chi2.cdf(chisq, ndof)

    return chisq, PTE


# --------------------------------------#
# DIFFERENCE SPECTRA
# --------------------------------------#


def make_difference_spectra(spec_str_1, spec_str_2, data_CMB_only, like, blinded=False):
    """
    Generate frequency conditional prediction.
    See Section 3.6.3 here https://arxiv.org/pdf/1907.12875.pdf for details.

    Parameters
    --------------
    spec_str_1 : str
        String identifying the first spectrum. Intended format is (type) (freq1)x(freq2), e.g. "TT 150x150"
    spec_str_2 : str
        String identifying the second spectrum. Intended format is (type) (freq1)x(freq2), e.g. "TT 150x220"
    data_CMB_only : array (float)
        The data bandpowers with all transformations undone.
    like : candl.Like
        The likelihood.
    blinded : bool
        Whether to operate in blinded space.

    Returns
    --------------
    dict :
        "diff spec" : Difference spectrum (array (float))
        "diff cov" : Difference covariance (array (float))

    Warning
    --------------
    The code only works for equal length spectra at the moment.
    """

    # Check that valid difference was requested
    if spec_str_1[:2] != spec_str_1[:2] or spec_str_1 == spec_str_2:
        print(
            "Requested difference spectrum for different spectrum types or entered same spectrum twice!"
        )
        return

    # Grab indices
    i = like.spec_order.index(spec_str_1)
    j = like.spec_order.index(spec_str_2)

    # Grab spectra
    spec_1_fg_sub = data_CMB_only[like.bins_start_ix[i] : like.bins_stop_ix[i]]
    spec_2_fg_sub = data_CMB_only[like.bins_start_ix[j] : like.bins_stop_ix[j]]

    if len(spec_1_fg_sub) != len(spec_2_fg_sub):
        print("Requested spectra of different length, no design matrix supplied!")
        return

    diff = spec_1_fg_sub - spec_2_fg_sub

    # Grab cov
    cov_to_use = deepcopy(like.covariance)
    if blinded:
        cov_to_use *= np.outer(like._blinding_function, like._blinding_function)

    spec_1_cov = cov_to_use[
        like.bins_start_ix[i] : like.bins_stop_ix[i],
        like.bins_start_ix[i] : like.bins_stop_ix[i],
    ]
    spec_2_cov = cov_to_use[
        like.bins_start_ix[j] : like.bins_stop_ix[j],
        like.bins_start_ix[j] : like.bins_stop_ix[j],
    ]
    spec_12_cov = cov_to_use[
        like.bins_start_ix[i] : like.bins_stop_ix[i],
        like.bins_start_ix[j] : like.bins_stop_ix[j],
    ]
    spec_21_cov = cov_to_use[
        like.bins_start_ix[j] : like.bins_stop_ix[j],
        like.bins_start_ix[i] : like.bins_stop_ix[i],
    ]

    # Assemble covariance of the difference
    cov_block = np.block([[spec_1_cov, spec_12_cov], [spec_21_cov, spec_2_cov]])
    A_mat = np.block([[np.eye(len(spec_2_fg_sub)), -np.eye(len(spec_2_fg_sub))]])
    diff_cov = np.matmul(A_mat, np.matmul(cov_block, A_mat.T))

    return {"diff spec": diff, "diff cov": diff_cov}


def test_statistic_difference(diff_dict):
    """
    Test difference spectrum against zero.

    Parameters
    --------------
    diff_dict : dict
        Dictionary containing information about the difference spectrum (intended is output from make_difference_spectra).

    Returns
    --------------
    float:
        chisq
    float:
        PTE
    """

    chisq = (
        diff_dict["diff spec"].T
        @ np.linalg.inv(diff_dict["diff cov"])
        @ diff_dict["diff spec"]
    )
    ndof = len(diff_dict["diff spec"])
    PTE = 1 - sp.stats.chi2.cdf(chisq, ndof)

    return chisq, PTE


# --------------------------------------#
# BUNDLING LIKELIHOOD AND THEORY CODE TOGETHER
# --------------------------------------#


def get_params_to_logl_func(like, pars_to_theory_specs):
    """
    Thin wrapper bundling together likelihood and theory code to move straight from parameters to logl.

    Parameters
    --------------
    like: candl.Like
        Likelihood to be used.
    pars_to_theory_specs: func
        function moving dictionary of cosmological parameters to dictionary of CMB Dls

    Returns
    --------------
    func:
        Function that takes a dictionary of parameter values as input and returns the log likelihood
        (bundling theory code and likelihood together).

    """

    def like_with_theory(sampled_pars):
        # Calculate theory Dls
        Dls = pars_to_theory_specs(sampled_pars, like.ell_max, like.ell_min)

        # Pass to likelihood
        new_pars = sampled_pars
        new_pars["Dl"] = Dls
        logl = like.log_like(new_pars)

        return logl

    return like_with_theory


def get_params_to_chi_square_func(like, pars_to_theory_specs):
    """
    Thin wrapper bundling together likelihood and theory code to move straight from parameters to chi square.

    Parameters
    --------------
    like: candl.Like
        Likelihood to be used.
    pars_to_theory_specs: func
        function moving dictionary of cosmological parameters to dictionary of CMB Dls

    Returns
    --------------
    func:
        Function that takes a dictionary of parameter values as input and returns the chi square value
        (bundling theory code and likelihood together).

    """

    def chi_square_with_theory(sampled_pars):
        # Calculate theory Dls
        Dls = pars_to_theory_specs(sampled_pars, like.ell_max, like.ell_min)

        # Pass to likelihood
        new_pars = sampled_pars
        new_pars["Dl"] = Dls
        logl = like.chi_square(new_pars)

        return logl

    return chi_square_with_theory


# --------------------------------------#
# HELPERS TO TRANSFORM MODEL SPECTRA
# --------------------------------------#


def pars_to_model_specs(like, pars, pars_to_theory_specs):
    """
    Helper to move parameters to transformed model spectra and bin them.
    For lensing likelihoods only binned model spectra are returned.

    Parameters
    --------------
    like : candl.Like
        Likelihood to take care of transformation and binning.
    pars : dict
        Parameters for evaluation.
    pars_to_theory_specs: func
        function moving dictionary of cosmological parameters to dictionary of CMB Dls

    Returns
    --------------
    array:
        Best-fit model unbinned.
    array:
        Best-fit model binned.
    """

    # Calculate theory Dls
    Dls = pars_to_theory_specs(pars, like.ell_max, like.ell_min)

    # Pass to likelihood to apply transformations, bin
    new_pars = deepcopy(pars)
    new_pars["Dl"] = Dls

    model_unbinned = like.get_model_specs(new_pars)
    if isinstance(like, candl.LensLike):
        # For lensing likelihoods, this is already binned
        return model_unbinned, model_unbinned
    model_binned = like.bin_model_specs(model_unbinned)

    return model_unbinned, model_binned


def pars_to_model_specs_partial_transformation(
    like, pars, pars_to_theory_specs, i_tr_stop
):
    """
    Helper to move parameters to transformed model spectra and crop them.
    Only applies some of the transformations.

    Parameters
    --------------
    like : candl.Like
        Likelihood to take care of transformation and binning.
    pars : dict
        Parameters for evaluation.
    pars_to_theory_specs: func
        function moving dictionary of cosmological parameters to dictionary of CMB Dls
    i_tr_stop : int
        Index of the transformation at which to stop

    Returns
    --------------
    array:
        Model unbinned.
    array:
        Model binned.
    """

    # Calculate theory Dls
    Dls = pars_to_theory_specs(pars, like.ell_max, like.ell_min)
    new_pars = deepcopy(pars)
    new_pars["Dl"] = Dls

    # Unpack theory Dls into long vector
    modified_theory_Dls = jnp.block([new_pars["Dl"][st] for st in like.spec_types])

    # Apply nuisance modules
    for transformation in like.data_model[:i_tr_stop]:
        modified_theory_Dls = transformation.transform(modified_theory_Dls, new_pars)

    # bin if necessary
    if isinstance(like, candl.LensLike):
        # For lensing likelihoods, this is already binned
        return modified_theory_Dls, modified_theory_Dls
    best_fit_model_binned = like.bin_model_specs(modified_theory_Dls)

    return modified_theory_Dls, best_fit_model_binned


def undo_transformations(like, pars, pars_to_theory_specs):
    """
    Undo (best-fit) transformations from data vector.
    The order of the operations in the transformation matters - they do not commute.
    Hence, when undoing them, we must be careful to proceed in the correct order.
    The attribute 'operation_hint' of Transformation is used in order, if this attribute is not properly set this function will not return the correct results.
    In general, we need to undo calibration first, and then subtract the additive foregrounds.
    Special care needs to be taken for transformations that take Dls as their input too
    (e.g. super-sample lensing).

    Parameters
    --------------
    like : candl.Like
        Likelihood to take care of transformation and binning.
    pars : dict
        Parameters for evaluation.
    pars_to_theory_specs: func
        function moving dictionary of cosmological parameters to dictionary of CMB Dls

    Returns
    --------------
    array:
        Data band power with all transformations undone.
    """

    data_CMB_only_vec = deepcopy(like._data_bandpowers)

    # Undo transformations, one-by-one
    for i_tr, transformation in enumerate(like.data_model[::-1]):

        if transformation.operation_hint == "multiplicative":
            if isinstance(
                transformation, candl.transformations.abstract_base.Calibration
            ):
                # Catch calibration
                calibration_vec = transformation.transform(
                    np.ones(like.N_ell_bins_theory * like.N_spectra_total), pars
                )
                calibration_vec = np.repeat(
                    calibration_vec[:: like.N_ell_bins_theory], like.N_bins
                )

                data_CMB_only_vec /= calibration_vec
            else:
                # More general case
                unbinned_Dls, binned_Dls = pars_to_model_specs_partial_transformation(
                    like, pars, pars_to_theory_specs, len(like.data_model) - i_tr - 1
                )
                tr_vec_unbinned = transformation.transform(unbinned_Dls, pars)
                data_CMB_only_vec /= like.bin_model_specs(tr_vec_unbinned) / binned_Dls

        elif transformation.operation_hint == "additive":

            # Check if only sample parameters are required, or Dls too
            req_args_req = list(inspect.signature(transformation.output).parameters)
            if req_args_req == ["sample_params"]:
                tr_vec_unbinned = transformation.output(pars)
                data_CMB_only_vec -= like.bin_model_specs(tr_vec_unbinned)
            elif req_args_req == ["Dls", "sample_params"] or req_args_req == [
                "Dl",
                "sample_params",
            ]:
                # For the Dls we require the unbinned best-fit model Dls with all transformations up until now applied!
                unbinned_Dls, binned_Dls = pars_to_model_specs_partial_transformation(
                    like, pars, pars_to_theory_specs, len(like.data_model) - i_tr - 1
                )
                tr_vec_unbinned = transformation.output(unbinned_Dls, pars)
                data_CMB_only_vec -= like.bin_model_specs(tr_vec_unbinned)
            elif req_args_req == ["Dls"]:
                # Catches aberration
                unbinned_Dls, binned_Dls = pars_to_model_specs_partial_transformation(
                    like, pars, pars_to_theory_specs, len(like.data_model) - i_tr - 1
                )
                tr_vec_unbinned = transformation.output(unbinned_Dls)
                data_CMB_only_vec -= like.bin_model_specs(tr_vec_unbinned)
            else:
                print(
                    f"candl: not clear how to undo this transformation: {transformation}"
                )

        else:
            print(f"candl: not clear how to undo this transformation: {transformation}")

    return data_CMB_only_vec

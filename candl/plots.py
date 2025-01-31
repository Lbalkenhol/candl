"""
Tools to plot that work together with the likelihood.

Warning:
------------
This is NOT a comprehensive plotting library. Instead, the functions below are largely designed as
templates that you should modify for your specific purposes.

Overview:
------------------------

Helper for pretty LaTeX parameter labels and general plot style:

* :const:`PAR_LABEL_DICT`
* :func:`set_plot_style`

Showing the parameter covariance matrix:

* :func:`triangle_plot_from_cov`
* :func:`add_confidence_ellipse`

Analysing minimisers:

* :func:`add_min_trajectory`
* :func:`plot_minimiser_convergence`

Other:

* :func:`plot_band_powers`
* :func:`plot_mcmc_chain_steps`
* :func:`plot_foreground_components`
"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *

# --------------------------------------#
# CONSTANTS AND HELPERS
# --------------------------------------#

PAR_LABEL_DICT = {
    "ombh2": "\\Omega_{\\mathrm{b}}h^2",
    "omch2": "\\Omega_{\\mathrm{c}}h^2",
    "theta": "100\\theta_{\\mathrm{MC}}",
    "clamp": "10^9 A_{\\mathrm{s}} e^{-2\\tau}",
    "logA": "\\log{ A_{\\mathrm{s}}}",
    "ns": "n_{\\mathrm{s}}",
    "H0": "H_{\\mathrm{0}}",
    "sigma8": "\\sigma_8",
    "tau": "\\tau",
    "S8": "S_8 \\equiv \\sigma_8 \\sqrt{\\Omega_{\\mathrm{m}}/0.3}",
    "omegal": "\\Omega_{\\mathrm{\\Lambda}}",
    "age": "{\\mathrm{Age}}/{\\mathrm{Gyr}}",
    "omega_b": "\\Omega_{\\mathrm{b}}h^2",
    "omega_cdm": "\\Omega_{\\mathrm{c}}h^2",
    "ln10^{10}A_s": "\\log{ A_{\\mathrm{s}}}",
    "n_s": "n_{\\mathrm{s}}",
    "tau_reio": "\\tau",
}
"""
Dictionary of common names of and abbreviations for cosmological parameters and corresponding pretty latex strings.
"""


# --------------------------------------#
# PLOT STYLE
# --------------------------------------#


def set_plot_style():
    """
    Sets the plotting style. Important to unify things across figures and machines and generally makes plots prettier.
    Thank you to Federico Bianchini for this template!
    """
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "serif"
    # })
    rc("text", usetex=True)
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Tahoma']
    # plt.rcParams['mathtext.fontset'] = 'cm'
    # rc('font',**{'family':'sans-serif'})#,'dejavuserif':['Computer Modern']})
    # plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 8  # 12
    plt.rcParams["ytick.labelsize"] = 8  # 12
    plt.rcParams["xtick.major.size"] = 4  # 7
    plt.rcParams["ytick.major.size"] = 4  # 7
    plt.rcParams["xtick.minor.size"] = 2  # 4
    plt.rcParams["ytick.minor.size"] = 2  # 4
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.frameon"] = False

    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1
    plt.clf()
    plt.close()
    # sns.set(rc('font',**{'family':'serif','serif':['Computer Modern']}))
    # sns.set_style("ticks", {'figure.facecolor': 'grey'})


# --------------------------------------#
# CONFIDENCE ELLIPSE
# --------------------------------------#


def add_confidence_ellipse(
    ax, this_cov, ix=0, jx=1, mean_x=0, mean_y=0, n_std=1, facecolor="none", **kwargs
):
    """
    Add a confidence based on the covariance matrix to ax.
    Based on matplotlib example code with some slight modifications.
    See: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    There may be a slight mix-up between ix/jx and mean_x/mean_y, worth trying out some permutations if you use
    this function by itself.

    Parameters
    --------------
    ax : matplotlib.axes
        The axes to add the ellipse to.
    this_cov : array
        Parameter covariance matrix.
    ix : int
        Index of first parameter.
    jx : int
        Index of first parameter.
    mean_x : float
        Where to centre the ellipse in x.
    mean_y : float
        Where to centre the ellipse in y.
    n_std : float
        Which number of standard deviations to show.
    facecolor : string/array
        Facecolour of ellipse.
    kwargs :
        Other arguments to be passed through to matplotlib.patches.Ellipse.
    """

    pearson = this_cov[ix, jx] / np.sqrt(this_cov[ix, ix] * this_cov[jx, jx])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Important scaling for 2 dimensions - otherwise contours are too tight.
    # See information here https://arxiv.org/pdf/0906.4123.pdf
    prob_amount = sp.stats.norm.cdf(n_std) - sp.stats.norm.cdf(-n_std)
    scale_factor = np.sqrt(sp.stats.chi2.ppf(prob_amount, 2))

    scale_x = np.sqrt(this_cov[ix, ix]) * scale_factor
    scale_y = np.sqrt(this_cov[jx, jx]) * scale_factor

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


# --------------------------------------#
# TRIANGLE PLOT WITH CONTOURS
# --------------------------------------#


def triangle_plot_from_cov(
    pars_to_plot,
    bf_point,
    par_cov,
    pars_in_cov,
    sigma_plot_range=5,
    sigma_levels=2,
    ax=None,
    contour_colours=None,
    contour_edgecolour=None,
    zorder_offset=0,
    ls_1d="-",
    alpha_1d=1.0,
    alpha=1.0,
    set_axlims=True,
    return_handle=False,
    show_ticks=True,
    lw_1d=0.75,
    **kwargs,
):
    """
    Create a triangle plot with contours.
    This can be called multiple times to add multiple data sets to one plot. For subsequent calls pass the axes
    instance and (you probably want to) set set_axlims=False.

    Parameters
    --------------
    pars_to_plot : list
        List of parameters to plot.
    bf_point : dict
        Best-fit point, used to centre the confidence ellipses.
    par_cov : array
        Parameter covariance matrix.
    pars_in_cov : list
        List of parameters in par_cov.
    sigma_plot_range : float
        Used to set axes limits.
    sigma_levels : int
        How far out to show contours.
    ax :
        if None makes new figure, if matplotlib.axes adds to passed axes.
    contour_colours: str/array
        Face colours of the contours.
    contour_edgecolour: str/array
        Edge colours of the contours.
    zorder_offset : int
        Offset to be applied to zorder of confidence ellipses. Handy to get right background/foreground ordering.
    ls_1d : str
        Linestyle for 1d panels.
    alpha_1d : float
        Alpha for the 1d lines.
    alpha : float
        Alpha to be applied to the confidence ellipses.
    set_axlims : bool
        Whether to set the ax lims or not. Likely only want this the first time this function is called on a given
        ax instance.
    return_handle : bool
        Whether to return a handle to create a legend with.
    show_ticks : bool
        Whether to show ticks (switch off to blind results).
    lw_1d: float
        Linewidth for 1d panels.
    kwargs:
        Other arguments to be passed through to matplotlib.patches.Ellipse.
    """

    # Create subplots
    if ax is None:
        fig, ax = plt.subplots(
            len(pars_to_plot),
            len(pars_to_plot),
            figsize=(2 * 3.464, 2 * 3.464),
            gridspec_kw={"hspace": 0.0, "wspace": 0.0},
        )

    # Plotting setup
    if contour_colours is None:
        contour_colours = sns.color_palette("Greys", sigma_levels + 1)[:-1][::-1]

    # Step through parameters
    for i, p1 in enumerate(pars_to_plot):
        for j, p2 in enumerate(pars_to_plot):
            # Only one side of triangle needed
            if i < j:
                ax[i, j].axis("off")
                continue

            # Convert between indices of pars to plot and in minimiser run
            i_par = pars_in_cov.index(p1)
            j_par = pars_in_cov.index(p2)

            if i == j:
                # Plot 1D Gaussian centred on bf point
                if set_axlims:
                    # Important scaling for 2 dimensions - otherwise contours are too tight.
                    # See information here https://arxiv.org/pdf/0906.4123.pdf
                    prob_amount = sp.stats.norm.cdf(sigma_levels) - sp.stats.norm.cdf(
                        -sigma_levels
                    )
                    scale_factor = np.sqrt(sp.stats.chi2.ppf(prob_amount, 2))

                    par_range = np.linspace(
                        -np.sqrt(np.diag(par_cov)[i_par])
                        * sigma_plot_range
                        * scale_factor
                        / 2.0,
                        np.sqrt(np.diag(par_cov)[i_par])
                        * sigma_plot_range
                        * scale_factor
                        / 2.0,
                        100,
                    )
                else:
                    par_range = np.linspace(
                        ax[i, j].get_xlim()[0], ax[i, j].get_xlim()[1], 100
                    )
                    par_range -= bf_point[p1]

                line_colour = contour_colours[0]
                if line_colour == "None":
                    line_colour = contour_edgecolour

                norm = sp.stats.norm.pdf(0, scale=np.sqrt(np.diag(par_cov)[i_par]))
                (handle,) = ax[i, j].plot(
                    par_range + bf_point[p1],
                    sp.stats.norm.pdf(par_range, scale=np.sqrt(np.diag(par_cov)[i_par]))
                    / norm,
                    color=line_colour,
                    alpha=alpha_1d,
                    ls=ls_1d,
                    lw=lw_1d,
                )

            else:
                # Plot sigma contours
                for n_std in range(sigma_levels):
                    add_confidence_ellipse(
                        ax[i, j],
                        par_cov,
                        j_par,
                        i_par,
                        bf_point[p2],
                        bf_point[p1],
                        n_std + 1,
                        facecolor=contour_colours[n_std],
                        zorder=-n_std + zorder_offset,
                        alpha=alpha,
                        edgecolor=contour_edgecolour,
                        **kwargs,
                    )

            # set axes limits
            if set_axlims:
                # Important scaling for 2 dimensions - otherwise contours are too tight.
                # See information here https://arxiv.org/pdf/0906.4123.pdf
                prob_amount = sp.stats.norm.cdf(sigma_levels) - sp.stats.norm.cdf(
                    -sigma_levels
                )
                scale_factor = np.sqrt(sp.stats.chi2.ppf(prob_amount, 2))

                ax[i, j].set_xlim(
                    (
                        bf_point[p2]
                        - np.sqrt(np.diag(par_cov)[j_par])
                        * sigma_plot_range
                        * scale_factor
                        / 2.0,
                        bf_point[p2]
                        + np.sqrt(np.diag(par_cov)[j_par])
                        * sigma_plot_range
                        * scale_factor
                        / 2.0,
                    )
                )

                if i != j:
                    ax[i, j].set_ylim(
                        (
                            bf_point[p1]
                            - np.sqrt(np.diag(par_cov)[i_par])
                            * sigma_plot_range
                            * scale_factor
                            / 2.0,
                            bf_point[p1]
                            + np.sqrt(np.diag(par_cov)[i_par])
                            * sigma_plot_range
                            * scale_factor
                            / 2.0,
                        )
                    )

            # Plotting stuff
            if i == len(pars_to_plot) - 1:
                par_lbl = p2
                if p2 in list(PAR_LABEL_DICT.keys()):
                    par_lbl = rf"${PAR_LABEL_DICT[p2]}$"
                ax[i, j].set_xlabel(par_lbl)
                if not show_ticks:
                    ax[i, j].set_xticklabels([])
            else:
                ax[i, j].set_xticklabels([])

            if j == 0 and i != j:
                par_lbl = p1
                if p1 in list(PAR_LABEL_DICT.keys()):
                    par_lbl = rf"${PAR_LABEL_DICT[p1]}$"
                ax[i, j].set_ylabel(par_lbl)
                ax[i, j].yaxis.set_label_coords(-0.45, 0.5)
                if not show_ticks:
                    ax[i, j].set_yticklabels([])
                    ax[i, j].yaxis.set_label_coords(-0.3, 0.5)
            else:
                ax[i, j].set_yticklabels([])

            ax[i, j].minorticks_on()
            ax[i, j].xaxis.set_tick_params(
                which="both", bottom=True, top=i != j, direction="in"
            )

    if return_handle:
        return ax, handle

    return ax


def add_min_trajectory(
    ax,
    pars_to_plot,
    eval_points,
    par_cov,
    pars_in_cov,
    bf_point=None,
    base_colour="blue",
    dark_colours=False,
    markeredgewidth=0.25,
):
    """
    Add the trajectory of a minimiser run to a triangle plot. Designed to work with plots created by
    triangle_plot_from_cov().

    Parameters
    --------------
    ax : matplotlib.axes
        The axes to add the ellipse to.
    pars_to_plot : list
        List of parameters to plot.
    eval_points : list
        List of parameter dictionaries giving the minimiser steps.
    par_cov : array
        Parameter covariance matrix. This is only needed to set the correct height of the points in the 1d panels.
    bf_point : dict
        This is only needed to set the correct height of the points in the 1d panels.
    base_colour : str
        Base colour for the series of points.
    dark_colours : bool
        Whether points should go from colour->black (default is colour->white).
    markeredgewidth : float
        Thickness of the outline of points
    """

    if bf_point is None:
        bf_point = eval_points[-1]

    N_points = len(eval_points)

    # Make colour palette
    if dark_colours:
        all_points_colours = sns.dark_palette(base_colour, N_points)
        all_points_colours[0] = "k"
    else:
        all_points_colours = sns.light_palette(base_colour, N_points)
        all_points_colours[0] = "white"

    all_points_colours = all_points_colours[::-1]  # start in colour then go dark/light

    for i, p1 in enumerate(pars_to_plot):
        for j, p2 in enumerate(pars_to_plot):
            i_par = pars_in_cov.index(p1)
            j_par = pars_in_cov.index(p2)

            if i == j:
                # Plot path of minimisers (only individual points)!
                for i_newton in range(N_points):
                    ax[i, j].plot(
                        eval_points[i_newton][p1],
                        np.sqrt(2 * np.pi * np.diag(par_cov)[i_par])
                        * sp.stats.norm.pdf(
                            eval_points[i_newton][p1] - bf_point[p1],
                            scale=np.sqrt(np.diag(par_cov)[i_par]),
                        ),
                        color=all_points_colours[i_newton],
                        marker="o",
                        markeredgecolor="k",
                        markeredgewidth=markeredgewidth,
                        ms=4,
                        zorder=99,
                    )

            if i > j:
                # Plot 2D path of minimiser!
                ax[i, j].plot(
                    [eval_points[i_newton][p2] for i_newton in range(N_points)],
                    [eval_points[i_newton][p1] for i_newton in range(N_points)],
                    color="k",
                    lw=0.5,
                    zorder=50,
                )

                # Highlight the individual points
                for i_newton in range(N_points):
                    ax[i, j].plot(
                        eval_points[i_newton][p2],
                        eval_points[i_newton][p1],
                        color=all_points_colours[i_newton],
                        marker="o",
                        markeredgecolor="k",
                        markeredgewidth=markeredgewidth,
                        ms=4,
                        zorder=99,
                    )


# --------------------------------------#
# PLOT MINIMISER CONVERGENCE
# --------------------------------------#


def plot_minimiser_convergence(
    eval_points, pars_to_plot=None, par_cov=None, par_cov_order=None, relative=False
):
    """
    Plots the steps of the minimizer to help assess convergence.

    Parameters
    -----------------
    eval_points : list
        List of dictionaries representing the evaluation points.
    pars_to_plot : list, optional
        List of parameters to plot. If None, all parameters in eval_points will be plotted.
    par_cov : ndarray, optional
        Covariance matrix of the parameters. If provided, error bars will be plotted.
    par_cov_order : list, optional
        Order of parameters in the covariance matrix. Required if par_cov is provided.
    relative : bool, optional
        If True, the parameter values will be plotted relative to the last value and in units of the covariance. Default is False.
    """

    if not par_cov is None and par_cov_order is None:
        raise Exception(f"candl: if covariance is to be used, specify parameter oder.")

    if pars_to_plot is None:
        pars_to_plot = list(eval_points[0].keys())

    fig, ax = plt.subplots(
        len(pars_to_plot),
        1,
        sharex=True,
        sharey=False,
        gridspec_kw={"hspace": 0},
        figsize=(2 * 3.464, len(pars_to_plot) / 4 * 3.464),
    )

    for i, par in enumerate(pars_to_plot):
        par_vals = np.array([p[par] for p in eval_points])

        offset = 0
        scale = 1
        if relative:
            offset = par_vals[-1]
            scale = np.sqrt(par_cov[par_cov_order.index(par), par_cov_order.index(par)])

        ax[i].plot((par_vals - offset) / scale, color="k")

        # If covariance is passed, but relative is false, plot the error bars
        if not relative and not par_cov is None:
            for i_sigma in np.arange(1, 3):
                ax[i].axhline(
                    par_vals[-1]
                    + i_sigma
                    * np.sqrt(
                        par_cov[par_cov_order.index(par), par_cov_order.index(par)]
                    ),
                    color=["0.5", "0.7"][i_sigma - 1],
                )
                ax[i].axhline(
                    par_vals[-1]
                    - i_sigma
                    * np.sqrt(
                        par_cov[par_cov_order.index(par), par_cov_order.index(par)]
                    ),
                    color=["0.5", "0.7"][i_sigma - 1],
                )

        # ylabel
        par_label = par
        if par in PAR_LABEL_DICT:
            par_label = rf"${PAR_LABEL_DICT[par]}$"
        if relative:
            par_label = rf"$\Delta$" + par_label + rf"$/\sigma$"
        ax[i].set_ylabel(par_label)

        ax[i].yaxis.set_label_coords(-0.11, 0.5)

    # x axis
    ax[-1].set_xlabel("Minimiser Step")
    ax[-1].set_xlim(0, len(eval_points) - 1)


# --------------------------------------#
# PLOT BAND POWERS (TT/TE/EE)
# --------------------------------------#


def plot_band_powers(like, show_legend=True, colour_by_spec_type=False):
    """
    Plots the band powers for a given primary CMB likelihood.

    Parameters
    --------------
    like : candl.like
        The likelihood.
    show_legend : bool (optional)
        Whether to show the legend. Default is True.
    colour_by_spec_type bool (optional)
        Whether to colour the band powers by spectrum type. Default is False.

    """

    fig = plt.gcf()
    fig.set_size_inches(2 * 3.464, 1 * 3.464)

    if colour_by_spec_type:
        all_spec_type_colours = sns.color_palette(
            n_colors=len(np.unique(like.spec_types))
        )
        spec_type_colours = {
            np.unique(like.spec_types)[i]: all_spec_type_colours[i]
            for i in range(len(all_spec_type_colours))
        }

    for i, spec in enumerate(like.spec_order):
        # Grab plotting data
        ells_to_plot = like.effective_ells[like.bins_start_ix[i] : like.bins_stop_ix[i]]
        bandpowers_to_plot = like.data_bandpowers[
            like.bins_start_ix[i] : like.bins_stop_ix[i]
        ]
        error_bars_to_plot = np.sqrt(
            np.diag(like.covariance)[like.bins_start_ix[i] : like.bins_stop_ix[i]]
        )

        # Plot
        if colour_by_spec_type:
            lbl = like.spec_types[i]
            if lbl in like.spec_types[:i]:
                lbl = None
            plt.errorbar(
                ells_to_plot,
                abs(bandpowers_to_plot),
                error_bars_to_plot,
                lw=0,
                ms=2,
                marker="o",
                label=lbl,
                elinewidth=1,
                color=spec_type_colours[like.spec_types[i]],
            )
        else:
            plt.errorbar(
                ells_to_plot,
                abs(bandpowers_to_plot),
                error_bars_to_plot,
                lw=0,
                ms=2,
                marker="o",
                label=spec,
                elinewidth=1,
            )

    plt.yscale("log")

    if show_legend:
        plt.legend()

    plt.xlabel("Angular Multipole")
    plt.ylabel(r"$D_\ell \, [ \mu K^2 ]$")
    plt.title(like.name)


# --------------------------------------#
# PLOT MCMC CHAIN STEPS
# --------------------------------------#


def plot_mcmc_chain_steps(
    gd_samples,
    pars_to_plot,
    bf_point=None,
    par_cov=None,
    pars_in_cov=None,
    show_logl=True,
):
    """
    Plot steps from an MCMC chain. Intended to be used in conjunction with getdist and Cobaya/CosmoMC.

    Parameters
    --------------
    gd_samples : getdist.mcsamples.MCSamples
        Samples instance containing all the parameter values.
    pars_to_plot : list
        List of strings of parameter names to plot.
    bf_point : dict
        Best-fit point to indicate.
    par_cov : array (float)
        Parameter covariance matrix at the best-fit point.
    pars_in_cov : list
        List of strings specifying the order of parameters in the covariance matrix.
    show_logl : bool
        Whether to show the log likelihood values or not.
    """

    fig, ax = plt.subplots(
        len(pars_to_plot) + int(show_logl),
        1,
        figsize=(2 * 3.464, 2 * 3.464),
        sharex=True,
        gridspec_kw={"hspace": 0},
    )

    steps = np.arange(len(gd_samples.loglikes))

    # Plot logl
    if show_logl:
        ax[0].plot(steps, gd_samples.loglikes, color=sns.color_palette()[0])

        # y axis label
        ax[0].set_ylabel(r"$\log{\mathcal{L}}$")
        ax[0].yaxis.set_label_coords(-0.11, 0.5)

    # Plot parameters
    for i, par_to_plot in enumerate(pars_to_plot):
        sample_vals = gd_samples.samples[:, gd_samples.index[par_to_plot]]
        ax[i + int(show_logl)].plot(steps, sample_vals, color=sns.color_palette()[0])

        # Plot bf val if available
        if not bf_point is None:
            bf_val = bf_point[par_to_plot]
            ax[i + int(show_logl)].axhline(bf_val, color="0.7", lw=0.75, zorder=99)

            # Also plot 1 sigma band from parameter covariance if passed
            if not par_cov is None:
                fisher_err = np.sqrt(np.diag(par_cov))[pars_in_cov.index(par_to_plot)]
                for i_sigma in [-1, 1]:
                    ax[i + int(show_logl)].axhline(
                        bf_val + i_sigma * fisher_err,
                        color="0.7",
                        ls="--",
                        lw=0.75,
                        zorder=99,
                    )

        # y labels
        if par_to_plot in PAR_LABEL_DICT:
            ax[i + int(show_logl)].set_ylabel(rf"${PAR_LABEL_DICT[par_to_plot]}$")
        else:
            ax[i + int(show_logl)].set_ylabel(par_to_plot)
        ax[i + int(show_logl)].yaxis.set_label_coords(-0.11, 0.5)

    plt.xlim((np.amin(steps), np.amax(steps)))

    plt.xlabel("MCMC Step")

    return fig, ax


# --------------------------------------#
# FG COMPONENT PLOT (TT/TE/EE)
# --------------------------------------#


def plot_foreground_components(like, fg_dict):
    """
    Plot foreground components. Intended to be used with output from get_foreground_contributions() of candl.Like.
    Warning: this function is hardcoded for full ell range TT/TE/EE spectra 6 frequency combinations.
    If you have other needs please create a copy and modify it.
    This means this function can also not deal with cropped likelihoods.

    Parameters
    --------------
    like : candl.Like
        Likelihood instance. Used to get effective ell centres, spectrum order and identifiers, etc.
    fg_dict : dict
        Dictionary of spectra where each value is a dictionary holding the unbinned fg conitribution for a given
        source, i.e. the output of candl.Like.get_foreground_contributions() in the format "dict by spec".
    """

    # Figure out all foreground sources and associate them with colours
    all_fg_names = []
    for spec in fg_dict:
        all_fg_names += list(fg_dict[spec].keys())
    all_fg_names = np.unique(all_fg_names)
    all_colours = sns.color_palette(n_colors=len(all_fg_names))
    fg_colours = {all_fg_names[i]: all_colours[i] for i in range(len(all_fg_names))}

    # ylims tuned for TT/TE/EE on log scale
    ylims = {"TT": [1e-1, 1e4], "TE": [1e-2, 1e3], "EE": [1e-3, 1e2]}

    # Generate array of subplots
    fig, ax = plt.subplots(
        6,
        3,
        figsize=(2 * 3.464, 2 * 3.464),
        sharex=True,
        gridspec_kw={"hspace": 0, "wspace": 1 / 3},
    )

    # Loop over all spectra
    for i_spec, spec in enumerate(like.spec_order):
        # Grab plot indices
        ix = int(np.floor(i_spec / 3))
        jx = i_spec - ix * 3

        # Grab ell bins centres
        ell_bins = like.effective_ells[
            like.bins_start_ix[i_spec] : like.bins_stop_ix[i_spec]
        ]

        # Show data band powers
        ax[ix, jx].plot(
            ell_bins,
            abs(
                like.data_bandpowers[
                    like.bins_start_ix[i_spec] : like.bins_stop_ix[i_spec]
                ]
            ),
            color="k",
            lw=1,
        )

        # Loop over foreground components
        for fg in fg_dict[spec]:
            ax[ix, jx].plot(
                like.ells, fg_dict[spec][fg], label=fg, color=fg_colours[fg]
            )

        # Plotting stuff
        ax[ix, jx].set_yscale("log")
        ax[ix, jx].set_ylim(ylims[like.spec_types[i_spec]])
        ax[ix, jx].set_xlim((300, 4000))

        # Add legend for top row of plots
        if ix == 0:
            handles, labels = ax[ix, jx].get_legend_handles_labels()
            labels = ["\n".join(wrap(lbl, 20)) for lbl in list(labels)]
            ax[ix, jx].legend(
                handles, labels, bbox_to_anchor=(0, 1.4, 0.8, 1), loc="lower center"
            )

        # bottom x label
        if ix == 5:
            ax[ix, jx].set_xlabel("Angular Multipole $\ell$")

        # y frequency labels for the first column
        if jx == 0:
            ax[ix, jx].set_ylabel(
                rf"${'x'.join(like.spec_freqs[i_spec])}$".replace("x", " \\times ")
            )

        # Spectrum types for the first row
        if ix == 0:
            ax[ix, jx].set_title(
                f"$|D^{{{like.spec_types[i_spec]}}}_\ell| \; [\mathrm{{\mu K^2}}]$"
            )

    # Make room for legends
    plt.subplots_adjust(top=0.8)

    return fig, ax

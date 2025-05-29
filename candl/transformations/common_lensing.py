"""
Common transformations for lensing likelihoods.

Note:
----------------
The transformations of lensing likelihoods act on binned spectra.
This is NOT a comprehensive foreground/data model library. Instead, the classes below are designed
for the data sets implemented in candl and serve as examples that you can use to implement any model you want.

Overview:
----------------

* :class:`BinnedTemplateForeground`
* :class:`ResponseFunctionM`
* :class:`LensingAmplitude`
"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
import candl.transformations.abstract_base

# --------------------------------------#
# LENSING TRANSFORMATIONS
# --------------------------------------#


class BinnedTemplateForeground(candl.transformations.abstract_base.Foreground):
    """
    Template foreground that is already binned, with a single amplitude parameter.

    Used in the SPT-3G 2018 Lensing likelihood implementation.

    Attributes
    ----------------
    template_arr : array (float)
        Template spectrum and ells.
    template_spec : array (float)
        Template spectrum.
    template_ells : array (int)
        Template ells.
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    amp_param : str
        The name of the amplitude parameter.
    crop_mask : array (bool)
        Mask of which bins to use.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * template_file (str) : path to file with template spectrum.
    * amp_param (str) : name of the amplitude parameter.

    Examples
    ----------------

    Example yaml block to use a template foreground: ::

        - Module: "common_lensing.BinnedTemplateForeground"
        template_file: "foreground_templates/spt3g_2018_lensing_foreground_template.txt"
        amp_param: "A_fg"
    """

    def __init__(self, ells, template_arr, amp_param, crop_mask, descriptor=""):
        """
        Initialise a new instance of the BinnedTemplateForeground class.

        Attributes
        ----------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        amp_param : str
            The name of the amplitude parameter.
        crop_mask : array (bool)
            Mask of which bins to use.
        template_arr : array (float)
            Array with two columns, the first for ell values, the second for values of the template spectrum.

        Returns
        ----------------
        Foreground
            A new instance of the BinnedTemplateForeground class.
        """

        super().__init__(ells=ells, descriptor=descriptor, param_names=[amp_param])

        # Read in template
        self.template_arr = template_arr
        self.template_spec = self.template_arr[crop_mask, 1]
        self.template_ells = self.template_arr[crop_mask, 0]

        # Grab other args
        self.ells = ells
        self.amp_param = amp_param

    def output(self, sample_params):
        """
        Return foreground spectrum.
        Intended to be overwritten by subclasses.

        Attributes
        ----------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        ----------------
        array, float
            Foreground spectrum.
        """

        return self.template_spec * sample_params[self.amp_param]

    def transform(self, Dls, sample_params):
        """
        Transform spectrum by adding foreground component (result of output method).
        Intended to be overwritten by subclasses.

        Attributes
        ----------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        ----------------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class ResponseFunctionM(candl.transformations.abstract_base.Transformation):
    """
    Calculates :math:`M * (C^{th}) - M * (C^{fid})`.

    Used in the SPT-3G 2018 Lensing and ACT DR6 Lensing likelihood implementations.

    Attributes
    ----------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    M_matrix : array (float)
        The matrix of the transformation.
    crop_mask : array (bool)
        Mask of which bins to use.
    operation_hint : str
        Type of the 'transform' operation: 'additive'.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive contribution.
    transform :
        Returns a transformed spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * M_matrices_folder (str) : path to folder with M matrices.
    * Mmodes (list) : list of spectra to use in M matrices.
    * fiducial_correction_file (str) : path to file with the fiducial correction.

    Examples
    ----------------

    Example yaml block: ::

        - Module: "common_lensing.ResponseFunctionM"
          M_matrices_folder: "lens_delta_windows_phionly/"
          Mmodes:
            - pp
            - TT
          fiducial_correction_file: "spt3g_2018_pp_lensing_fiducial_correction_phionly.txt"
    """

    def __init__(
        self,
        ells,
        M_matrices,
        fiducial_correction,
        crop_mask,
        descriptor="ResponseFunctionM",
    ):
        """
        Initialise the ResponseFunctionM transformation.

        Attributes
        ----------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        M_matrices : dictionary of array (float)
            A dictionary with the matrix(es) of the transformation.
        fiducial_correction : array (float)
            An array with the fiducial correction (M * C_fid) values.
        crop_mask : array (bool)
            Mask of which bins to use.

        Returns
        ----------------
        Transformation
            A new instance of the ResponseFunctionM class.
        """

        super().__init__(ells=ells, descriptor=descriptor, operation_hint="additive")

        # Crop down fiducial correction and M_matrices for bin selection
        self.M_matrices = {}
        for mode in list(M_matrices.keys()):
            self.M_matrices[mode] = jnp.array(M_matrices[mode][:, crop_mask])
        self.fiducial_correction = jnp.array(fiducial_correction[crop_mask])

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return the correction.
        Needs to use unbinned (from params) spectra.

        Attributes
        ----------------
        sample_params dict
            Contains the unbinned (pp, TT) theory spectra to be used in calculating (M * C).

        Returns
        ----------------
        array, float
            Response function contribution.
        """

        # M * (Cth) - M * (Cfid)

        # get required modes and fiducial correction
        M_modes = self.M_matrices.keys()
        M_correction = -self.fiducial_correction

        # multiply arrays according to length of theory spectra
        for mode in M_modes:
            M_correction += jnp.dot(
                np.transpose(self.M_matrices[mode][: len(self.ells)]),
                jnp.block(sample_params["Dl"][mode]),
            )
        return M_correction

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Attributes
        ----------------
        Dictionary of Dls : dict of array (float)
            The binned spectra (pp) to add to (M * C).
        sample_params dict
            Contains the unbinned (pp, TT) theory spectra to be used in calculating (M * C).

        Returns
        ----------------
        array : float
            Response function contribution.
        """

        return Dls + self.output(sample_params)


class LensingAmplitude(candl.transformations.abstract_base.Transformation):
    """
    Lensing amplitude function, which multiplies the input by some factor.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * amp_param (str) : name of the amplitude parameter.

    Examples
    ----------------

    Example yaml block to a template foreground: ::

        - Module: "common_lensing.LensingAmplitude"
          amp_param: "AL"

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        amp_param.
    transform :
        transforms an input spectrum as amp_param * input.

    Attributes
    ----------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    amp_param : str
        The name of the amplitude parameter.
    operation_hint : str
        Type of the 'transform' operation: 'additive'.
    """

    def __init__(self, ells, amp_param, descriptor=""):
        """
        Initialise a new instance of the LensingAmplitude class.

        Attributes
        ----------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        amp_param : str
            The name of the amplitude parameter.

        Returns
        ----------------
        Foreground
            A new instance of the LensingAmplitude class.
        """

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=[amp_param],
            operation_hint="multiplicative",
        )

        # Grab other args
        self.ells = ells
        self.amp_param = amp_param

    def output(self, sample_params):
        """
        Return multiplicative factor.
        Intended to be overwritten by subclasses.

        Attributes
        ----------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        ----------------
        float
            Multiplicative factor.
        """

        return sample_params[self.amp_param]

    def transform(self, Dls, sample_params):
        """
        Transform spectrum by multiplying by overall constant factor (result of output method).
        Intended to be overwritten by subclasses.

        Attributes
        ----------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        ----------------
        array, float
            Transformed spectrum.
        """

        return Dls * self.output(sample_params)


class NormCorr(candl.transformations.abstract_base.Transformation):
    """
    Calculates :math:`-2 \\frac{\\frac{d A}{d C} \Delta C}{f} C^{fid}`.

    Used in the ACT DR6 Lensing likelihood.

    Attributes
    ----------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    cl2dl : array (float)
        Conversion helper.
    dAL_dC : array (bool)
        Array for matrix multiplication (see equation in docs).
    fAl : array (bool)
        Array for normalisation (see equation in docs).
    template_arr_fiducial_Cl_kk : array (bool)
        Fiducial lensing spectrum.
    fiducial_correction : array (bool)
        Fiducial correction.
    window_functions : list
        List of window functions to use for binning.
    operation_hint : str
        Type of the 'transform' operation: 'additive'.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive contribution.
    transform :
        Returns a transformed spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * M_matrices_folder (str) : path to folder with M matrices.
    * Mmodes (list) : list of spectra to use in M matrices.
    * fiducial_correction_file (str) : path to file with the fiducial correction.

    Examples
    ----------------

    Example yaml block: ::

        - Module: "common_lensing.NormCorr"
          template_file_dAL_dC_TT: "data/ACT_DR6_Lens_v1/dAL_dC_TT.txt"
          template_file_dAL_dC_TE: "data/ACT_DR6_Lens_v1/dAL_dC_TE.txt"
          template_file_dAL_dC_EE: "data/ACT_DR6_Lens_v1/dAL_dC_EE.txt"
          template_file_dAL_dC_BB: "data/ACT_DR6_Lens_v1/dAL_dC_BB.txt"
          template_file_fAL: "data/ACT_DR6_Lens_v1/fAL.txt"
          template_file_fiducial_correction: "data/ACT_DR6_Lens_v1/norm_corr_fiducial_correction.txt"
          template_file_fiducial_Cl_kk: "data/ACT_DR6_Lens_v1/fiducial_Cl_kk.txt"

    """

    def __init__(
        self,
        ells,
        template_arr_dAL_dC_TT,
        template_arr_dAL_dC_TE,
        template_arr_dAL_dC_EE,
        template_arr_dAL_dC_BB,
        template_arr_fAL,
        template_arr_fiducial_Cl_kk,
        template_arr_fiducial_correction,
        window_functions,
        descriptor="NormCorr",
    ):
        """
        Initialise a new instance of the LensingAmplitude class.

        Attributes
        ----------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        amp_param : str
            The name of the amplitude parameter.

        Returns
        ----------------
        Foreground
            A new instance of the LensingAmplitude class.
        """

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            operation_hint="additive",
        )

        # Grab other args
        self.ells = ells
        self.cl2dl = ells * (ells + 1) / (2.0 * np.pi)
        self.dAL_dC = {
            "TT": template_arr_dAL_dC_TT,
            "TE": template_arr_dAL_dC_TE,
            "EE": template_arr_dAL_dC_EE,
            "BB": template_arr_dAL_dC_BB,
        }
        self.fAL = template_arr_fAL
        self.template_arr_fiducial_Cl_kk = template_arr_fiducial_Cl_kk
        self.fiducial_correction = template_arr_fiducial_correction
        self.window_functions = window_functions

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """ """

        # multiply arrays according to length of theory spectra
        norm_corr = 0
        for i, spec in enumerate(["TT", "TE", "EE", "BB"]):
            norm_corr += -2.0 * (
                self.dAL_dC[spec] @ (sample_params["Dl"][spec] / self.cl2dl)
            )
        norm_corr = norm_corr / self.fAL
        norm_corr = norm_corr * self.template_arr_fiducial_Cl_kk

        # now bin
        binned_norm_corr = self.window_functions[0].T @ norm_corr
        binned_norm_corr -= self.fiducial_correction

        return binned_norm_corr

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        return Dls + self.output(sample_params)

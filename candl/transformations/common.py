"""
Common transformations, including additive foregrounds (Poisson, CIB, ...), calibration, super-sample lensing, and aberration as well as required helper functions.
These are all that is required to run the data sets made available at initial release.
Most of the specific foreground subclasses contain examples for how to include them in the data set yaml file.

Note:
---------------
Warning: this is NOT a comprehensive foreground/data model library. Instead, the classes below are designed for the data sets implemented in candl and serve as examples that you can use to implement any model you want.

Overview:
-----------------

Extragalactic foregrounds:

* :class:`PoissonPower`
* :class:`CIBClustering`
* :class:`tSZTemplateForeground`
* :class:`kSZTemplateForeground`
* :class:`CIBtSZCorrelationGeometricMean`

Galactic contamination:

* :class:`GalacticDust`
* :class:`GalacticDustBandPass`

Calibration:

* :class:`CalibrationCross`
* :class:`PolarisationCalibration`

Other:

* :class:`SuperSampleLensing`
* :class:`AberrationCorrection`

Frequency scaling functions:

* :func:`dust_frequency_scaling`
* :func:`dust_frequency_scaling_bandpass`
* :func:`tSZ_frequency_scaling`
* :func:`black_body`
* :func:`black_body_deriv`
"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
import candl.transformations.abstract_base
import candl.constants

# --------------------------------------#
# RADIO GALAXIES
# --------------------------------------#


class PoissonPower(candl.transformations.abstract_base.Foreground):
    """
    Adds individual Poisson terms to a series of spectra.

    .. math::
        A * \\left( \\ell / \\ell_{ref} \\right)^2

    where:

    * :math:`A` is the amplitude
    * :math:`\\ell_{ref}` is the reference ell

    Used by SPT-3G 2018 TT/TE/EE implementation.

    Attributes
    --------------------
    ells : array (float)
            The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    ell_ref : float
        Reference ell.
    spec_param_dict : dict
        A dictionary with keys that are spectrum identifiers and values that are lists of the nuisance parameter names
        that are used to transform this spectrum.
    spec_order : list
        Order of the spectra in the long data vector.
    spec_mask : array (int)
        Masks which parts of the long data vector are affected by the transformation.
    N_spec : int
        The total number of spectra in the long data vector.
    affected_specs_ix : list (int)
        Indices in spectra_order of spectra the transformation is applied to.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    ----------------
    User required arguments in data set yaml file:

    * ell_ref (float) : Reference ell.
    * spec_param_dict (dict) : A dictionary with keys that are spectrum identifiers and values that are lists of the nuisance parameter names that are used to transform this spectrum.

    Examples
    -------------------
    Example yaml block to add individual Poisson terms to a series of TT spectra: ::

        - Module: "common.PoissonPower"
          ell_ref: 3000
          spec_param_dict:
            TT 90x90: "TT_Poisson_90x90"
            TT 90x150: "TT_Poisson_90x150"
            TT 90x220: "TT_Poisson_90x220"
            TT 150x150: "TT_Poisson_150x150"
            TT 150x220: "TT_Poisson_150x220"
            TT 220x220: "TT_Poisson_220x220"

    """

    def __init__(
        self, ells, spec_order, spec_param_dict, ell_ref, descriptor="Poisson Power"
    ):
        """
        Initialise a new instance of the PoissonPower class.

        Arguments
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        spec_order : array (str)
            Identifiers of spectra in the order in which spectra are handled in the long data vector.
        spec_param_dict : dict
            A dictionary with keys that are spectrum identifiers and values that are lists of the nuisance parameter names
            that are used to transform this spectrum.
        spec_order : list
            Order of the spectra in the long data vector.
        ell_ref : float
            Reference ell.

        Returns
        --------------
        PoissonPower
            A new instance of the PoissonPower class.
        """

        super().__init__(
            ells=ells,
            ell_ref=ell_ref,
            descriptor=descriptor,
            param_names=list(spec_param_dict.values()),
        )

        self.spec_param_dict = spec_param_dict
        self.affected_specs = list(self.spec_param_dict.keys())

        self.spec_order = spec_order
        self.N_spec = len(self.spec_order)

        # Generate boolean mask of affected specs
        self.spec_mask = np.zeros(
            len(spec_order)
        )  # Generate as np array for easier item assignment
        for i, spec in enumerate(self.spec_order):
            if spec in self.affected_specs:
                self.spec_mask[i] = 1
                # self.spec_mask = self.spec_mask.at[i].set(1)
        self.spec_mask = self.spec_mask == 1
        self.spec_mask = jnp.array(self.spec_mask)
        self.affected_specs_ix = [ix[0] for ix in jnp.argwhere(self.spec_mask)]

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part
        amp_vals = jnp.zeros(len(self.spec_order))
        for ix in self.affected_specs_ix:
            amp_vals = jax_optional_set_element(
                amp_vals, ix, sample_params[self.spec_param_dict[self.spec_order[ix]]]
            )
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** 2
        tiled_ell_dependence = jnp.tile(ell_dependence, self.N_spec)

        # Complete foreground contribution
        fg_pow = tiled_amp_vals * tiled_ell_dependence
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sample_params : dict
            A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
            The transformed spectrum in Dl.
        """

        return Dls + self.output(sample_params)


class CIBClustering(candl.transformations.abstract_base.DustyForeground):
    """
    Adds CIB clustering power using a power law with fixed index.

    .. math::

        A * g(\\nu_1, \\beta) * g(\\nu_2, \\beta) * \\left( \\ell / \\ell_{ref} \\right)^\\alpha

    where:

    * :math:`A` is the amplitude
    * :math:`\\ell_{ref}` is the reference ell
    * :math:`\\alpha` is the power law index
    * :math:`\\beta` is the frequency scaling parameter
    * :math:`g(\\nu, \\beta)` is the frequency scaling for a modified black body

    Used by SPT-3G 2018 TT/TE/EE implementation.

    Attributes
    --------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    spec_order : array (str)
        Identifiers of spectra in the order in which spectra are handled in the long data vector.
    freq_info : list
        List of lists, where each sublist contains the two effective frequencies for a given spectrum.
    affected_specs : list (str)
        List of the spectra to apply this foreground to.
    ell_ref : int
        Reference ell for normalisation.
    nu_ref : float
        Reference frequency.
    T_dust : float
        Temperature of the dust.
    spec_mask : array (int)
        Masks which spectra of the long data vector are affected by the transformation.
    full_mask : array (int)
        Masks which elements of the long data vector are affected by the transformation.
    N_spec : int
        The total number of spectra in the long data vector.
    amp_param : str
        The name of the amplitude parameter.
    beta_param : str
        The name of the frequency scaling parameter.
    alpha : float
        The power law index.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * ell_ref (float) : Reference ell.
    * nu_ref (float) : Reference frequency.
    * T_CIB (float) : Temperature of the CIB.
    * amp_param (str) : The name of the amplitude parameter.
    * beta_param (str) : The name of the frequency scaling parameter.
    * alpha (float) : The power law index.
    * effective_frequencies (str) : Keyword to look for in effective frequencies yaml file.
    * affected_specs (str) : List of spectrum identifiers the transformation is applied to.

    Examples
    -------------------

    Example yaml block to add CIB clustering power to all TT spectra::

        - Module: "common.CIBClustering"
          amp_param: "TT_CIBClustering_Amp"
          alpha: 0.8
          beta_param: "TT_CIBClustering_Beta"
          effective_frequencies: "CIB"# keyword in effective frequencies file with corresponding entry
          affected_specs: ["TT 90x90", "TT 90x150", "TT 90x220", "TT 150x150", "TT 150x220", "TT 220x220"]
          ell_ref: 3000
          nu_ref: 150
          T_CIB: 25
    """

    def __init__(
        self,
        ells,
        spec_order,
        freq_info,
        affected_specs,
        amp_param,
        beta_param,
        alpha,
        ell_ref,
        nu_ref,
        T_CIB,
        descriptor="CIB clustering",
    ):
        """
        Initialise a new instance of the CIBClustering class.

        Arguments
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        spec_order : array (str)
            Identifiers of spectra in the order in which spectra are handled in the long data vector.
        freq_info : list
            List of lists, where each sublist contains the two effective frequencies for a given spectrum.
        affected_specs : list (str)
            List of the spectra to apply this foreground to.
        ell_ref : int
            Reference ell for normalisation.
        nu_ref : float
            Reference frequency.
        T_CIB : float
            Temperature of the CIB.
        amp_param : str
            The name of the amplitude parameter.
        beta_param : str
            The name of the frequency scaling parameter.
        alpha : float
            The power law index.

        Returns
        --------------
        CIBClustering
            A new instance of the class.
        """

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            freq_info=freq_info,
            affected_specs=affected_specs,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            T_dust=T_CIB,
            descriptor=descriptor,
            param_names=[amp_param, beta_param],
        )

        # Hold onto names of parameters
        self.amp_param = amp_param
        self.beta_param = beta_param
        self.alpha = alpha

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part
        amp_vals = jnp.array(
            [
                dust_frequency_scaling(
                    sample_params[self.beta_param],
                    self.T_dust,
                    self.nu_ref,
                    self.freq_info[i][0],
                )
                * dust_frequency_scaling(
                    sample_params[self.beta_param],
                    self.T_dust,
                    self.nu_ref,
                    self.freq_info[i][1],
                )
                for i in range(self.N_spec)
            ]
        )
        amp_vals *= sample_params[self.amp_param]
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** self.alpha
        tiled_ell_dependence = jnp.tile(
            ell_dependence, self.N_spec
        )  # tiled ell dependence

        # Complete foreground contribution and mask down
        fg_pow = self.full_mask * tiled_amp_vals * tiled_ell_dependence
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform spectrum by adding foreground component (result of output method).

        Arguments
        --------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class GalacticDust(candl.transformations.abstract_base.DustyForeground):
    """
    Adds galactic dust power using a power law.

    .. math::

        A * g(\\nu_1, b\\eta) * g(\\nu_2, b\\eta) * \\left( \\ell / \\ell_{ref} \\right)^{(\\alpha+2)}

    where:

    * :math:`A` is the amplitude
    * :math:`\\ell_{ref}` is the reference ell
    * :math:`\\alpha` is the power law index
    * :math:`\\beta` is the frequency scaling parameter
    * :math:`g(\\nu, \\beta)` is the frequency scaling for a modified black body

    Used by SPT-3G 2018 TT/TE/EE implementation.

    Attributes
    --------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    spec_order : array (str)
        Identifiers of spectra in the order in which spectra are handled in the long data vector.
    freq_info : list
        List of lists, where each sublist contains the two effective frequencies for a given spectrum.
    affected_specs : list (str)
        List of the spectra to apply this foreground to.
    ell_ref : int
        Reference ell for normalisation.
    nu_ref : float
        Reference frequency.
    T_dust : float
        Temperature of the dust.
    spec_mask : array (int)
        Masks which spectra of the long data vector are affected by the transformation.
    full_mask : array (int)
        Masks which elements of the long data vector are affected by the transformation.
    N_spec : int
        The total number of spectra in the long data vector.
    amp_param : str
        The name of the amplitude parameter.
    beta_param : str
        The name of the frequency scaling parameter.
    alpha_param : str
        The name of the power law index parameter.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * ell_ref (float) : Reference ell.
    * nu_ref (float) : Reference frequency.
    * T_GALDUST (float) : Temperature of the dust.
    * amp_param (str) : The name of the amplitude parameter.
    * beta_param (str) : The name of the frequency scaling parameter.
    * alpha (float) : The power law index.
    * effective_frequencies (str) : Keyword to look for in effective frequencies yaml file.
    * affected_specs (str) : List of spectrum identifiers the transformation is applied to.

    Examples
    -------------------

    Example yaml block to add residual cirrus power to all TT spectra::

        - Module: "common.GalacticDust"
          descriptor: "Cirrus"
          amp_param: "TT_GalCirrus_Amp"
          alpha_param: "TT_GalCirrus_Alpha"
          beta_param: "TT_GalCirrus_Beta"
          effective_frequencies: "cirrus"# keyword in effective frequencies file with corresponding entry
          affected_specs: ["TT 90x90", "TT 90x150", "TT 90x220", "TT 150x150", "TT 150x220", "TT 220x220"]
          ell_ref: 80
          nu_ref: 150
          T_GALDUST: 19.6
    """

    def __init__(
        self,
        ells,
        spec_order,
        freq_info,
        affected_specs,
        amp_param,
        alpha_param,
        beta_param,
        ell_ref,
        nu_ref,
        T_GALDUST,
        descriptor="Galactic Dust",
    ):
        """
        Initialise a new instance of the GalacticDust class.

        Arguments
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        spec_order : array (str)
            Identifiers of spectra in the order in which spectra are handled in the long data vector.
        freq_info : list
            List of lists, where each sublist contains the two effective frequencies for a given spectrum.
        affected_specs : list (str)
            List of the spectra to apply this foreground to.
        ell_ref : int
            Reference ell for normalisation.
        nu_ref : float
            Reference frequency.
        T_GALDUST : float
            Temperature of the CIB.
        amp_param : str
            The name of the amplitude parameter.
        beta_param : str
            The name of the frequency scaling parameter.
        alpha_param : str
            The name of the power law index parameter.

        Returns
        --------------
        GalacticDust
            A new instance of the class.
        """

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            freq_info=freq_info,
            affected_specs=affected_specs,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            T_dust=T_GALDUST,
            descriptor=descriptor,
            param_names=[amp_param, beta_param, alpha_param],
        )

        # Hold onto names of parameters
        self.amp_param = amp_param
        self.beta_param = beta_param
        self.alpha_param = alpha_param

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part
        amp_vals = jnp.array(
            [
                dust_frequency_scaling(
                    sample_params[self.beta_param],
                    self.T_dust,
                    self.nu_ref,
                    self.freq_info[i][0],
                )
                * dust_frequency_scaling(
                    sample_params[self.beta_param],
                    self.T_dust,
                    self.nu_ref,
                    self.freq_info[i][1],
                )
                for i in range(self.N_spec)
            ]
        )
        amp_vals *= sample_params[self.amp_param]
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** (
            sample_params[self.alpha_param] + 2
        )
        tiled_ell_dependence = jnp.tile(
            ell_dependence, self.N_spec
        )  # tiled ell dependence

        # Complete foreground contribution and mask down
        fg_pow = self.full_mask * tiled_amp_vals * tiled_ell_dependence
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform spectrum by adding foreground component (result of output method).

        Arguments
        --------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class GalacticDustBandPass(candl.transformations.abstract_base.ForegroundBandPass):
    """
    Dusty foreground with modified black-body frequency scaling with integral over band pass with a power law ell
    power spectrum.

    .. math::

        A * f(\\beta, \\mathrm{bdp}_1) * f(\\beta, \\mathrm{bdp}_2) * \\left( \\ell / \\ell_{ref} \\right)^{(\\alpha + 2)}

    where:

    * :math:`A` is the amplitude
    * :math:`\\ell_{ref}` is the reference ell
    * :math:`\\alpha` is the power law index
    * :math:`\\beta` is the frequency scaling parameter
    * :math:`f(\\beta, \\mathrm{bdp})` is the frequency scaling for a modified black body with band pass :math:`\\mathrm{bdp}`

    The +2 to the exponent is convention.

    Attributes
    --------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    spec_order : array (str)
        Identifiers of spectra in the order in which spectra are handled in the long data vector.
    bandpass_info : list
        List of lists, where each sublist contains the two candl.transformations.abstract_base.BandPass instances for the two
        frequencies involved.
    ell_ref : int
        Reference ell for normalisation.
    nu_ref : float
        Reference frequency.
    N_spec : int
        The total number of spectra in the long data vector.
    affected_specs : list (str)
        List of the spectra to apply this foreground to.
    spec_mask : array (int)
        Masks which spectra of the long data vector are affected by the transformation.
    full_mask : array (int)
        Masks which elements of the long data vector are affected by the transformation.
    T_dust : float
        Dust temperature.
    amp_param : str
        The name of the amplitude parameter.
    alpha_param : str
        The name of the power law index parameter.
    beta_param : str
        The name of the frequency scaling parameter.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * ell_ref (float) : Reference ell.
    * nu_ref (float) : Reference frequency.
    * T_GALDUST (float) : Temperature of the dust.
    * amp_param (str) : The name of the amplitude parameter.
    * beta_param (str) : The name of the frequency scaling parameter.
    * alpha (float) : The power law index.
    * affected_specs (str) : List of spectrum identifiers the transformation is applied to.

    Examples
    -------------------

    Example yaml block to add polarised galactic dust to all EE spectra::

        - Module: "common.GalacticDustBandPass"
          amp_param: "BB_GalDust_BDP_Amp"
          alpha_param: "BB_GalDust_BDP_Alpha"
          beta_param: "BB_GalDust_BDP_Beta"
          nu_ref: 353
          affected_specs: ["BB 90x90", "BB 90x150", "BB 90x220", "BB 150x150", "BB 150x220", "BB 220x220"]
          ell_ref: 80
          T_GALDUST: 19.6
          descriptor: "BB Polarised Galactic Dust (Bandpass)"

    """

    def __init__(
        self,
        ells,
        spec_order,
        bandpass_info,
        affected_specs,
        amp_param,
        alpha_param,
        beta_param,
        ell_ref,
        nu_ref,
        T_GALDUST,
        descriptor="Galactic Dust (Band pass)",
    ):
        """
        Initialise a new instance of the GalacticDustBandPass class.

        Arguments
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        bandpass_info : list
            List of lists, where each sublist contains the two candl.transformations.abstract_base.BandPass instances for the two
            frequencies involved.
        spec_order : array (str)
            Identifiers of spectra in the order in which spectra are handled in the long data vector.
        affected_specs : list (str)
            List of the spectra to apply this foreground to.
        ell_ref : int
            Reference ell for normalisation.
        nu_ref : float
            Reference frequency.
        T_GALDUST : float
            Temperature of the CIB.
        amp_param : str
            The name of the amplitude parameter.
        alpha_param : str
            The name of the power law index parameter.
        beta_param : str
            The name of the frequency scaling parameter.

        Returns
        --------------
        GalacticDustBandPass
            A new instance of the class.
        """

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            bandpass_info=bandpass_info,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            descriptor=descriptor,
            param_names=[amp_param, alpha_param, beta_param],
        )

        self.T_dust = T_GALDUST
        self.amp_param = amp_param
        self.alpha_param = alpha_param
        self.beta_param = beta_param

        self.affected_specs = affected_specs
        self.spec_mask = jnp.asarray(
            [spec in self.affected_specs for spec in self.spec_order]
        )

        # Turn spectrum mask into a full mask
        self.full_mask = jnp.asarray(
            jnp.repeat(self.spec_mask, len(self.ells)), dtype=float
        )

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part (frequency scaling)
        amp_vals = jnp.array(
            [
                dust_frequency_scaling_bandpass(
                    sample_params[self.beta_param],
                    self.T_dust,
                    self.nu_ref,
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref],
                )
                * dust_frequency_scaling_bandpass(
                    sample_params[self.beta_param],
                    self.T_dust,
                    self.nu_ref,
                    self.bandpass_info[i][1].nu_spacing,
                    self.bandpass_info[i][1].nu_vals,
                    self.bandpass_info[i][1].bandpass_vals,
                    self.bandpass_info[i][1].thermo_conv[self.nu_ref],
                )
                for i in range(self.N_spec)
            ]
        )

        amp_vals *= sample_params[self.amp_param]
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** (
            sample_params[self.alpha_param] + 2
        )
        tiled_ell_dependence = jnp.tile(
            ell_dependence, self.N_spec
        )  # tiled ell dependence

        # Complete foreground contribution and mask down
        fg_pow = self.full_mask * tiled_amp_vals * tiled_ell_dependence
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform spectrum by adding foreground component (result of output method).

        Arguments
        --------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


# --------------------------------------#
# tSZ AND kSZ TEMPLATE FOREGROUNDS
# --------------------------------------#


class tSZTemplateForeground(candl.transformations.abstract_base.TemplateForeground):
    """
    tSZ template with frequency scaling and one free amplitude parameter.

    .. math::
        A * g(\\nu_1) * g(\\nu_2) * D^{\\mathrm{template}}_{\\ell_{ref}}

    where:

    * :math:`A` is the amplitude parameter
    * :math:`g(\\nu)` is the appropriate frequency scaling

    Used by SPT-3G 2018 TT/TE/EE implementation.

    Attributes
    --------------
    template_arr : array (float)
        Template spectrum and ells.
    template_spec : array (float)
        Template spectrum.
    template_ells : array (int)
        Template ells.
    ell_ref : int
        Reference ell for normalisation.
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    nu_ref : float
        Reference frequency.
    spec_order : array (str)
        Identifiers of spectra in the order in which spectra are handled in the long data vector.
    freq_info : list
        List of lists, where each sublist contains the two effective frequencies for a given spectrum.
    affected_specs : list (str)
        List of the spectra to apply this foreground to.
    spec_mask : array (int)
        Masks which spectra of the long data vector are affected by the transformation.
    full_mask : array (int)
        Masks which elements of the long data vector are affected by the transformation.
    N_spec : int
        The total number of spectra in the long data vector.
    amp_param : str
        The name of the amplitude parameter.
    template_spec_tiled : array (float)
        Template spectrum repeated N_spec times.
    T_CMB : float
        CMB temperature.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * ell_ref (float) : Reference ell.
    * template_file (str) : Relative path to the template file from the candl/ folder.
    * affected_specs (list) : List of the spectra to apply this foreground to.
    * amp_param (str) : The name of the amplitude parameter.
    * nu_ref (float) : Reference frequency.
    * effective_frequencies (str) : Keyword to look for in effective frequencies yaml file.

    Examples
    ------------
    Example yaml block to add tSZ power to all TT spectra::

        - Module: "common.tSZTemplateForeground"
          template_file: "foreground_templates/dl_shaw_tsz_s10_153ghz_norm1_fake25000.txt"
          amp_param: "TT_tSZ_Amp"
          effective_frequencies: "tSZ"# keyword in effective frequencies file with corresponding entry
          affected_specs: ["TT 90x90", "TT 90x150", "TT 90x220", "TT 150x150", "TT 150x220", "TT 220x220"]
          ell_ref: 3000
          nu_ref: 143

    """

    def __init__(
        self,
        ells,
        spec_order,
        freq_info,
        affected_specs,
        template_arr,
        amp_param,
        ell_ref,
        nu_ref,
        descriptor="tSZ",
    ):
        """
        Initialise a new instance of the tSZTemplateForeground class.

        Arguments
        --------------
        template_arr : array (float)
            Template spectrum and ells.
        ell_ref : int
            Reference ell for normalisation.
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        nu_ref : float
            Reference frequency.
        spec_order : array (str)
            Identifiers of spectra in the order in which spectra are handled in the long data vector.
        freq_info : list
            List of lists, where each sublist contains the two effective frequencies for a given spectrum.
        affected_specs : list (str)
            List of the spectra to apply this foreground to.
        amp_param : str
            The name of the amplitude parameter.

        Returns
        --------------
        Foreground
            A new instance of the tSZTemplateForeground class.
        """

        super().__init__(
            ells=ells,
            template_arr=template_arr,
            ell_ref=ell_ref,
            descriptor=descriptor,
            param_names=[amp_param],
        )

        self.spec_order = spec_order
        self.affected_specs = affected_specs
        self.spec_mask = jnp.asarray(
            [spec in self.affected_specs for spec in self.spec_order]
        )
        self.nu_ref = nu_ref
        self.amp_param = amp_param
        self.freq_info = freq_info
        self.T_CMB = candl.constants.T_CMB
        self.N_spec = len(freq_info)

        # Turn spectrum mask into a full mask
        self.full_mask = jnp.asarray(
            jnp.repeat(self.spec_mask, len(self.ells)), dtype=float
        )

        # Tile template
        self.template_spec_tiled = jnp.tile(self.template_spec, self.N_spec)

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part
        amp_vals = jnp.array(
            [
                tSZ_frequency_scaling(self.freq_info[i][0], self.nu_ref, self.T_CMB)
                * tSZ_frequency_scaling(self.freq_info[i][1], self.nu_ref, self.T_CMB)
                for i in range(self.N_spec)
            ]
        )
        amp_vals *= sample_params[self.amp_param]
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # Put together with ell template and mask
        fg_pow = self.full_mask * tiled_amp_vals * self.template_spec_tiled
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform spectrum by adding foreground component (result of output method).

        Arguments
        --------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class kSZTemplateForeground(candl.transformations.abstract_base.TemplateForeground):
    """
    kSZ template spectrum.
    Used by SPT-3G 2018 TT/TE/EE implementation.

    Attributes
    --------------
    template_arr : array (float)
        Template spectrum and ells.
    template_spec : array (float)
        Template spectrum.
    template_ells : array (int)
        Template ells.
    ell_ref : int
        Reference ell for normalisation.
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    spec_order : array (str)
        Identifiers of spectra in the order in which spectra are handled in the long data vector.
    affected_specs : list (str)
        List of the spectra to apply this foreground to.
    spec_mask : array (int)
        Masks which spectra of the long data vector are affected by the transformation.
    full_mask : array (int)
        Masks which elements of the long data vector are affected by the transformation.
    N_spec : int
        The total number of spectra in the long data vector.
    amp_param : str
        The name of the amplitude parameter.
    template_spec_tiled : array (float)
        Template spectrum repeated N_spec times.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * ell_ref (float) : Reference ell.
    * template_file (str) : Relative path to the template file from the candl/ folder.
    * affected_specs (list) : List of the spectra to apply this foreground to.
    * amp_param (str) : The name of the amplitude parameter.


    Examples
    ------------

    Example yaml block to add kSZ power to all TT spectra::

        - Module: "common.kSZTemplateForeground"
          template_file: "foreground_templates/dl_ksz_CSFplusPATCHY_13sep2011_norm1_fake25000.txt"
          amp_param: "TT_kSZ_Amp"
          affected_specs: [ "TT 90x90", "TT 90x150", "TT 90x220", "TT 150x150", "TT 150x220", "TT 220x220" ]
          ell_ref: 3000
    """

    def __init__(
        self,
        ells,
        spec_order,
        affected_specs,
        template_arr,
        amp_param,
        ell_ref,
        descriptor="kSZ",
    ):
        """
        Initialise a new instance of the kSZTemplateForeground class.

        Arguments
        --------------
        template_arr : array (float)
            Template spectrum and ells.
        ell_ref : int
            Reference ell for normalisation.
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        spec_order : array (str)
            Identifiers of spectra in the order in which spectra are handled in the long data vector.
        affected_specs : list (str)
            List of the spectra to apply this foreground to.
        amp_param : str
            The name of the amplitude parameter.

        Returns
        --------------
        Foreground
            A new instance of the kSZTemplateForeground class.
        """

        super().__init__(
            ells=ells,
            template_arr=template_arr,
            ell_ref=ell_ref,
            descriptor=descriptor,
            param_names=[amp_param],
        )

        self.spec_order = spec_order
        self.affected_specs = affected_specs
        self.spec_mask = jnp.asarray(
            [spec in self.affected_specs for spec in self.spec_order]
        )
        self.amp_param = amp_param
        self.N_spec = len(self.spec_mask)

        # Turn spectrum mask into a full mask
        self.full_mask = jnp.asarray(
            jnp.repeat(self.spec_mask, len(self.ells)), dtype=float
        )

        # Tile template
        self.template_spec_tiled = jnp.tile(self.template_spec, self.N_spec)

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part
        tiled_amp_vals = jnp.repeat(
            sample_params[self.amp_param], len(self.ells) * self.N_spec
        )

        # Put together with ell template and mask
        fg_pow = self.full_mask * tiled_amp_vals * self.template_spec_tiled
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform spectrum by adding foreground component (result of output method).

        Arguments
        --------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


# CIB-tSZ correlation functions defined outside of class to guarantee differentiability
@partial(custom_jvp, nondiff_argnums=(0,))
def _CIBtSZCorrelationGeometricMean_output(fg_class_instance, sample_params):
    """
    Method to calculate tSZ-CIB correlation term using a geometric mean of individual terms.
    Separated from class to allow for custom derivative method.
    See also: candl.transformations.common.CIBtSZCorrelationGeometricMean

    Parameters
    ----------
    fg_class_instance : candl.transformations.common.CIBtSZCorrelationGeometricMean
        The CIB-tSZ class instance to use.
    pars : dict
        Dictionary of nuisance parameter values.

    Returns
    -------
    array
        tSZ-CIB correlation term.

    """
    # CIB
    CIB_nu_1 = fg_class_instance.CIB[0].output(sample_params)
    CIB_nu_2 = fg_class_instance.CIB[1].output(sample_params)

    # tSZ
    tSZ_nu_1 = fg_class_instance.tSZ[0].output(sample_params)
    tSZ_nu_2 = fg_class_instance.tSZ[1].output(sample_params)

    # CIB x tSZ
    CIB_x_tSZ = jnp.sqrt(CIB_nu_1 * tSZ_nu_2) + jnp.sqrt(CIB_nu_2 * tSZ_nu_1)

    # Complete foreground contribution and mask down
    fg_pow = (
        -1.0
        * fg_class_instance.full_mask
        * sample_params[fg_class_instance.amp_param]
        * CIB_x_tSZ
    )
    return fg_pow


@_CIBtSZCorrelationGeometricMean_output.defjvp
def _CIBtSZCorrelationGeometricMean_output_jvp(fg_class_instance, primals, tangents):
    """
    Hand-defined derivative of CIB-tSZ correlation term output function.
    See also: candl.transformations.common._CIBtSZCorrelationGeometricMean_output, jax.custom_jvp
    """
    # Process input into regular dictionary
    (full_pars,) = primals
    (pars_dot,) = tangents

    # Don't pass on Dl array - it's unnecessary
    pars = deepcopy(full_pars)
    if "Dl" in pars:
        del pars["Dl"]

    # Pass to original function for values
    ans = fg_class_instance.output(pars)

    # Calculate derivatives

    # xi
    xi_deriv = ans / pars["TT_tSZ_CIB_Corr_Amp"]

    # A_tSZ
    tSZ_amp_deriv_term1 = (
        0.5
        * jacfwd(fg_class_instance.tSZ[1].output)(pars)["TT_tSZ_Amp"]
        * jnp.sqrt(
            fg_class_instance.CIB[0].output(pars)
            / fg_class_instance.tSZ[1].output(pars)
        )
    )
    tSZ_amp_deriv_term2 = (
        0.5
        * jacfwd(fg_class_instance.tSZ[0].output)(pars)["TT_tSZ_Amp"]
        * jnp.sqrt(
            fg_class_instance.CIB[1].output(pars)
            / fg_class_instance.tSZ[0].output(pars)
        )
    )
    tSZ_amp_deriv = -pars["TT_tSZ_CIB_Corr_Amp"] * (
        tSZ_amp_deriv_term1 + tSZ_amp_deriv_term2
    )
    tSZ_amp_deriv = tSZ_amp_deriv.at[
        np.invert(np.asarray(fg_class_instance.full_mask, dtype=bool))
    ].set(0.0)

    # A_CIB
    CIB_amp_deriv_term1 = (
        0.5
        * jacfwd(fg_class_instance.CIB[1].output)(pars)["TT_CIBClustering_Amp"]
        * jnp.sqrt(
            fg_class_instance.tSZ[0].output(pars)
            / fg_class_instance.CIB[1].output(pars)
        )
    )
    CIB_amp_deriv_term2 = (
        0.5
        * jacfwd(fg_class_instance.CIB[0].output)(pars)["TT_CIBClustering_Amp"]
        * jnp.sqrt(
            fg_class_instance.tSZ[1].output(pars)
            / fg_class_instance.CIB[0].output(pars)
        )
    )
    CIB_amp_deriv = -pars["TT_tSZ_CIB_Corr_Amp"] * (
        CIB_amp_deriv_term1 + CIB_amp_deriv_term2
    )
    CIB_amp_deriv = CIB_amp_deriv.at[
        np.invert(np.asarray(fg_class_instance.full_mask, dtype=bool))
    ].set(0.0)

    # beta
    beta_deriv_term1 = (
        0.5
        * jacfwd(fg_class_instance.CIB[1].output)(pars)["TT_CIBClustering_Beta"]
        * jnp.sqrt(
            fg_class_instance.tSZ[0].output(pars)
            / fg_class_instance.CIB[1].output(pars)
        )
    )
    beta_deriv_term2 = (
        0.5
        * jacfwd(fg_class_instance.CIB[0].output)(pars)["TT_CIBClustering_Beta"]
        * jnp.sqrt(
            fg_class_instance.tSZ[1].output(pars)
            / fg_class_instance.CIB[0].output(pars)
        )
    )
    beta_deriv = -pars["TT_tSZ_CIB_Corr_Amp"] * (beta_deriv_term1 + beta_deriv_term2)
    beta_deriv = beta_deriv.at[
        np.invert(np.asarray(fg_class_instance.full_mask, dtype=bool))
    ].set(0.0)

    ans_dot = (
        xi_deriv * pars_dot["TT_tSZ_CIB_Corr_Amp"]
        + tSZ_amp_deriv * pars_dot["TT_tSZ_Amp"]
        + CIB_amp_deriv * pars_dot["TT_CIBClustering_Amp"]
        + beta_deriv * pars_dot["TT_CIBClustering_Beta"]
    )

    return ans, ans_dot


class CIBtSZCorrelationGeometricMean(candl.transformations.abstract_base.Foreground):
    """
    Simple correlation term between power-law CIB and template tSZ modules above with a free amplitude..
    Note that the sign is defined such that a positive correlation parameter leads to a reduction of power at 150GHz.
    Used by SPT-3G 2018 TT/TE/EE implementation.
    Note that the meat has been taken out of the output method in order to allow for differentiability;
    auto-diff struggles with this module due to the square-roots, hence hand-defined defined custom derivate rules.
    Thanks to Marco Bonici for the pointer.

    Attributes
    --------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    spec_order : array (str)
        Identifiers of spectra in the order in which spectra are handled in the long data vector.
    ell_ref : int
        Reference ell for normalisation.
    spec_mask : array (int)
        Masks which spectra of the long data vector are affected by the transformation.
    affected_specs : list (str)
        List of the spectra to apply this foreground to.
    full_mask : array (int)
        Masks which elements of the long data vector are affected by the transformation.
    affected_specs_ix : list (int)
        Indices of affected spectra
    N_spec : int
        The total number of spectra in the long data vector.
    amp_param : str
        The name of the amplitude parameter.
    CIB : candl.transformations.abstract_base.transformation
        CIB module.
    tSZ : candl.transformations.abstract_base.transformation
        tSZ module.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * link_transformation_module_CIB (str) : Class of the CIB module to scan initialised transformations for.
    * link_transformation_module_tSZ (str) : Class of the tSZ module to scan initialised transformations for.
    * amp_param (str) : Name of the free amplitude parameter.
    * affected_specs (list) : List of the spectra to apply this foreground to.

    Examples
    ----------------

    Example yaml block to add tSZ-CIB correlation power::

        - Module: "common.CIBtSZCorrelationGeometricMean"
          link_transformation_module_CIB: "common.CIBClustering"
          link_transformation_module_tSZ: "common.tSZTemplateForeground"
          amp_param: "TT_tSZ_CIB_Corr_Amp"
          affected_specs: ["TT 90x90", "TT 90x150", "TT 90x220", "TT 150x150", "TT 150x220", "TT 220x220"]
    """

    def __init__(
        self,
        ells,
        spec_order,
        affected_specs,
        amp_param,
        link_transformation_module_CIB,
        link_transformation_module_tSZ,
        descriptor="CIB-tSZ correlation",
    ):
        """
        Initialise a new instance of the CIBtSZCorrelationGeometricMean class.

        Arguments
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        amp_param : str
            The name of the amplitude parameter.
        spec_order : array (str)
            Identifiers of spectra in the order in which spectra are handled in the long data vector.
        link_transformation_module_CIB : abstract_base.Transformation
            CIB transformation
        link_transformation_module_tSZ : abstract_base.Transformation
            tSZ transformation

        Output
        --------------
        CIBtSZCorrelationGeometricMean instance.
        """

        super().__init__(
            ells=ells,
            ell_ref=0,  # reference ell not required
            descriptor=descriptor,
            param_names=[amp_param],
        )

        self.amp_param = amp_param
        self.spec_order = spec_order
        self.affected_specs = affected_specs
        self.spec_mask = jnp.asarray(
            [spec in self.affected_specs for spec in self.spec_order]
        )
        self.N_spec = len(self.spec_mask)

        # Turn spectrum mask into a full mask
        self.full_mask = jnp.asarray(
            jnp.repeat(self.spec_mask, len(self.ells)), dtype=float
        )

        # Make 2 copies of the CIB and tSZ classes and modify their effective frequencies
        # (need to have nu_1-only and nu_2-only versions of each).
        self.CIB = [
            deepcopy(link_transformation_module_CIB),
            deepcopy(link_transformation_module_CIB),
        ]
        self.tSZ = [
            deepcopy(link_transformation_module_tSZ),
            deepcopy(link_transformation_module_tSZ),
        ]

        for i, (CIB_freq_pair, tSZ_freq_pair) in enumerate(
            zip(
                link_transformation_module_CIB.freq_info,
                link_transformation_module_tSZ.freq_info,
            )
        ):
            self.CIB[0].freq_info[i] = [CIB_freq_pair[0], CIB_freq_pair[0]]
            self.CIB[1].freq_info[i] = [CIB_freq_pair[1], CIB_freq_pair[1]]
            self.tSZ[0].freq_info[i] = [tSZ_freq_pair[0], tSZ_freq_pair[0]]
            self.tSZ[1].freq_info[i] = [tSZ_freq_pair[1], tSZ_freq_pair[1]]

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.
        Direct call to _CIBtSZCorrelationGeometricMean_output.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        return _CIBtSZCorrelationGeometricMean_output(self, sample_params)

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform spectrum by adding foreground component (result of output method).

        Arguments
        --------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class FGSpectraInterfaceFactorizedCrossSpectrum(
    candl.transformations.abstract_base.Foreground
):
    """
    Wrapper for SO's FGSpectra FactorizedCrossSpectrum (https://github.com/simonsobs/fgspectra/tree/main) with a free amplitude.

    Attributes
    --------------
    fgspectra_sed : Instance of a fgspectra.frequency class
        Used by FGSpectra for SED.
    fgspectra_sed_args : list (str)
        Names of sampled parameters that need to be passed to the SED instance.
    fgspectra_sed_args_fixed : dictionary of string : float
        Names and values of fixed parameters to be passed to the SED instance.
    fgspectra_cl : Instance of a fgspectra.power class
        Used by FGSpectra for Cls.
    fgspectra_cl_args : list (str))
        Names of sampled parameters that need to be passed to the Cl instance.
    fgspectra_cl_args_fixed : dictionary of string : float
        Names and values of fixed parameters to be passed to the Cl instance.
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    spec_order : array (str)
        Identifiers of spectra in the order in which spectra are handled in the long data vector.
    freq_info : list
        List of lists, where each sublist contains the two effective frequencies for a given spectrum.
    ell_ref : int
        Reference ell for normalisation.
    spec_mask : array (int)
        Masks which spectra of the long data vector are affected by the transformation.
    affected_specs : list (str)
        List of the spectra to apply this foreground to.
    affected_specs_ix : list (int)
        Indices of affected spectra
    N_spec : int
        The total number of spectra in the long data vector.
    amp_param : str
        The name of the amplitude parameter.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Notes
    -----------

    User required arguments in data set yaml file:

    * fg_spectra_sed (str) : Name of fgspectra.frequency class to use for SED.
    * fg_spectra_sed_args (list) : Names of sampled parameters that need to be passed to the SED instance.
    * fg_spectra_sed_args_fixed (dict) : Names and values of fixed parameters to be passed to the SED instance.
    * fg_spectra_cl (str) : Name of fgspectra.power class to use for Cls.
    * fg_spectra_cl_args (list) : Names of sampled parameters that need to be passed to the Cl instance.
    * fg_spectra_cl_args_fixed (dict) : Names and values of fixed parameters to be passed to the Cl instance.
    * amp_param (str) : The name of the amplitude parameter.
    * affected_specs (list) : List of the spectra to apply this foreground to.
    * effective_frequencies (str) : Keyword to look for in effective frequencies yaml file.

    Examples
    ----------------

    Example yaml block::

        - Module: "common.FGSpectraInterfaceFactorizedCrossSpectrum"
          fgspectra_sed: "ThermalSZ"
          fgspectra_sed_args: []
          fgspectra_sed_args_fixed: {nu_0: 150.0}
          fgspectra_cl: "tSZ_150_bat"
          fgspectra_cl_args: []
          fgspectra_cl_args_fixed: {ell_0: 3000}
          amp_param: "FGSpec_amp"
          affected_specs: ["TT 90x90", "TT 150x150", "TT 220x220"]
          effective_frequencies: "tSZ"
    """

    def __init__(
        self,
        ells,
        fgspectra_sed,
        fgspectra_sed_args,
        fgspectra_sed_args_fixed,
        fgspectra_cl,
        fgspectra_cl_args,
        fgspectra_cl_args_fixed,
        amp_param,
        freq_info,
        spec_order,
        affected_specs,
        descriptor="FGSpectra Interface",
    ):
        """
        Initialise a new instance of the FGSpectraInterfaceFactorizedCrossSpectrum class.

        Arguments
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        fgspectra_sed : str
            Name of fgspectra.frequency class to use for SED.
        fgspectra_sed_args : list (str)
            Names of sampled parameters that need to be passed to the SED instance.
        fgspectra_sed_args_fixed : dictionary of string : float
            Names and values of fixed parameters to be passed to the SED instance.
        fgspectra_cl : str
            Name of fgspectra.power class to use for Cls.
        fgspectra_cl_args : list (str))
            Names of sampled parameters that need to be passed to the Cl instance.
        fgspectra_cl_args_fixed : dictionary of string : float
            Names and values of fixed parameters to be passed to the Cl instance.
        amp_param : str
            The name of the amplitude parameter.
        freq_info : list
            List of lists, where each sublist contains the two effective frequencies for a given spectrum.
        spec_order : array (str)
            Identifiers of spectra in the order in which spectra are handled in the long data vector.
        affected_specs : list (str)
            List of the spectra to apply this foreground to.
        descriptor : str (optional)
            A short descriptor.

        Output
        --------------
        FGSpectraInterfaceFactorizedCrossSpectrum instance.
        """

        super().__init__(
            ells=ells,
            ell_ref=(
                fgspectra_cl_args_fixed["ell_0"]
                if "ell_0" in fgspectra_cl_args_fixed
                else None
            ),
            descriptor=descriptor,
            param_names=list(
                np.unique([amp_param] + fgspectra_sed_args + fgspectra_cl_args)
            ),
        )

        # Initialise fgspectra.cross.FactorizedCrossSpectrum instance
        fgc = importlib.import_module("fgspectra.cross")
        fgf = importlib.import_module("fgspectra.frequency")
        fgp = importlib.import_module("fgspectra.power")
        self.fgspectra_instance = fgc.FactorizedCrossSpectrum(
            eval(f"fgf.{fgspectra_sed}()"), eval(f"fgp.{fgspectra_cl}()")
        )

        self.fgspectra_sed_args = fgspectra_sed_args
        self.fgspectra_sed_args_fixed = fgspectra_sed_args_fixed

        self.fgspectra_cl_args = fgspectra_cl_args
        self.fgspectra_cl_args_fixed = fgspectra_cl_args_fixed

        self.amp_param = amp_param

        # Saving spectra order etc.
        self.affected_specs = affected_specs
        self.spec_order = spec_order
        self.N_spec = len(self.spec_order)
        self.freq_info = [
            np.array(f) for f in freq_info
        ]  # FGspectra needs these as arrays

        # Generate boolean mask of affected specs
        self.spec_mask = np.zeros(
            len(spec_order)
        )  # Generate as np array for easier item assignment
        for i, spec in enumerate(self.spec_order):
            if spec in self.affected_specs:
                self.spec_mask[i] = 1
        self.spec_mask = self.spec_mask == 1
        self.spec_mask = jnp.array(self.spec_mask)
        self.affected_specs_ix = [ix[0] for ix in jnp.argwhere(self.spec_mask)]

    # Note: FGSpectra code is not necessarily jit safe
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # Grab parameter dict for FGSpectra
        sed = self.fgspectra_sed_args_fixed
        cl = self.fgspectra_cl_args_fixed
        for p in list(sample_params.keys()):
            if p in self.fgspectra_sed_args:
                sed[p] = sample_params[p]
            if p in self.fgspectra_cl_args:
                cl[p] = sample_params[p]
        cl["ell"] = self.ells

        # Loop over spectra and slot into array
        full_fg_array = jnp.zeros(len(self.spec_order) * len(self.ells))
        for ix in self.affected_specs_ix:
            sed["nu"] = self.freq_info[ix]
            full_fg_array = jax_optional_set_element(
                full_fg_array,
                jnp.arange(ix * len(self.ells), (ix + 1) * len(self.ells)),
                self.fgspectra_instance(sed, cl)[0, 1],
            )  # These are typically in Dl already, I believe

        return sample_params[self.amp_param] * full_fg_array

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sample_params : dict
            A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
            The transformed spectrum in Dl.
        """

        return Dls + self.output(sample_params)


# --------------------------------------#
# CALIBRATION
# --------------------------------------#


class CalibrationSingleScalar(candl.transformations.abstract_base.Calibration):
    """
    Simple calibration model for spectra.
    Scales all model spectra by :math:`1/X`, where :math:`X` is specified as `cal_param`.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    transform :
        transforms an input spectrum.

    Attributes
    --------------
    descriptor : str
        A short descriptor.
    cal_param : str
        Name of the calibration parameter.
    par_names : list
        Names of parameters involved in transformation.
    """

    def __init__(self, cal_param, descriptor="Calibration (single number"):
        super().__init__(ells=None, descriptor=descriptor, param_names=[cal_param])
        self.cal_param = cal_param

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
           The spectrum to transform in Dl.
        sample_params : dict
           A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
           The transformed spectrum in Dl.
        """

        return Dls / sample_params[self.cal_param]


class CalibrationSingleScalarSquared(candl.transformations.abstract_base.Calibration):
    """
    Simple calibration model for spectra.
    Scales all model spectra by :math:`1/X^2`, where :math:`X` is specified as `cal_param`.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    transform :
        transforms an input spectrum.

    Attributes
    --------------
    descriptor : str
        A short descriptor.
    cal_param : str
        Name of the calibration parameter.
    par_names : list
        Names of parameters involved in transformation.
    """

    def __init__(self, cal_param, descriptor="Calibration (single number"):
        super().__init__(ells=None, descriptor=descriptor, param_names=[cal_param])
        self.cal_param = cal_param

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
           The spectrum to transform in Dl.
        sample_params : dict
           A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
           The transformed spectrum in Dl.
        """

        return Dls / sample_params[self.cal_param] ** 2.0


class CalibrationAuto(candl.transformations.abstract_base.IndividualCalibration):
    """
    Calibration model that divides each spectrum by the series of specified parameters.
    Scales model spectra by :math:`1/\Pi_i X_i`, where :math:`X_i` are specified in the spec_param_dict.
    Contributed by Etienne Camphuis.

    Attributes
    --------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    spec_param_dict : dict
        A dictionary with keys that are spectrum identifiers and values that are lists of the nuisance parameter names
        that are used to transform this spectrum.
    spec_order : list
        Order of the spectra in the long data vector.
    N_specs : int
        Total number of spectra.
    affected_specs : list (str)
            List of the spectra to apply this foreground to.
    spec_mask : array (int)
        Masks which parts of the long data vector are affected by the transformation.
    affected_specs_ix : list (int)
        Indices in spectra_order of spectra the transformation is applied to.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file

    * spec_param_dict (dict) : A dictionary with keys that are spectrum identifiers and values that are lists of the nuisance parameter names that are used to transform this spectrum.


    Examples
    ----------------

    Example yaml block to calibrate TT 90GHz and 150GHz spectra in two steps (internal and external)::

        - Module: "common.CalibrationAuto"
          spec_param_dict:
            TT 90x90: ["cal_ext", "cal_ext", "cal_rel", "cal_rel"]
            TT 90x150: ["cal_ext", "cal_ext", "cal_rel"]
            TT 150x150: ["cal_ext", "cal_ext"]
    """

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
           The spectrum to transform in Dl.
        sample_params : dict
           A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
           The transformed spectrum in Dl.
        """

        # amplitude part
        cal_vals = jnp.ones(len(self.spec_order))
        for ix in self.affected_specs_ix:
            this_cal_val = 1.0
            for par in self.spec_param_dict[self.spec_order[ix]]:
                this_cal_val *= sample_params[par]
            cal_vals = jax_optional_set_element(cal_vals, ix, this_cal_val)
        tiled_cal_vals = jnp.repeat(cal_vals, len(self.ells))

        return Dls / tiled_cal_vals

    @partial(jit, static_argnums=(0,))
    def get_cal_vec(self, sample_params):
        """
        Shortcut to access calibration vector.
        See also: transformation()
        """

        # amplitude part
        cal_vals = jnp.ones(len(self.spec_order))
        for ix in self.affected_specs_ix:
            this_cal_val = 1.0
            for par in self.spec_param_dict[self.spec_order[ix]]:
                this_cal_val *= sample_params[par]
            cal_vals = jax_optional_set_element(cal_vals, ix, this_cal_val)

        return cal_vals


class CalibrationCross(candl.transformations.abstract_base.IndividualCalibration):
    """
    Calibration model for summed spectra, e.g. for TE_90x150: :math:` 0.5 * ( T_{90}xE_{150} + E_{90}xT_{150} )`.
    Scales model spectra by :math:`1/[0.5*(X*Y+WV)]`, where :math:`X,Y,W,V` are specified in the spec_param_dict (most likely want Tcal and/or Ecal in there).
    Reduces to CalibrationAuto if parameters are repeated appropriately.
    Used by SPT-3G 2018 TT/TE/EE implementation.

    Attributes
    --------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    spec_param_dict : dict
        A dictionary with keys that are spectrum identifiers and values that are lists of the nuisance parameter names
        that are used to transform this spectrum.
    spec_order : list
        Order of the spectra in the long data vector.
    N_specs : int
        Total number of spectra.
    affected_specs : list (str)
            List of the spectra to apply this foreground to.
    spec_mask : array (int)
        Masks which parts of the long data vector are affected by the transformation.
    affected_specs_ix : list (int)
        Indices in spectra_order of spectra the transformation is applied to.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file

    * spec_param_dict (dict) : A dictionary with keys that are spectrum identifiers and values that are lists of the nuisance parameter names that are used to transform this spectrum.

    Examples
    ----------------

    Example yaml block to calibrate TE spectra that are the sum of the two (TE/ET) crosses::

        - Module: "common.CalibrationCross"
          spec_param_dict:
            TE 90x90: ["Tcal90", "Ecal90", "Tcal90", "Ecal90"]
            TE 90x150: [ "Tcal90", "Ecal150", "Tcal150", "Ecal90" ]
            TE 90x220: [ "Tcal90", "Ecal220", "Tcal220", "Ecal90" ]
            TE 150x150: [ "Tcal150", "Ecal150", "Tcal150", "Ecal150" ]
            TE 150x220: [ "Tcal150", "Ecal220", "Tcal220", "Ecal150" ]
            TE 220x220: [ "Tcal220", "Ecal220", "Tcal220", "Ecal220" ]
    """

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
           The spectrum to transform in Dl.
        sample_params : dict
           A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
           The transformed spectrum in Dl.
        """

        # amplitude part
        cal_vals = jnp.ones(len(self.spec_order))
        for ix in self.affected_specs_ix:
            this_cal_val = (
                sample_params[self.spec_param_dict[self.spec_order[ix]][0]]
                * sample_params[self.spec_param_dict[self.spec_order[ix]][1]]
            )
            this_cal_val += (
                sample_params[self.spec_param_dict[self.spec_order[ix]][2]]
                * sample_params[self.spec_param_dict[self.spec_order[ix]][3]]
            )
            cal_vals = jax_optional_set_element(cal_vals, ix, this_cal_val / 2.0)
        tiled_cal_vals = jnp.repeat(cal_vals, len(self.ells))

        return Dls / tiled_cal_vals

    @partial(jit, static_argnums=(0,))
    def get_cal_vec(self, sample_params):
        """
        Shortcut to access calibration vector.
        See also: transformation()
        """

        # amplitude part
        cal_vals = jnp.ones(len(self.spec_order))
        for ix in self.affected_specs_ix:
            this_cal_val = (
                sample_params[self.spec_param_dict[self.spec_order[ix]][0]]
                * sample_params[self.spec_param_dict[self.spec_order[ix]][1]]
            )
            this_cal_val += (
                sample_params[self.spec_param_dict[self.spec_order[ix]][2]]
                * sample_params[self.spec_param_dict[self.spec_order[ix]][3]]
            )
            cal_vals = jax_optional_set_element(cal_vals, ix, this_cal_val / 2.0)

        return cal_vals


class PolarisationCalibration(candl.transformations.abstract_base.Calibration):
    """
    Simple calibration model for spectra.
    Scales all TE by :math:`X` and all EE by :math:`X^2`, where :math:`X` is specified as `cal_param`.
    Used by ACT DR4 likelihood implementation.

    Attributes
    --------------
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    cal_param : str
        Name of the calibration parameter.
    par_names : list
        Names of parameters involved in transformation.
    spec_order : array (str)
        Identifiers of spectra in the order in which spectra are handled in the long data vector.
    TE_affected_specs_ix : list
        List of indices of spectra that get a yp factor
    EE_affected_specs_ix : list
        List of indices of spectra that get a yp^2 factor

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    transform :
        transforms an input spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * cal_param (str) : Name of the calibration parameter.

    Examples
    ----------------

    Example yaml block to calibrate TE and EE spectra::

        - Module: "common.PolarisationCalibration"
          cal_param: "yp"
          descriptor: "Calibration"
    """

    def __init__(
        self, ells, cal_param, spec_order, descriptor="Calibration (single number"
    ):
        super().__init__(ells=ells, descriptor=descriptor, param_names=[cal_param])
        self.spec_order = spec_order
        self.cal_param = cal_param

        # Generate boolean mask of affected specs
        self.affected_specs = [spec for spec in self.spec_order if "E" in spec[:2]]
        self.spec_mask = np.zeros(
            len(spec_order)
        )  # Generate as np array for easier item assignment
        for i, spec in enumerate(self.spec_order):
            if spec in self.affected_specs:
                if spec[:2] == "TE":
                    self.spec_mask[i] = 1
                elif spec[:2] == "EE":
                    self.spec_mask[i] = 2
        self.TE_affected_specs_ix = [ix[0] for ix in jnp.argwhere(self.spec_mask == 1)]
        self.EE_affected_specs_ix = [ix[0] for ix in jnp.argwhere(self.spec_mask == 2)]

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
           The spectrum to transform in Dl.
        sample_params : dict
           A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
           The transformed spectrum in Dl.
        """

        # amplitude part
        cal_vals = jnp.ones(len(self.spec_order))
        for ix in self.TE_affected_specs_ix:
            cal_vals = jax_optional_set_element(
                cal_vals, ix, sample_params[self.cal_param]
            )
        for ix in self.EE_affected_specs_ix:
            cal_vals = jax_optional_set_element(
                cal_vals, ix, sample_params[self.cal_param] ** 2.0
            )
        tiled_cal_vals = jnp.repeat(cal_vals, len(self.ells))

        return Dls * tiled_cal_vals


class PolarisationCalibrationDivision(PolarisationCalibration):
    """
    Simple calibration model for spectra. Same as PolarisationCalibration, but divides, rather than multiplies by the calibration parameter.
    """

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
           The spectrum to transform in Dl.
        sample_params : dict
           A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
           The transformed spectrum in Dl.
        """

        # amplitude part
        cal_vals = jnp.ones(len(self.spec_order))
        for ix in self.TE_affected_specs_ix:
            cal_vals = jax_optional_set_element(
                cal_vals, ix, sample_params[self.cal_param]
            )
        for ix in self.EE_affected_specs_ix:
            cal_vals = jax_optional_set_element(
                cal_vals, ix, sample_params[self.cal_param] ** 2.0
            )
        tiled_cal_vals = jnp.repeat(cal_vals, len(self.ells))

        return Dls / tiled_cal_vals


# --------------------------------------#
# TRANSFORMATIONS INVOLVING dCl/dl DERIVATIVE
# --------------------------------------#


class SuperSampleLensing(candl.transformations.abstract_base.Transformation):
    """
    Super sample lensing.
    Following Equation 32 in Manzotti, Hu, Benoit-Levy 2014 (https://arxiv.org/pdf/1401.7992.pdf).
    The addition in Cl space is:

    .. math::
        - \\frac{\\partial\\ell^2 C_\\ell^{XY}}{\\partial \\ln{\\ell}} \\frac{\\kappa}{\\ell^2}

    or:

    .. math::

        - \\frac{\\kappa}{\\ell^2} \\frac{\\partial}{\\partial\\ln{\\ell}} (\\ell^2 C_\\ell)
        = -\\kappa(\\ell*\\frac{\\partial C_\\ell}{\\partial\\ell} + 2 C_\\ell)

    Used by SPT-3G 2018 TT/TE/EE implementation.

    Attributes
    --------------
    ells : array (float)
            The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    kappa_param : str
        Name of the kappa parameter to be used (i.e. the mean lensing covergence across the field).
    long_ells : array (float)
        Long vector of concatenated theory ells.
    operation_hint : str
        Type of the 'transform' operation: 'additive'.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive SSL contribution.
    transform :
        Returns a transformed spectrum.

    Notes
    ----------------

    User required arguments in data set yaml file:

    * kappa_param (str) : Name of the kappa parameter to be used.

    Examples
    ----------------

    Example yaml block to add SSL::

        - Module: "common.SuperSampleLensing"
          kappa_param: "Kappa"
    """

    def __init__(
        self,
        ells,
        long_ells,
        kappa_param,
        descriptor="Super-Sample Lensing",
    ):
        """
        Initialise the SuperSamleLensing transformation.

        Arguments
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        long_ells : array (float)
            Long vector of concatenated theory ells.
        kappa_param : str
            Name of the kappa parameter to be used (i.e. the mean lensing covergence across the field).
        descriptor : str
            A short descriptor.

        Returns
        --------------
        Transformation
            A new instance of the SuperSampleLensing class.
        """

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=[kappa_param],
            operation_hint="additive",
        )

        self.kappa_param = kappa_param
        self.long_ells = long_ells

    @partial(jit, static_argnums=(0,))
    def output(self, Dls, sample_params):
        """
        Return SSL contribution.
        Sightly different signature to foregrounds (Dl dependent), but intended to be accessed through
        transformation() only.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            SSL contribution.
        """

        # Looks like annoying back and forth, but calculation is more straightforward in Cl space!
        # Needs to slice up theory vector to avoid spikes in derivatives
        Cl_deriv = jnp.concatenate(
            [
                jnp.gradient(
                    Dls[i * len(self.ells) : (i + 1) * len(self.ells)]
                    * 2
                    * jnp.pi
                    / (self.ells * (self.ells + 1))
                )
                for i in range(int(len(self.long_ells) / len(self.ells)))
            ]
        )  # convert to Cls and calculate derivative
        ssl_correction = (
            self.long_ells
            * Cl_deriv
            * self.long_ells
            * (self.long_ells + 1)
            / (2 * jnp.pi)
        )  # extra factor of ell and move to Dl space
        ssl_correction += 2 * Dls  # add second term in brackets
        ssl_correction *= -sample_params[
            self.kappa_param
        ]  # fix sign and scale by amplitude parameter

        return ssl_correction

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sample_params : dict
            A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
            The transformed spectrum in Dl.
        """

        return Dls + self.output(Dls, sample_params)


class AberrationCorrection(candl.transformations.abstract_base.Transformation):
    """
    AberrationCorrection.
    Following Equation 23 in Jeong et al. 2013 (https://arxiv.org/pdf/1309.2285.pdf).
    Note that this is a fixed transformation and does not depend on any nuisance parameters.

    The addition in Cl space is:

    .. math::

        - AC * \\ell * \\frac{\\partial C_\\ell}{\\partial \\ell}

    where AC is the aberration coefficient with :math:`AC = \\beta \\langle\\cos{\\theta}\\rangle`.
    Here :math:`\\beta` is the speed in units of c (typically ~0.00123) and :math:`\\langle\\cos{\\theta}\\rangle` is the average direction (w.r.t. the observed field).

    Used by SPT-3G 2018 TT/TE/EE implementation.

    Attributes
    --------------
    ells : array (float)
        The ell range the transformation acts on.
    aberration_coefficient : float
        Product of the beta and cos(theta) terms.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    long_ells : array (float)
        Long vector of concatenated theory ells.
    operation_hint : str
        Type of the 'transform' operation: 'additive'.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive aberration contribution.
    transform :
        Returns a transformed spectrum.

    Examples
    ----------------

    Example yaml block to add Aberration::

        - Module: "common.SuperSampleLensing"
          aberration_coefficient: 0.0001
    """

    def __init__(
        self,
        ells,
        long_ells,
        aberration_coefficient,
        descriptor="Aberration Correction",
    ):
        """
        Initialise the AberrationCorrection transformation.

        Arguments
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        long_ells : array (float)
            Long vector of concatenated theory ells.
        aberration_coefficient : float
            Product of the beta and cos(theta) terms.
        descriptor : str
            A short descriptor.

        Returns
        --------------
        Transformation
            A new instance of the AberrationCorrection class.
        """

        super().__init__(ells=ells, descriptor=descriptor, operation_hint="additive")
        self.aberration_coefficient = aberration_coefficient
        self.long_ells = long_ells

    @partial(jit, static_argnums=(0,))
    def output(self, Dls):
        """
        Return the aberration correction.
        Sightly different signature to foregrounds (only Dl dependent), but intended to be accessed through
        transformation() only.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Aberration correcation.
        """

        # Looks like annoying back and forth, but calculation is actually nicer to do with Cl derivative!
        # Needs to slice up theory vector to avoid spikes in derivatives
        Cl_deriv = jnp.concatenate(
            [
                jnp.gradient(
                    Dls[i * len(self.ells) : (i + 1) * len(self.ells)]
                    * 2
                    * jnp.pi
                    / (self.ells * (self.ells + 1))
                )
                for i in range(int(len(self.long_ells) / len(self.ells)))
            ]
        )  # convert to Cls and calculate derivative
        ab_correction = (
            Cl_deriv * self.long_ells * (self.long_ells + 1) / (2 * jnp.pi)
        )  # move to Dl space
        ab_correction *= self.long_ells
        ab_correction *= -self.aberration_coefficient
        return ab_correction

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.
        Note that sample_params is never accessed, but for uniformity across transformation() methods still included.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sample_params : dict
            A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
            The transformed spectrum in Dl.
        """

        return Dls + self.output(Dls)


# --------------------------------------#
# FREQUENCY HELPERS
# --------------------------------------#


# Dust frequency scaling (modified black body) including integral over the band pass
@jit
def dust_frequency_scaling_bandpass(
    beta, Tdust, nu_0_dust, nu_spacing, nu_vals, bandpass_vals, thermo_conv
):
    """
    Modified black body frequency scaling with the integral over the bandpass.
    Band pass information is expanded out (rather than passing an instance of BandPass) to make @jit easier.
    Frequency power law index is 3 + beta.
    Following BK_Planck likelihood.

    Arguments
    --------------
    beta : float
        Spectral index
    Tdust : float
        Dust temperature.
    nu_0_dust : float
        Reference frequency.
    nu_spacing : float
        Frequency spacing of the band pass data.
    nu_spacing : array (float)
        Frequency values of band pass measurements.
    bandpass_vals : array (float)
        Band pass measurements.
    thermo_conv : float
        Thermodynamic conversion for the reference frequency.

    Returns
    --------------
    float :
        Frequency scaling
    """

    # Calculate grey-body integrals
    grey_body_int = jnp.sum(
        nu_spacing
        * bandpass_vals
        * nu_vals ** (3 + beta)
        / (jnp.exp(nu_vals * candl.constants.GHz_KELVIN / Tdust) - 1)
    )

    grey_body_norm = nu_0_dust ** (3 + beta) / (
        jnp.exp(nu_0_dust * candl.constants.GHz_KELVIN / Tdust) - 1
    )

    # Put everything together including thermodynamic conversion
    f_dust = (grey_body_int / grey_body_norm) / thermo_conv

    return f_dust


# Simple Dust Frequency Scaling, ignores correction to band pass centre from changing SED
@jit
def dust_frequency_scaling(
    beta: jnp.float64, Tdust: jnp.float64, nu_0_dust: jnp.float64, nu: jnp.float64
) -> jnp.float64:
    """
    Modified black body frequency scaling.
    Based on code shared by Christian Reichardt - thank you!

    Arguments
    --------------
    beta : float
        Spectral index
    Tdust : float
        Dust temperature.
    nu_0_dust : float
        Reference frequency.
    nu : float
        Requested frequency.

    Returns
    --------------
    float :
        Frequency scaling
    """

    fdust = (nu / nu_0_dust) ** beta
    fdust *= black_body(nu, nu_0_dust, Tdust) / black_body_deriv(
        nu, nu_0_dust, candl.constants.T_CMB
    )

    return fdust


# tSZ Frequency Scaling
@jit
def tSZ_frequency_scaling(
    nu: jnp.float64, nu0: jnp.float64, T: jnp.float64
) -> jnp.float64:
    """
    tSZ frequency scaling.
    Based on code shared by Christian Reichardt - thank you!

    Arguments
    --------------
    T : float
        CMB temperature.
    nu_0_dust : float
        Reference frequency.
    nu : float
        Requested frequency.

    Returns
    --------------
    float :
        Frequency scaling
    """

    x0 = candl.constants.GHz_KELVIN * nu0 / T
    x = candl.constants.GHz_KELVIN * nu / T
    tSZfac0 = x0 * (jnp.exp(x0) + 1) / (jnp.exp(x0) - 1) - 4
    tSZfac = x * (jnp.exp(x) + 1) / (jnp.exp(x) - 1) - 4
    tSZfac = tSZfac / tSZfac0

    return tSZfac


@jit
def black_body(nu: jnp.float64, nu0: jnp.float64, T: jnp.float64) -> jnp.float64:
    """
    Black body function, normalised to 1 at nu0.
    Based on code shared by Christian Reichardt - thank you!

    Arguments
    --------------
    T : float
        Temperature.
    nu0 : float
        Reference frequency.
    nu : float
        Requested frequency.

    Returns
    --------------
    float :
        Frequency scaling
    """

    fac = (nu / nu0) ** 3
    fac *= (jnp.exp(candl.constants.GHz_KELVIN * nu0 / T) - 1) / (
        jnp.exp(candl.constants.GHz_KELVIN * nu / T) - 1
    )

    return fac


# Derivative of Planck function normalised to 1 at nu0
@jit
def black_body_deriv(nu: jnp.float64, nu0: jnp.float64, T: jnp.float64) -> jnp.float64:
    """
    Derivative of black body function, normalised to 1 at nu0.
    Based on code shared by Christian Reichardt - thank you!

    Arguments
    --------------
    T : float
        Temperature.
    nu0 : float
        Reference frequency.
    nu : float
        Requested frequency.

    Returns
    --------------
    float :
        Frequency scaling
    """

    x0 = candl.constants.GHz_KELVIN * nu0 / T
    x = candl.constants.GHz_KELVIN * nu / T
    dBdT0 = x0**4 * jnp.exp(x0) / (jnp.exp(x0) - 1) ** 2
    dBdT = x**4 * jnp.exp(x) / (jnp.exp(x) - 1) ** 2
    dBdT = dBdT / dBdT0

    return dBdT

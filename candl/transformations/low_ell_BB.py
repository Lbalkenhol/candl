"""
Galactic Foreground transformations with band pass integrals intended for BB analysis.

Foreground Classes:
--------
GalacticDustBandPass
SynchrotronBandPass
GalacticDustSynchrotronCorrelationBandPass
GalacticDustFixedAlphaBandPass
SynchrotronFixedAlphaBandPass
GalacticDustSynchrotronCorrelationFixedAlphaBandPass
"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
import candl.transformations.abstract_base
import candl.transformations.common

# --------------------------------------#
# DUST AND SYNCHROTRON WITH BAND PASS INTEGRALS
# --------------------------------------#


class GalacticDustBandPass(candl.transformations.abstract_base.ForegroundBandPass):
    """
    Dusty foreground with modified black-body frequency scaling with integral over band pass with a power law ell
    power spectrum.
    $ A * f(beta, bdp_1) * f(beta, bdp_2) * \left( \ell / \ell_{ref} \right)^(\alpha + 2) $
    where:
        A : amplitude
        ell_{ref} : reference ell
        beta : frequency scaling parameter
        alpha : power law index
        f(beta, bdp_1) : is the frequency scaling given the band pass
    The +2 to the exponent is convention.

    User required arguments in data set yaml file
    ---------
    ell_ref : float
        Reference ell.
    nu_ref : float
        Reference frequency.
    T_GALDUST : float
        Temperature of the dust.
    amp_param : str
        The name of the amplitude parameter.
    alpha_param : str
        The name of the power law index parameter.
    beta_param : str
        The name of the frequency scaling parameter.
    affected_specs : str
        List of spectrum identifiers the transformation is applied to.

    Example yaml block to add polarised galactic dust to all EE spectra:
    - Module: "candl.transformations.common.GalacticDustBandPass"
      amp_param: "BB_GalDust_BDP_Amp"
      alpha_param: "BB_GalDust_BDP_Alpha"
      beta_param: "BB_GalDust_BDP_Beta"
      nu_ref: 353
      affected_specs: ["BB 90x90", "BB 90x150", "BB 90x220", "BB 150x150", "BB 150x220", "BB 220x220"]
      ell_ref: 80
      T_GALDUST: 19.6
      descriptor: "BB Polarised Galactic Dust (Bandpass)"

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Attributes
    -------
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

    See also
    -------
    dust_frequency_scaling_bandpass
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
        -------
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
        -------
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
        -------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Foreground spectrum.
        """

        # amplitude part (frequency scaling)
        amp_vals = jnp.array(
            [
                candl.transformations.common.dust_frequency_scaling_bandpass(
                    sample_params[self.beta_param],
                    self.T_dust,
                    self.nu_ref,
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref],
                )
                * candl.transformations.common.dust_frequency_scaling_bandpass(
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
        -------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class SynchrotronBandPass(candl.transformations.abstract_base.ForegroundBandPass):
    """
    Synchrotron foreground with modified power law frequency scaling with integral over band pass with a power law ell
    power spectrum.
    $ A * f(beta, bdp_1) * f(beta, bdp_2) * \left( \ell / \ell_{ref} \right)^(\alpha) $
    where:
        A : amplitude
        ell_{ref} : reference ell
        beta : frequency scaling parameter
        alpha_param : power law index
        f(beta, bdp_1) : is the frequency scaling given the band pass

    User required arguments in data set yaml file
    ---------
    ell_ref : float
        Reference ell.
    nu_ref : float
        Reference frequency.
    amp_param : str
        The name of the amplitude parameter.
    alpha_param : str
        The name of the power law index parameter.
    beta_param : str
        The name of the frequency scaling parameter.
    affected_specs : str
        List of spectrum identifiers the transformation is applied to.

    Example yaml block to add synchrotron power to all BB spectra:
    - Module: "candl.transformations.common.SynchrotronBandPass"
      amp_param: "BB_Sync_BDP_Amp"
      alpha_param : "BB_Sunc_BDP_Alpha"
      beta_param: "BB_Sync_BDP_Beta"
      nu_ref: 150
      affected_specs: ["BB 90x90", "BB 90x150", "BB 90x220", "BB 150x150", "BB 150x220", "BB 220x220"]
      ell_ref: 80
      descriptor: "BB Synchrotron (Bandpass)"

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Attributes
    -------
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
    amp_param : str
        The name of the amplitude parameter.
    alpha_param : str
        The name of the power law index parameter.
    beta_param : str
        The name of the frequency scaling parameter.

    See also
    -------
    sync_frequency_scaling_bandpass
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
        descriptor="Synchrotron (Band pass)",
    ):
        """
        Initialise a new instance of the SynchrotronBandPass class.

        Arguments
        -------
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
        amp_param : str
            The name of the amplitude parameter.
        alpha_param : str
            The name of the power law index parameter.
        beta_param : str
            The name of the frequency scaling parameter.

        Returns
        -------
        SynchrotronBandPass
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

        self.amp_param = amp_param
        self.beta_param = beta_param
        self.alpha_param = alpha_param

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
        -------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Foreground spectrum.
        """

        # amplitude part (frequency scaling)
        amp_vals = jnp.array(
            [
                candl.transformations.common.sync_frequency_scaling_bandpass(
                    sample_params[self.beta_param],
                    self.nu_ref,
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref],
                )
                * candl.transformations.common.sync_frequency_scaling_bandpass(
                    sample_params[self.beta_param],
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
        ell_dependence = (self.ells / self.ell_ref) ** (sample_params[self.alpha_param])
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
        -------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


# --------------------------------------#
# DUST AND SYNCHROTRON WITH BAND PASS INTEGRALS - FIXED ALPHA
# --------------------------------------#


class GalacticDustSynchrotronCorrelationBandPass(
    candl.transformations.abstract_base.ForegroundBandPass
):
    """
    Correlation between galactic dust and synchrotron.
    Uses geometric mean of GalacticDustBandPass and SynchrotronBandPass, intended to be used in conjunction with these
    two classes.
    Note: assumes that both have the same ell_ref!

    User required arguments in data set yaml file
    ---------
    ell_ref : float
        Reference ell.
    nu_ref_dust : float
        Dust reference frequency.
    nu_ref_sync : float
        Synchrotron reference frequency.
    correlation_param : str
        Name of the correlation parameter.
    T_GALDUST : float
        Temperature of the dust.
    dust_amp_param : str
        The name of the dust amplitude parameter.
    dust_alpha_param : str
        The name of the dust power law index parameter.
    dust_beta_param : str
        The name of the dust frequency scaling parameter.
    affected_specs : str
        List of spectrum identifiers the transformation is applied to.
    sync_amp_param : str
        The name of the synchrotron amplitude parameter.
    sync_alpha_param : str
        The name of the synchrotron power law index parameter.
    sync_beta_param : str
        The name of the synchrotron frequency scaling parameter.


    Example yaml block to add correlation to BB spectra:
    - Module: "candl.transformations.low_ell_BB.GalacticDustSynchrotronCorrelationBandPass"
      correlation_param: "BB_rho_dust_sync"
      dust_amp_param: "BB_GalDust_BDP_Amp"
      dust_alpha_param: "BB_GalDust_BDP_Alpha"
      dust_beta_param: "BB_GalDust_BDP_Beta"
      sync_amp_param: "BB_Sync_BDP_Amp"
      sync_alpha_param: "BB_Sync_BDP_Alpha"
      sync_beta_param: "BB_Sync_BDP_Beta"
      nu_ref_dust: 353
      nu_ref_sync: 23
      affected_specs: ["BB 90x90", "BB 90x150", "BB 150x150"]
      ell_ref: 80
      T_GALDUST: 19.6
      descriptor: "BB Dust Synchrotron Correlation (Bandpass)"

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Attributes
    -------
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
    nu_ref_dust : float
        Dust reference frequency.
    nu_ref_sync : float
        Synchrotron reference frequency.
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
    dust_amp_param : str
        The name of the dust amplitude parameter.
    dust_alpha_param : str
        The name of the dust power law index parameter.
    dust_beta_param : str
        The name of the dust frequency scaling parameter.
    sync_amp_param : str
        The name of the sync amplitude parameter.
    sync_alpha_param : str
        The name of the sync power law index parameter.
    sync_beta_param : str
        The name of the sync frequency scaling parameter.
    correlation_param : str
        Name of the correlation parameter.

    See also
    -------
    dust_frequency_scaling_bandpass
    """

    def __init__(
        self,
        ells,
        spec_order,
        bandpass_info,
        affected_specs,
        correlation_param,
        dust_amp_param,
        dust_alpha_param,
        dust_beta_param,
        sync_amp_param,
        sync_alpha_param,
        sync_beta_param,
        ell_ref,
        nu_ref_dust,
        nu_ref_sync,
        T_GALDUST,
        descriptor="Dust Synchrotron Correlation (Band pass)",
    ):
        """
        Initialise a new instance of the GalacticDustSynchrotronCorrelationBandPass class.

        Arguments
        -------
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
        nu_ref_dust : float
            Dust reference frequency.
        nu_ref_sync : float
            Synchrotron reference frequency.
        T_GALDUST : float
            Temperature of the CIB.
        dust_amp_param : str
            The name of the dust amplitude parameter.
        dust_alpha_param : str
            The name of the dust power law index parameter.
        dust_beta_param : str
            The name of the dust frequency scaling parameter.
        sync_amp_param : str
            The name of the sync amplitude parameter.
        sync_alpha_param : str
            The name of the sync power law index parameter.
        sync_beta_param : str
            The name of the sync frequency scaling parameter.
        correlation_param : str
            Name of the correlation parameter.

        Returns
        -------
        GalacticDustSynchrotronCorrelationBandPass
            A new instance of the class.
        """

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            bandpass_info=bandpass_info,
            ell_ref=ell_ref,
            nu_ref={"dust": nu_ref_dust, "sync": nu_ref_sync},
            descriptor=descriptor,
            param_names=[
                correlation_param,
                dust_amp_param,
                dust_alpha_param,
                dust_beta_param,
                sync_amp_param,
                sync_alpha_param,
                sync_beta_param,
            ],
        )

        self.T_dust = T_GALDUST
        self.dust_amp_param = dust_amp_param
        self.dust_alpha_param = dust_alpha_param
        self.dust_beta_param = dust_beta_param

        self.sync_amp_param = sync_amp_param
        self.sync_alpha_param = sync_alpha_param
        self.sync_beta_param = sync_beta_param

        self.correlation_param = correlation_param

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
        -------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Foreground spectrum.
        """

        # amplitude part (frequency scaling)
        amp_vals_1 = jnp.array(
            [
                candl.transformations.common.dust_frequency_scaling_bandpass(
                    sample_params[self.dust_beta_param],
                    self.T_dust,
                    self.nu_ref["dust"],
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref["dust"]],
                )
                * candl.transformations.common.sync_frequency_scaling_bandpass(
                    sample_params[self.sync_beta_param],
                    self.nu_ref["sync"],
                    self.bandpass_info[i][1].nu_spacing,
                    self.bandpass_info[i][1].nu_vals,
                    self.bandpass_info[i][1].bandpass_vals,
                    self.bandpass_info[i][1].thermo_conv[self.nu_ref["sync"]],
                )
                for i in range(self.N_spec)
            ]
        )

        amp_vals_2 = jnp.array(
            [
                candl.transformations.common.dust_frequency_scaling_bandpass(
                    sample_params[self.dust_beta_param],
                    self.T_dust,
                    self.nu_ref["dust"],
                    self.bandpass_info[i][1].nu_spacing,
                    self.bandpass_info[i][1].nu_vals,
                    self.bandpass_info[i][1].bandpass_vals,
                    self.bandpass_info[i][1].thermo_conv[self.nu_ref["dust"]],
                )
                * candl.transformations.common.sync_frequency_scaling_bandpass(
                    sample_params[self.sync_beta_param],
                    self.nu_ref["sync"],
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref["sync"]],
                )
                for i in range(self.N_spec)
            ]
        )

        amp_vals = amp_vals_1 + amp_vals_2

        amp_vals *= sample_params[self.correlation_param] * jnp.sqrt(
            sample_params[self.dust_amp_param] * sample_params[self.sync_amp_param]
        )
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** (
            (
                sample_params[self.dust_alpha_param]
                + 2.0
                + sample_params[self.sync_alpha_param]
            )
            / 2.0
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
        -------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class GalacticDustFixedAlphaBandPass(
    candl.transformations.abstract_base.ForegroundBandPass
):
    """
    Dusty foreground with modified black-body frequency scaling with integral over band pass with a power law ell
    power spectrum.
    $ A * f(beta, bdp_1) * f(beta, bdp_2) * \left( \ell / \ell_{ref} \right)^(-0.4) $
    where:
        A : amplitude
        ell_{ref} : reference ell
        beta : frequency scaling parameter
        f(beta, bdp_1) : is the frequency scaling given the band pass

    User required arguments in data set yaml file
    ---------
    ell_ref : float
        Reference ell.
    nu_ref : float
        Reference frequency.
    T_GALDUST : float
        Temperature of the dust.
    amp_param : str
        The name of the amplitude parameter.
    beta_param : str
        The name of the frequency scaling parameter.
    affected_specs : str
        List of spectrum identifiers the transformation is applied to.

    Example yaml block to add polarised galactic dust to all BB spectra:
    - Module: "candl.transformations.common.GalacticDustBandPass"
      amp_param: "BB_GalDust_BDP_Amp"
      beta_param: "BB_GalDust_BDP_Beta"
      nu_ref: 353
      affected_specs: ["BB 90x90", "BB 90x150", "BB 90x220", "BB 150x150", "BB 150x220", "BB 220x220"]
      ell_ref: 80
      T_GALDUST: 19.6
      descriptor: "BB Polarised Galactic Dust (Bandpass)"

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Attributes
    -------
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
    beta_param : str
        The name of the frequency scaling parameter.

    See also
    -------
    dust_frequency_scaling_bandpass
    """

    def __init__(
        self,
        ells,
        spec_order,
        bandpass_info,
        affected_specs,
        amp_param,
        beta_param,
        ell_ref,
        nu_ref,
        T_GALDUST,
        descriptor="Galactic Dust (Band pass)",
    ):
        """
        Initialise a new instance of the GalacticDustBandPass class.

        Arguments
        -------
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
        beta_param : str
            The name of the frequency scaling parameter.

        Returns
        -------
        GalacticDustFixedAlphaBandPass
            A new instance of the class.
        """

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            bandpass_info=bandpass_info,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            descriptor=descriptor,
            param_names=[amp_param, beta_param],
        )

        self.T_dust = T_GALDUST
        self.amp_param = amp_param
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
        -------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Foreground spectrum.
        """

        # amplitude part (frequency scaling)
        amp_vals = jnp.array(
            [
                candl.transformations.common.dust_frequency_scaling_bandpass(
                    sample_params[self.beta_param],
                    self.T_dust,
                    self.nu_ref,
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref],
                )
                * candl.transformations.common.dust_frequency_scaling_bandpass(
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
        ell_dependence = (self.ells / self.ell_ref) ** (-0.4)
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
        -------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class SynchrotronFixedAlphaBandPass(
    candl.transformations.abstract_base.ForegroundBandPass
):
    """
    Synchrotron foreground with modified power law frequency scaling with integral over band pass with a power law ell
    power spectrum.
    $ A * f(beta, bdp_1) * f(beta, bdp_2) * \left( \ell / \ell_{ref} \right)^(-0.6) $
    where:
        A : amplitude
        ell_{ref} : reference ell
        beta : frequency scaling parameter
        f(beta, bdp_1) : is the frequency scaling given the band pass

    User required arguments in data set yaml file
    ---------
    ell_ref : float
        Reference ell.
    nu_ref : float
        Reference frequency.
    amp_param : str
        The name of the amplitude parameter.
    beta_param : str
        The name of the frequency scaling parameter.
    affected_specs : str
        List of spectrum identifiers the transformation is applied to.

    Example yaml block to add synchrotron power to all BB spectra:
    - Module: "candl.transformations.common.SynchrotronBandPass"
      amp_param: "BB_Sync_BDP_Amp"
      beta_param: "BB_Sync_BDP_Beta"
      nu_ref: 150
      affected_specs: ["BB 90x90", "BB 90x150", "BB 90x220", "BB 150x150", "BB 150x220", "BB 220x220"]
      ell_ref: 80
      descriptor: "BB Synchrotron (Bandpass)"

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Attributes
    -------
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
    amp_param : str
        The name of the amplitude parameter.
    beta_param : str
        The name of the frequency scaling parameter.

    See also
    -------
    sync_frequency_scaling_bandpass
    """

    def __init__(
        self,
        ells,
        spec_order,
        bandpass_info,
        affected_specs,
        amp_param,
        beta_param,
        ell_ref,
        nu_ref,
        descriptor="Synchrotron (Band pass)",
    ):
        """
        Initialise a new instance of the SynchrotronBandPass class.

        Arguments
        -------
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
        amp_param : str
            The name of the amplitude parameter.
        beta_param : str
            The name of the frequency scaling parameter.

        Returns
        -------
        SynchrotronFixedAlphaBandPass
            A new instance of the class.
        """

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            bandpass_info=bandpass_info,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            descriptor=descriptor,
            param_names=[amp_param, beta_param],
        )

        self.amp_param = amp_param
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
        -------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Foreground spectrum.
        """

        # amplitude part (frequency scaling)
        amp_vals = jnp.array(
            [
                candl.transformations.common.sync_frequency_scaling_bandpass(
                    sample_params[self.beta_param],
                    self.nu_ref,
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref],
                )
                * candl.transformations.common.sync_frequency_scaling_bandpass(
                    sample_params[self.beta_param],
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
        ell_dependence = (self.ells / self.ell_ref) ** (-0.6)
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
        -------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


class GalacticDustSynchrotronCorrelationFixedAlphaBandPass(
    candl.transformations.abstract_base.ForegroundBandPass
):
    """
    Correlation between galactic dust and synchrotron.
    Uses geometric mean of GalacticDustFixedAlphaBandPass and SynchrotronFixedAlphaBandPass, intended to be used in conjunction with these
    two classes.
    Note: assumes that both have the same ell_ref!

    User required arguments in data set yaml file
    ---------
    ell_ref : float
        Reference ell.
    nu_ref_dust : float
        Dust reference frequency.
    nu_ref_sync : float
        Synchrotron reference frequency.
    correlation_param : str
        Name of the correlation parameter.
    T_GALDUST : float
        Temperature of the dust.
    dust_amp_param : str
        The name of the dust amplitude parameter.
    dust_beta_param : str
        The name of the dust frequency scaling parameter.
    affected_specs : str
        List of spectrum identifiers the transformation is applied to.
    sync_amp_param : str
        The name of the synchrotron amplitude parameter.
    sync_beta_param : str
        The name of the synchrotron frequency scaling parameter.


    Example yaml block to add correlation to BB spectra:
    - Module: "candl.transformations.low_ell_BB.GalacticDustSynchrotronCorrelationFixedAlphaBandPass"
      correlation_param: "BB_rho_dust_sync"
      dust_amp_param: "BB_GalDust_BDP_Amp"
      dust_beta_param: "BB_GalDust_BDP_Beta"
      sync_amp_param: "BB_Sync_BDP_Amp"
      sync_beta_param: "BB_Sync_BDP_Beta"
      nu_ref_dust: 353
      nu_ref_sync: 23
      affected_specs: ["BB 90x90", "BB 90x150", "BB 150x150"]
      ell_ref: 80
      T_GALDUST: 19.6
      descriptor: "BB Dust Synchrotron Correlation (Bandpass)"

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.

    Attributes
    -------
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
    nu_ref_dust : float
        Dust reference frequency.
    nu_ref_sync : float
        Synchrotron reference frequency.
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
    dust_amp_param : str
        The name of the dust amplitude parameter.
    dust_beta_param : str
        The name of the dust frequency scaling parameter.
    sync_amp_param : str
        The name of the sync amplitude parameter.
    sync_beta_param : str
        The name of the sync frequency scaling parameter.
    correlation_param : str
        Name of the correlation parameter.

    See also
    -------
    dust_frequency_scaling_bandpass
    """

    def __init__(
        self,
        ells,
        spec_order,
        bandpass_info,
        affected_specs,
        correlation_param,
        dust_amp_param,
        dust_beta_param,
        sync_amp_param,
        sync_beta_param,
        ell_ref,
        nu_ref_dust,
        nu_ref_sync,
        T_GALDUST,
        descriptor="Dust Synchrotron Correlation (Band pass)",
    ):
        """
        Initialise a new instance of the GalacticDustSynchrotronCorrelationFixedAlphaBandPass class.

        Arguments
        -------
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
        nu_ref_dust : float
            Dust reference frequency.
        nu_ref_sync : float
            Synchrotron reference frequency.
        T_GALDUST : float
            Temperature of the CIB.
        dust_amp_param : str
            The name of the dust amplitude parameter.
        dust_beta_param : str
            The name of the dust frequency scaling parameter.
        sync_amp_param : str
            The name of the sync amplitude parameter.
        sync_beta_param : str
            The name of the sync frequency scaling parameter.
        correlation_param : str
            Name of the correlation parameter.

        Returns
        -------
        GalacticDustSynchrotronCorrelationFixedAlphaBandPass
            A new instance of the class.
        """

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            bandpass_info=bandpass_info,
            ell_ref=ell_ref,
            nu_ref={"dust": nu_ref_dust, "sync": nu_ref_sync},
            descriptor=descriptor,
            param_names=[
                correlation_param,
                dust_amp_param,
                dust_beta_param,
                sync_amp_param,
                sync_beta_param,
            ],
        )

        self.T_dust = T_GALDUST
        self.dust_amp_param = dust_amp_param
        self.dust_beta_param = dust_beta_param

        self.sync_amp_param = sync_amp_param
        self.sync_beta_param = sync_beta_param

        self.correlation_param = correlation_param

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
        -------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Foreground spectrum.
        """

        # amplitude part (frequency scaling)
        amp_vals_1 = jnp.array(
            [
                candl.transformations.common.dust_frequency_scaling_bandpass(
                    sample_params[self.dust_beta_param],
                    self.T_dust,
                    self.nu_ref["dust"],
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref["dust"]],
                )
                * candl.transformations.common.sync_frequency_scaling_bandpass(
                    sample_params[self.sync_beta_param],
                    self.nu_ref["sync"],
                    self.bandpass_info[i][1].nu_spacing,
                    self.bandpass_info[i][1].nu_vals,
                    self.bandpass_info[i][1].bandpass_vals,
                    self.bandpass_info[i][1].thermo_conv[self.nu_ref["sync"]],
                )
                for i in range(self.N_spec)
            ]
        )

        amp_vals_2 = jnp.array(
            [
                candl.transformations.common.dust_frequency_scaling_bandpass(
                    sample_params[self.dust_beta_param],
                    self.T_dust,
                    self.nu_ref["dust"],
                    self.bandpass_info[i][1].nu_spacing,
                    self.bandpass_info[i][1].nu_vals,
                    self.bandpass_info[i][1].bandpass_vals,
                    self.bandpass_info[i][1].thermo_conv[self.nu_ref["dust"]],
                )
                * candl.transformations.common.sync_frequency_scaling_bandpass(
                    sample_params[self.sync_beta_param],
                    self.nu_ref["sync"],
                    self.bandpass_info[i][0].nu_spacing,
                    self.bandpass_info[i][0].nu_vals,
                    self.bandpass_info[i][0].bandpass_vals,
                    self.bandpass_info[i][0].thermo_conv[self.nu_ref["sync"]],
                )
                for i in range(self.N_spec)
            ]
        )

        amp_vals = amp_vals_1 + amp_vals_2

        amp_vals *= sample_params[self.correlation_param] * jnp.sqrt(
            sample_params[self.dust_amp_param] * sample_params[self.sync_amp_param]
        )
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** (-0.5)
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
        -------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        -------
        array, float
            Transformed spectrum.
        """

        return Dls + self.output(sample_params)


# --------------------------------------#
# DUST AND SYNCHROTRON WITH BAND PASS INTEGRALS - DECORRELATION ACROSS FREQUENCIES
# --------------------------------------#


class GalacticDustDecorrelationBandPass(GalacticDustBandPass):
    """
    Polarised galactic dust including decorrelation for cross-frequency spectra.
    Slightly modified version of GalacticDustBandPass (see for more extensive documentation).
    Requires additionally "decorr_param", "nu_pivot" set.
    Argument "ell_scaling" is optional: default is flat ell-scaling, other options are "linear" and "quadratic".

    Example yaml block to add polarised galactic dust to all BB spectra:
    - Module: "candl.transformations.low_ell_BB.GalacticDustDecorrelationBandPass"
      decorr_param: "BB_GalDust_BDP_Decorr"
      amp_param: "BB_GalDust_BDP_Amp"
      alpha_param: "BB_GalDust_BDP_Alpha"
      beta_param: "BB_GalDust_BDP_Beta"
      nu_ref: 353
      nu_pivot: [353, 217]
      affected_specs: ["BB 90x90", "BB 90x150", "BB 150x150"]
      ell_ref: 80
      T_GALDUST: 19.6
      descriptor: "BB Polarised Galactic Dust (Decorrelation, Bandpass)"

    Follows implementation in BK Likelihood, see BK15 (https://arxiv.org/pdf/1810.05216.pdf) Appendix F for details.
    Not extended for unphysical values of decorr_param.
    """

    def __init__(
        self,
        ells,
        spec_order,
        bandpass_info,
        affected_specs,
        decorr_param,
        amp_param,
        alpha_param,
        beta_param,
        ell_ref,
        nu_ref,
        nu_pivot,
        T_GALDUST,
        ell_scaling=None,
        descriptor="Galactic Dust (Decorrelation, Band pass)",
    ):

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            bandpass_info=bandpass_info,
            affected_specs=affected_specs,
            amp_param=amp_param,
            alpha_param=alpha_param,
            beta_param=beta_param,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            T_GALDUST=T_GALDUST,
            descriptor=descriptor,
        )

        self.decorr_param = decorr_param

        self.nu_pivot = nu_pivot

        self.ell_scaling = ell_scaling
        self.ell_part = 1.0
        if self.ell_scaling == "linear":
            self.ell_part = jnp.tile(self.ells, self.N_spec)
        elif self.ell_scaling == "quadratic":
            self.ell_part = jnp.tile(self.ells**2, self.N_spec)

        # Make mask of auto-frequency spectra (decorrelation does not apply to them)
        self.auto_spec_mask = jnp.asarray(
            [len(set(s[3:].split("x"))) == 1 for s in self.spec_order]
        )
        self.full_auto_mask = jnp.repeat(self.auto_spec_mask, len(self.ells))

    def output(self, sample_params):

        # Get regular dust power
        fg_pow = super().output(sample_params)

        # Calculate decorrelation
        amp_vals = jnp.array(
            [
                jnp.log(
                    self.bandpass_info[i][0].central_nu
                    / self.bandpass_info[i][1].central_nu
                )
                ** 2
                / jnp.log(self.nu_pivot[0] / self.nu_pivot[1]) ** 2
                for i in range(self.N_spec)
            ]
        )
        amp_vals_tiled = jnp.repeat(amp_vals, len(self.ells))
        delta_prime = jnp.exp(
            jnp.log(sample_params[self.decorr_param]) * amp_vals_tiled * self.ell_part
        )
        delta_prime = jax_optional_set_element(delta_prime, self.full_auto_mask, 1.0)

        return delta_prime * fg_pow


class GalacticDustFixedAlphaDecorrelationBandPass(GalacticDustFixedAlphaBandPass):
    """
    Slightly modified version of GalacticDustDecorrelationBandPass to fix the power law index (as in GalacticDustFixedAlphaBandPass).
    """

    def __init__(
        self,
        ells,
        spec_order,
        bandpass_info,
        affected_specs,
        decorr_param,
        amp_param,
        beta_param,
        ell_ref,
        nu_ref,
        nu_pivot,
        T_GALDUST,
        ell_scaling=None,
        descriptor="Galactic Dust (Decorrelation, Band pass)",
    ):

        super().__init__(
            ells=ells,
            spec_order=spec_order,
            bandpass_info=bandpass_info,
            affected_specs=affected_specs,
            amp_param=amp_param,
            beta_param=beta_param,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            T_GALDUST=T_GALDUST,
            descriptor=descriptor,
        )

        self.decorr_param = decorr_param

        self.nu_pivot = nu_pivot

        self.ell_scaling = ell_scaling
        self.ell_part = 1.0
        if self.ell_scaling == "linear":
            self.ell_part = jnp.tile(self.ells, self.N_spec)
        elif self.ell_scaling == "quadratic":
            self.ell_part = jnp.tile(self.ells**2, self.N_spec)

        # Make mask of auto-frequency spectra (decorrelation does not apply to them)
        self.auto_spec_mask = jnp.asarray(
            [len(set(s[3:].split("x"))) == 1 for s in self.spec_order]
        )
        self.full_auto_mask = jnp.repeat(self.auto_spec_mask, len(self.ells))

    def output(self, sample_params):

        # Get regular dust power
        fg_pow = super().output(sample_params)

        # Calculate decorrelation
        amp_vals = jnp.array(
            [
                jnp.log(
                    self.bandpass_info[i][0].central_nu
                    / self.bandpass_info[i][1].central_nu
                )
                ** 2
                / jnp.log(self.nu_pivot[0] / self.nu_pivot[1]) ** 2
                for i in range(self.N_spec)
            ]
        )
        amp_vals_tiled = jnp.repeat(amp_vals, len(self.ells))
        delta_prime = jnp.exp(
            jnp.log(sample_params[self.decorr_param]) * amp_vals_tiled * self.ell_part
        )
        delta_prime = jax_optional_set_element(delta_prime, self.full_auto_mask, 1.0)

        return delta_prime * fg_pow

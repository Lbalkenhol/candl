"""
Transformation module containing abstract base classes. They establish the general framework of how transformations
work in the likelihood. Specific foregrounds or other transformations are intended to be implemented as subclasses
of these in their own files. See candl.transformations.common.py for examples.

Overview
-----------

* :class:`Transformation`
* :class:`Foreground`
* :class:`TemplateForeground`
* :class:`DustyForeground`
* :class:`Calibration`
* :class:`ForegroundBandPass`
* :class:`BandPass`
"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
import candl.constants

# --------------------------------------#
# ABSTRACT BASE CLASSES
# --------------------------------------#


class Transformation:
    """
    Abstract base class for transformation.
    Transformations are applied to theory spectra in order to make them comparable to the data.
    Transformations are instantiated by the likelihood.

    Attributes
    --------------
    ells : array (float)
            The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    operation_hint : str
        Type of the 'transform' operation, i.e. 'additive', 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    transform :
        Returns a transformed spectrum.

    Notes
    -----------------
    On making subclasses: initialisation arguments can either correspond the names of attributes of the likelihood, supplied by the user, or one of the few special keywords that the likelihood understands.
    """

    def __init__(self, ells, descriptor="", param_names=[], operation_hint=""):
        """
        Initialise the Transformation.
        Intended to be expanded upon by subclasses.

        Parameters
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        par_names : list
            Names of parameters involved in transformation.
        operation_hint : str
            Type of the 'transform' operation, i.e. 'additive', 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

        Returns
        --------------
        Transformation
            A new instance of the Transformation class.
        """

        self.ells = ells
        self.descriptor = descriptor
        self.param_names = param_names
        self.operation_hint = operation_hint

    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.
        Intended to be overwritten by subclasses.

        Parameters
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

        return Dls


# --------------------------------------#
# FOREGROUND BASE CLASS
# --------------------------------------#


class Foreground(Transformation):
    """
    Abstract base class for foregrounds.

    Attributes
    --------------
    ells : array (float)
            The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    ell_ref : float
        Reference ell.
    nu_ref : float
        Reference frequency.
    operation_hint : str
        Type of the 'transform' operation, i.e. 'additive' (default), 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.
    """

    def __init__(
        self,
        ells,
        ell_ref=None,
        nu_ref=None,
        descriptor="",
        param_names=[],
        operation_hint="additive",
    ):
        """
        Initialise a new instance of the Foreground class.
        Intended to be expanded upon by subclasses.

        Parameters
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        par_names : list
            Names of parameters involved in transformation.
        nu_ref : float
            Reference frequency.
        ell_ref : float
            Reference ell.
        operation_hint : str
            Type of the 'transform' operation, i.e. 'additive' (default), 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.


        Returns
        -------
        Foreground
            A new instance of the foreground class.
        """

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=param_names,
            operation_hint=operation_hint,
        )

        self.nu_ref = nu_ref
        self.ell_ref = ell_ref

    def output(self, sampled_params):
        """
        Return foreground spectrum.
        Intended to be overwritten by subclasses.

        Parameters
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        return jnp.zeros(1)

    def transform(self, Dls, sampled_params):
        """
        Transform spectrum by adding foreground component (result of output method).
        Intended to be overwritten by subclasses.

        Parameters
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

        return Dls


class TemplateForeground(Foreground):
    """
    Abstract base class for template foregrounds, i.e. some D_\ell template spectrum that gets reused with potentially
    a free amplitude or frequency scaling added on top.

    Attributes
    --------------
    template_arr : array (float)
        Template spectrum and ells.
    template_spec : array (float)
        Template spectrum.
    template_ells : array (int)
        Template ells.
    ell_ref : int
        Reference ell for normalisation. If zero, do not normalise.
    ells : array (float)
        The ell range the transformation acts on.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    nu_ref : float
        Reference frequency.
    operation_hint : str
        Type of the 'transform' operation, i.e. 'additive' (default), 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.
    """

    def __init__(
        self,
        ells,
        template_arr,
        ell_ref,
        descriptor="",
        param_names=[],
        operation_hint="additive",
    ):
        """
        Initialise a new instance of the TemplateForeground class.
        Intended to be expanded upon by subclasses.
        Crops template to required ell range and normalised to amplitude of 1 at ell_ref, unless ell_ref is zero, in which case no normalisation is performed.

        Parameters
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        par_names : list
            Names of parameters involved in transformation.
        template_arr : array (float)
            Array with two columns, the first for ell values, the second for values of the template spectrum.
        ell_ref : float
            Reference ell to normalise the template at (amplitude = 1).
        operation_hint : str
            Type of the 'transform' operation, i.e. 'additive' (default), 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

        Returns
        --------------
        Foreground
            A new instance of the TemplateForeground class.
        """

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=param_names,
            operation_hint=operation_hint,
        )

        # Read in template
        self.template_arr = template_arr  # jnp.asarray(np.loadtxt(template_file))
        self.template_spec = self.template_arr[:, 1]
        self.template_ells = jnp.asarray(self.template_arr[:, 0], dtype=int)

        # Grab other args
        self.ells = ells
        self.ell_ref = ell_ref

        # Normalise
        if self.ell_ref > 0:
            self.template_spec /= self.template_spec[
                jnp.argwhere(self.template_ells == self.ell_ref)[0][0]
            ]

        # Trim template for ell range
        start_ix = jnp.argwhere(self.template_ells == jnp.amin(self.ells))[0][0]
        stop_ix = jnp.argwhere(self.template_ells == jnp.amax(self.ells))[0][0]

        self.template_spec = self.template_spec[start_ix : stop_ix + 1]
        self.template_ells = self.template_ells[start_ix : stop_ix + 1]

    def output(self, sample_params):
        """
        Return foreground spectrum.
        Intended to be overwritten by subclasses.

        Parameters
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        return self.template_spec

    def transform(self, Dls, sample_params):
        """
        Transform spectrum by adding foreground component (result of output method).
        Intended to be overwritten by subclasses.

        Parameters
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


class DustyForeground(Foreground):
    """
    Abstract base class for dusty foregrounds using modified black-body spectra.

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
    operation_hint : str
        Type of the 'transform' operation, i.e. 'additive' (default), 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.
    """

    def __init__(
        self,
        ells,
        spec_order,
        freq_info,
        affected_specs,
        ell_ref,
        nu_ref,
        T_dust,
        descriptor="",
        param_names=[],
        operation_hint="additive",
    ):
        """
        Initialise a new instance of the DustyForeground class.
        Intended to be expanded upon by subclasses.

        Parameters
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
        operation_hint : str
            Type of the 'transform' operation, i.e. 'additive' (default), 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

        Returns
        --------------
        DustyForeground
            A new instance of the DustyForeground class.
        """

        super().__init__(
            ells=ells,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            descriptor=descriptor,
            param_names=param_names,
            operation_hint=operation_hint,
        )

        self.spec_order = spec_order
        self.affected_specs = affected_specs
        self.spec_mask = jnp.asarray(
            [spec in self.affected_specs for spec in self.spec_order]
        )
        self.N_spec = len(self.spec_mask)
        self.freq_info = freq_info
        self.T_dust = T_dust

        # Turn spectrum mask into a full mask
        self.full_mask = jnp.asarray(
            jnp.repeat(self.spec_mask, len(self.ells)), dtype=float
        )


class Calibration(Transformation):
    """
    Abstract base class for calibration.
    Useful to catch tool methods that identify calibration transformations.
    """

    def __init__(
        self,
        ells,
        descriptor="",
        param_names=[],
        operation_hint="multiplicative",
    ):
        """
        Initialise a new instance of the Calibration class.
        Intended to be expanded upon by subclasses.

        Parameters
        --------------
        ells : array (float)
            The ell range the transformation acts on.
        descriptor : str
            A short descriptor.
        par_names : list
            Names of parameters involved in transformation.
        operation_hint : str
            Type of the 'transform' operation, i.e. 'additive', 'multiplicative' (default), or other (anything else). Non-binding, used as a hint for helper functions in tools module.


        Returns
        -------
        Calibration
            A new instance of the calibration class.
        """

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=param_names,
            operation_hint=operation_hint,
        )


class IndividualCalibration(Calibration):
    """
    Base class to calibrate individual spectra one by one.

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
    operation_hint : str
        Type of the 'transform' operation, i.e. 'additive', 'multiplicative' (default), or other (anything else). Non-binding, used as a hint for helper functions in tools module.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    transform :
        transforms an input spectrum.
    """

    def __init__(
        self,
        ells,
        spec_order,
        spec_param_dict,
        descriptor="Calibration",
        operation_hint="multiplicative",
    ):
        """
        Initialise a new instance of the Calibration class.
        Intended to be expanded upon by subclasses.

        Parameters
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
        operation_hint : str
            Type of the 'transform' operation, i.e. 'additive', 'multiplicative' (default), or other (anything else). Non-binding, used as a hint for helper functions in tools module.

        Returns
        --------------
        DustyForeground
            A new instance of the Calibration class.
        """

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=list(
                np.unique(
                    [
                        cal_par
                        for spec_cal_pars in list(spec_param_dict.values())
                        for cal_par in spec_cal_pars
                    ]
                )
            ),
        )

        self.spec_param_dict = spec_param_dict
        self.spec_order = spec_order
        self.N_specs = len(self.spec_order)

        # Generate boolean mask of affected specs
        self.affected_specs = list(self.spec_param_dict.keys())
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

    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.
        Intended to be overwritten by subclasses with the details of the calibration model.

        Parameters
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

        return Dls


class ForegroundBandPass(Foreground):
    """
    Abstract base class for foreground with a frequency scaling using integrals over the band pass.

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
    operation_hint : str
        Type of the 'transform' operation, i.e. 'additive' (default), 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    output :
        gives the additive foreground contribution.
    transform :
        transforms an input spectrum.
    """

    def __init__(
        self,
        ells,
        spec_order,
        bandpass_info,
        ell_ref,
        nu_ref,
        descriptor="",
        param_names=[],
        operation_hint="additive",
    ):
        """
        Initialise a new instance of the DustyForeground class.
        Intended to be expanded upon by subclasses.

        Parameters
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
        operation_hint : str
            Type of the 'transform' operation, i.e. 'additive' (default), 'multiplicative', or other (anything else). Non-binding, used as a hint for helper functions in tools module.

        Returns
        --------------
        ForegroundBandPass
            A new instance of the ForegroundBandPass class.

        """

        super().__init__(
            ells=ells,
            ell_ref=ell_ref,
            nu_ref=nu_ref,
            descriptor=descriptor,
            param_names=param_names,
            operation_hint=operation_hint,
        )

        self.spec_order = spec_order
        self.N_spec = len(self.spec_order)
        self.bandpass_info = bandpass_info


# --------------------------------------#
# BAND PASS CLASS
# --------------------------------------#


class BandPass:
    """
    Base class to hold band pass. Assumes all band pass measurements are equally spaced.
    Modelled on procedure in BK_Planck likelihood:
    BICEP2/Keck Array and Planck Joint Analysis January 2015 Data Products, The BICEP2/Keck and Planck Collaborations, A Joint Analysis of BICEP2/Keck Array and Planck Data (http://bicepkeck.org/).

    Attributes
    --------------
    nu_vals : array (float)
        Frequencies where the band pass is measured
    bandpass_vals : array (float)
        Band pass values
    nu_spacing : float
        Spacing between measurements.
    thermo_conv : dict
        Dictionary holding the thermodynamic conversion factors for different reference frequencies.

    Methods
    ---------
    __init__ :
        initialises an instance of the class.
    calculate_thermodynamic_conversion :
        Calculates the thermodynamic conversion.

    """

    def __init__(self, bandpass_array):
        """
        Initialise the Band pass.

        Parameters
        --------------
        bandpass_array : array (float)
            (2, N) array holding frequencies and band pass measurements in this order.

        Returns
        --------------
        Transformation
            A new instance of the Bandpass class.
        """

        # Grab helpers/reshuffle data
        self.nu_vals = bandpass_array[:, 0]
        self.bandpass_vals = bandpass_array[:, 1]
        self.nu_spacing = np.diff(self.nu_vals)[
            0
        ]  # Assume measurements are equally spaced!
        self.thermo_conv = {}
        self.central_nu = np.sum(
            self.nu_spacing * self.nu_vals * self.bandpass_vals
        ) / np.sum(self.nu_spacing * self.bandpass_vals)

    def calculate_thermodynamic_conversion(self, nu_ref):
        """
        Calculate the thermodynamic conversion factor for a given reference frequency.

        Parameters
        --------------
        nu_ref : float
            Reference frequency.
        """

        exp_term = np.exp(
            self.nu_vals * candl.constants.GHz_KELVIN / candl.constants.T_CMB
        )
        thermo_integral = np.sum(
            self.nu_spacing
            * self.bandpass_vals
            * self.nu_vals**4
            * exp_term
            / (exp_term - 1) ** 2.0
        )

        exp_term_0 = np.exp(nu_ref * candl.constants.GHz_KELVIN / candl.constants.T_CMB)
        thermo_dust_norm = nu_ref**4 * exp_term_0 / (exp_term_0 - 1) ** 2

        self.thermo_conv[nu_ref] = thermo_integral / thermo_dust_norm

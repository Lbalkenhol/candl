"""
Tools to help interface the likelihood code with other software.

Overview:
---------------

Theory Code Interface Functions:

* :func:`get_CosmoPowerJAX_pars_to_theory_specs_func`
* :func:`get_CosmoPower_pars_to_theory_specs_func`
* :func:`get_PyCapse_pars_to_theory_specs_func`
* :func:`get_CAMB_pars_to_theory_specs_func`
* :func:`get_CLASS_pars_to_theory_specs_func`
* :func:`get_CobayaTheory_pars_to_theory_specs_func`

MontePython Interface:

* :func:`get_montepython_nuisance_param_block_for_like`

Cobaya Interface:

* :class:`CandlCobayaLikelihood`
* :func:`get_cobaya_likelihood_class_for_like`
* :func:`get_cobaya_info_dict_for_like`

Cobaya Theory Code Classes:

* :class:`CobayaTheoryCosmoPowerJAX`
* :class:`CobayaTheoryCosmoPower`
* :class:`CobayaTheoryPyCapse`
* :class:`CobayaTheoryBBTemplate`
* :class:`CobayaTheoryCosmoPowerJAXLensing`
* :class:`CobayaTheoryCosmoPowerLensing`
"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
import candl.io
import candl

# --------------------------------------#
# COSMOPOWER - COBAYA INTERFACE
# --------------------------------------#


class CobayaTheoryCosmoPowerJAX(cobaya_theory_Theory):
    """
    Wraps CosmoPower-JAX model into a cobaya.theory.Theory class.
    See D. Piras, A. Spurio Mancini 2023 and A. Spurio Mancini et al. 2021 for more (https://arxiv.org/abs/2305.06347, https://arxiv.org/abs/2106.03846).

    To initialise, pass 'emulator_filenames' in the relevant cobaya block:
    A dictionary of spectrum types and emulator file names (if they are placed in cosmopower-jax's trained_models/ folder), or the full path of the files, including in either case the file ending '.pkl' or '.npz'.

    This code is taken from the Cobaya example for custom theory codes and only slightly modified.
    See https://cobaya.readthedocs.io/en/latest/theories_and_dependencies.html.
    Torrado and Lewis, 2020 (https://arxiv.org/abs/2005.05290)

    Attributes
    ----------------
    emulator_filenames : dict
        File names of emulators.
    cp_emulators : dict
        CosmoPower emulators.
    cp_pars : list
        List of parameters required by the emulators.
    provider : Provider
        Cobaya provider.
    current_state : dict
        Dict containing current parameters and results.

    Methods
    ---------
    initialize :
        Initialises an instance of the class (loads emulators).
    initialize_with_provider :
        Initialization after other components initialised.
    get_requirements :
        Returns what the theory code needs to run.
    must_provide :
        Returns what the theory code needs to run.
    get_can_provide :
        Return what the theory code can supply.
    calculate :
        Carry out the calculation.
    get_Cl :
        Return result of calculation (Cl and Dl).
    """

    # I know having this be a class variable is not super elegant, but it's the easiest way to interface with Cobaya
    emulator_filenames: dict = {}

    def initialize(self):
        """
        Called from __init__ to initialise. Loads the emulator models.

        Notes
        ----------
        For CosmoPower-JAX emulator models are expected to be placed in the package directory.
        TE emulators are loaded as PCA+NN, other spectra as NN-only models.

        """

        self.cp_emulators = {}
        for spec_type in list(self.emulator_filenames.keys()):
            if spec_type == "TE":
                try:
                    self.cp_emulators[spec_type] = CPJ(
                        probe="custom_pca",
                        filename=self.emulator_filenames[spec_type],
                    )
                except:
                    self.cp_emulators[spec_type] = CPJ(
                        probe="custom_pca",
                        filepath=self.emulator_filenames[spec_type],
                    )
            else:
                try:
                    self.cp_emulators[spec_type] = CPJ(
                        probe="custom_log",
                        filename=self.emulator_filenames[spec_type],
                    )
                except:
                    self.cp_emulators[spec_type] = CPJ(
                        probe="custom_log",
                        filepath=self.emulator_filenames[spec_type],
                    )

        # Just using a random spec here, would be weird if TT/TE/EE had different requirements
        self.cp_pars = list(list(self.cp_emulators.values())[0].parameters)
        if "h" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("h")] = "H0"

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """

        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {p: None for p in self.cp_pars}

    def must_provide(self, **requirements):
        """
        Return dictionary of parameters that must be provided.
        """
        return {p: None for p in self.cp_pars}

    def get_can_provide(self):
        """
        Return list of quantities that can be provided.
        """
        return ["Cl", "Dl"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Calculate the CMB spectra.
        Calls the CosmoPower-JAX emulator models.
        """
        # Prepare hand-off to CosmoPower
        # numpy array creation is faster, but need jnp to get derivs
        # This is the order expected by CosmoPower-JAX

        pars_for_cp = [state["params"][p] for p in self.cp_pars]
        if "H0" in self.cp_pars:
            pars_for_cp[self.cp_pars.index("H0")] /= 100
        pars_for_cp = jnp.array(pars_for_cp)

        # Get CMB Cls
        state["Cl"] = {
            "ell": self.cp_emulators[list(self.cp_emulators.keys())[0]].modes
        }  # assume all supply the same ell range}
        state["Dl"] = {"ell": state["Cl"]["ell"]}
        for spec_type in list(self.cp_emulators.keys()):
            state["Cl"][spec_type] = (
                self.cp_emulators[spec_type].predict(pars_for_cp).ravel()
            )
            state["Dl"][spec_type] = (
                state["Cl"][spec_type]
                * self.cp_emulators[spec_type].modes
                * (self.cp_emulators[spec_type].modes + 1)
                / (2 * jnp.pi)
            )

    def get_Cl(self, ell_factor=False, **kwargs):
        """
        Return the Cls or Dls
        """
        if ell_factor:
            return self.current_state["Dl"].copy()
        else:
            return self.current_state["Cl"].copy()


class CobayaTheoryCosmoPower(cobaya_theory_Theory):
    """
    Wraps regular CosmoPower model into a cobaya theory class.
    See A. Spurio Mancini et al. 2021 for more (https://arxiv.org/abs/2106.03846).

    To initialise, pass 'emulator_filenames' in the relevant cobaya block:
    A dictionary of spectrum type (TT, TE, ...) and file names of corresponding emulator models.

    This code is taken from the Cobaya example for custom theory codes and only slightly modified.
    See https://cobaya.readthedocs.io/en/latest/theories_and_dependencies.html.
    Torrado and Lewis, 2020 (https://arxiv.org/abs/2005.05290)

    Attributes
    -------------
    emulator_filenames : dict
        File names of emulators.
    cp_emulators : dict
        CosmoPower emulators.
    cp_pars : list
        List of parameters required by the emulators.
    provider : Provider
        Cobaya provider.
    current_state : dict
        Dict containing current parameters and results.

    Methods
    ---------
    initialize :
        Complete set-up.
    initialize_with_provider :
        Initialization after other components initialised.
    get_requirements :
        Returns what the theory code needs to run.
    must_provide :
        Returns what the theory code needs to run.
    get_can_provide :
        Return what the theory code can supply.
    calculate :
        Carry out the calculation.
    get_Cl :
        Return result of calculation (Cl and Dl).
    """

    # I know having this be a class variable is not super elegant, but it's the easiest way to interface with Cobaya
    emulator_filenames: dict = {}

    def initialize(self):
        """
        Called from __init__ to initialise.
        Loads the emulator models.
        """

        # Load models and unify prediction methods
        self.cp_emulators = {}
        for spec_type in list(self.emulator_filenames.keys()):
            if spec_type == "TE":
                self.cp_emulators[spec_type] = cp.cosmopower_PCAplusNN(
                    restore=True, restore_filename=self.emulator_filenames[spec_type]
                )
                self.cp_emulators[spec_type].get_prediction = self.cp_emulators[
                    spec_type
                ].predictions_np
            else:
                self.cp_emulators[spec_type] = cp.cosmopower_NN(
                    restore=True, restore_filename=self.emulator_filenames[spec_type]
                )
                self.cp_emulators[spec_type].get_prediction = self.cp_emulators[
                    spec_type
                ].ten_to_predictions_np

        # Just using a random spec here, would be weird if TT/TE/EE had different requirements
        self.cp_pars = list(list(self.cp_emulators.values())[0].parameters)
        if "h" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("h")] = "H0"

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """

        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {p: None for p in self.cp_pars}

    def must_provide(self, **requirements):
        """
        Return dictionary of parameters that must be provided.
        """
        return {p: None for p in self.cp_pars}

    def get_can_provide(self):
        """
        Return list of quantities that can be provided.
        """
        return ["Cl", "Dl"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Calculate the CMB spectra.
        Calls the CosmoPower emulator models.
        """
        # Prepare hand-off to CosmoPower

        pars_for_cp = {p: [state["params"][p]] for p in self.cp_pars}
        if "H0" in self.cp_pars:
            pars_for_cp["h"] = [pars_for_cp["H0"][0] / 100]

        # Get CMB Cls
        state["Cl"] = {
            "ell": self.cp_emulators[list(self.cp_emulators.keys())[0]].modes
        }  # assume all supply the same ell range}
        state["Dl"] = {"ell": state["Cl"]["ell"]}
        for spec_type in list(self.cp_emulators.keys()):
            state["Cl"][spec_type] = (
                self.cp_emulators[spec_type].get_prediction(pars_for_cp).ravel()
            )
            state["Dl"][spec_type] = (
                state["Cl"][spec_type]
                * self.cp_emulators[spec_type].modes
                * (self.cp_emulators[spec_type].modes + 1)
                / (2 * jnp.pi)
            )

    def get_Cl(self, ell_factor=False, **kwargs):
        """Get the Cls or Dls."""
        if ell_factor:
            return self.current_state["Dl"].copy()
        else:
            return self.current_state["Cl"].copy()


# --------------------------------------#
# pycapse wrapper
# --------------------------------------#


class CobayaTheoryPyCapse(cobaya_theory_Theory):
    """
    Wraps a capse model into a cobaya theory class through the pycapse interface.
    See Bonici, Bianchini, Ruiz-Zapatero 2023 for more (https://arxiv.org/abs/2307.14339).

    To initialise pass in the relevant Cobaya block:
    base_path : str
        Base path where emulator files are stored.
    specs_to_emulate : list
        Spectrum type (TT, TE, ...) of corresponding emulator models.

    This code is taken from the Cobaya example for custom theory codes and only slightly modified.
    See https://cobaya.readthedocs.io/en/latest/theories_and_dependencies.html.
    Torrado and Lewis, 2020 (https://arxiv.org/abs/2005.05290)

    Attributes
    ------------
    base_path : str
        Path of the dictionary containing the emulator files.
    specs_to_emulate : list
        List of spectrum types (from TT, TE, EE, pp) indicating which emulators to load.
    pc_emulators : dict
        Pycapse emulators.
    pc_pars : list
        List of parameters required by the emulators. May not match expected parameter order of emulators,
        as it can be the collection of parameters for a series of emulators requiring different inputs.
    pc_pars_to_reg_pars : list
        Dictionary translating the pycapse parameter names to more commonly used ones.
    provider : Provider
        Cobaya provider.
    current_state : dict
        Dict containing current parameters and results.

    Methods
    ---------
    __init__ :
        Initialises an instance of the class.
    initialize :
        Complete set-up.
    initialize_with_provider :
        Initialization after other components initialised.
    get_requirements :
        Returns what the theory code needs to run.
    must_provide :
        Returns what the theory code needs to run.
    get_can_provide :
        Return what the theory code can supply.
    calculate :
        Carry out the calculation.
    get_Cl :
        Return result of calculation (Dl and Cl).
    """

    # I know having these be a class variables is not super elegant, but it's the easiest way to interface with Cobaya
    base_path: str = "./"
    specs_to_emulate: list = []

    def initialize(self):
        """
        Called from __init__ to initialise.
        Loads the emulator models.
        """

        with open(self.base_path + "nn_setup.json") as f:
            nn_setup = json.load(f)

        l = np.load(self.base_path + "l.npy")
        low_crop_ix = np.argwhere(l == 2)
        self.pc_l = l[low_crop_ix:]

        self.pc_emulators = {}
        self.pc_pars = []
        for spec_type in self.specs_to_emulate:
            weights = np.load(self.base_path + f"weights_{spec_type}_lcdm.npy")
            trained_emu = pc.init_emulator(nn_setup, weights, pc.simplechainsemulator)
            emu = pc.cl_emulator(
                trained_emu,
                self.pc_l,
                np.load(self.base_path + "inMinMax_lcdm.npy"),
                np.load(self.base_path + f"outMinMaxCℓ{spec_type}_lcdm.npy"),
            )
            self.pc_emulators[spec_type] = emu
            self.pc_pars += pc.get_parameters_list(emu)
        self.pc_pars = list(np.unique(self.pc_pars))

        self.pc_pars_to_reg_pars = {
            "ln10As": "logA",
            "ns": "ns",
            "H0": "H0",
            "ωb": "ombh2",
            "ωc": "omch2",
            "τ": "tau",
        }

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """

        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {p: None for p in self.pc_pars}

    def must_provide(self, **requirements):
        """
        Return dictionary of parameters that must be provided.
        """
        return {p: None for p in self.pc_pars}

    def get_can_provide(self):
        """
        Return list of quantities that can be provided.
        """
        return ["Cl", "Dl"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Calculate the CMB spectra.
        Hands off to pycapse emulators.
        """
        # Hand-off to pycapse emulators
        state["Cl"] = {"ell": self.pc_l}
        state["Dl"] = {"ell": self.pc_l}
        for spec_type in self.specs_to_emulate:
            par_order = pc.get_parameters_list(self.pc_emulators[spec_type])
            pars_for_pc = np.array(
                [state["params"][self.pc_pars_to_reg_pars[p]] for p in par_order]
            )
            state["Dl"][spec_type] = pc.compute_Cl(
                pars_for_pc, self.pc_emulators[spec_type]
            )
            state["Cl"][spec_type] = (
                pc.compute_Cl(pars_for_pc, self.pc_emulators[spec_type])
                * 2.0
                * np.pi
                / (state["Cl"]["ell"] * (state["Cl"]["ell"] + 1))
            )

    def get_Cl(self, ell_factor=False, **kwargs):
        """Get the Cls or Dls."""
        if ell_factor:
            return self.current_state["Dl"].copy()
        else:
            return self.current_state["Cl"].copy()


# --------------------------------------#
# BB TEMPLATE FOREGROUND - COBAYA INTERFACE
# --------------------------------------#


class CobayaTheoryBBTemplate(cobaya_theory_Theory):
    """
    Wrapper for a two template BB calculator in a cobaya theory class.

    To initialise, pass 'template_filenames' in the relevant cobaya block:
    A dictionary with 'lensing_B_modes' and 'r' template file names.

    This code is taken from the Cobaya example for custom theory codes and only slightly modified.
    See https://cobaya.readthedocs.io/en/latest/theories_and_dependencies.html.
    Torrado and Lewis, 2020 (https://arxiv.org/abs/2005.05290)

    Attributes
    ------------
    template_filenames : dict
        File names of templates.
    provider : Provider
        Cobaya provider.
    current_state : dict
        Dict containing current parameters and results.
    templates : dict
        Dict containing cropped BB spectrum templates.
    ells : array (int)
        Ell range provided.

    Methods
    ---------
    initialize :
        Complete set-up, loading templates.
    initialize_with_provider :
        Initialization after other components initialised.
    get_requirements :
        Returns what the theory code needs to run.
    must_provide :
        Returns what the theory code needs to run.
    get_can_provide :
        Return what the theory code can supply.
    calculate :
        Carry out the calculation.
    get_Cl :
        Return result of calculation.

    """

    # I know having this be a class variable is not super elegant, but it's the easiest way to interface with Cobaya
    template_filenames: dict = {}

    def initialize(self):
        """
        Called from __init__ to initialise.
        Loads the templates requested and crops them to the right ell range.
        """

        # Read in templates
        # Assumes these are CAMB outputs, such that the first column is ell and the fourth one is BB
        # Cast into numpy arrays for now to make ell slicing easier
        r_template_arr = np.array(
            candl.io.read_file_from_path(self.template_filenames["r"]).T
        )
        lensing_B_modes_template_arr = np.array(
            candl.io.read_file_from_path(self.template_filenames["lensing_B_modes"]).T
        )

        # Determine common ell range
        common_ell = np.sort(
            list(
                set(r_template_arr[0, :]).intersection(
                    set(lensing_B_modes_template_arr[0, :])
                )
            )
        )
        smallest_common_ell = np.amin(common_ell)
        largest_common_ell = np.amax(common_ell)
        self.ells = np.arange(smallest_common_ell, largest_common_ell + 1)

        # Crop to common ell range and hold onto templates
        self.templates = {}
        for kw, template_arr in zip(
            ["r", "lensing_B_modes"], [r_template_arr, lensing_B_modes_template_arr]
        ):
            low_ell_ix = list(template_arr[0, :]).index(int(smallest_common_ell))
            high_ell_ix = list(template_arr[0, :]).index(int(largest_common_ell))
            self.templates[kw] = jnp.array(
                template_arr[3, low_ell_ix : high_ell_ix + 1]
            )

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """

        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {"r": None, "A_L_BB": None}

    def must_provide(self, **requirements):
        """Return dictionary of parameters that must be provided."""
        return {"r": None, "A_L_BB": None}

    def get_can_provide(self):
        """Return list of quantities that can be provided."""
        return ["Cl"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Calculate the CMB spectra. Sums the two templates with their respective amplitudes."""
        # Scale templates and return sum
        BB_spec = (
            state["params"]["r"] * self.templates["r"]
            + state["params"]["A_L_BB"] * self.templates["lensing_B_modes"]
        )
        state["Dl"] = {"BB": BB_spec, "ell": self.ells}
        state["Cl"] = {
            "BB": 2 * jnp.pi * state["Dl"]["BB"] / (self.ells * (self.ells + 1.0)),
            "ell": state["Dl"]["ell"],
        }

    def get_Cl(self, ell_factor=False, **kwargs):
        """Get the Cls or Dls."""
        if ell_factor:
            return self.current_state["Dl"].copy()
        else:
            return self.current_state["Cl"].copy()


# --------------------------------------#
# COSMOPOWER LENSING EMULATOR - COBAYA INTERFACE
# --------------------------------------#


class CobayaTheoryCosmoPowerJAXLensing(cobaya_theory_Theory):
    """
    Wraps CosmoPower-JAX model for lensing into a cobaya theory code.
    See D. Piras, A. Spurio Mancini 2023 and A. Spurio Mancini et al. 2021 for more (https://arxiv.org/abs/2305.06347, https://arxiv.org/abs/2106.03846).

    To initialise, pass 'emulator_filenames' in the relevant cobaya block:
    A dictionary of spectrum types and emulator file names as placed in cosmopower-jax's trained_models/ folder.

    This code is taken from the Cobaya example for custom theory codes and only slightly modified.
    See https://cobaya.readthedocs.io/en/latest/theories_and_dependencies.html.
    Torrado and Lewis, 2020 (https://arxiv.org/abs/2005.05290)

    Attributes
    ------------
    emulator_filenames : dict
        File names of emulators.
    cp_emulators : dict
        CosmoPower emulators.
    cp_pars : list
        List of parameters required by the emulators.
    provider : Provider
        Cobaya provider.
    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    current_state : dict
        Dict containing current parameters and results.

    Methods
    ---------
    initialize :
        Complete set-up.
    initialize_with_provider :
        Initialization after other components initialised.
    get_requirements :
        Returns what the theory code needs to run.
    must_provide :
        Returns what the theory code needs to run.
    get_can_provide :
        Return what the theory code can supply.
    calculate :
        Carry out the calculation.
    get_Cl :
        Return result of calculation (Cl and Dl).

    """

    # I know having this be a class variable is not super elegant, but it's the easiest way to interface with Cobaya
    emulator_filenames: dict = {}

    def initialize(self):
        """Called from __init__ to initialise. Calls the CosmoPower-JAX emulator models."""

        self.cp_emulators = {}
        for fileType in self.emulator_filenames.keys():
            if fileType == "pp":
                self.cp_emulators[fileType] = CPJ(
                    probe="cmb_pp", filename=self.emulator_filenames[fileType] + ".pkl"
                )
            elif fileType == "TE":
                self.cp_emulators[fileType] = CPJ(
                    probe="custom_pca",
                    filename=self.emulator_filenames[fileType] + ".pkl",
                )
            else:
                self.cp_emulators[fileType] = CPJ(
                    probe="custom_log",
                    filename=self.emulator_filenames[fileType] + ".pkl",
                )

        # changing all of the parameter names
        self.cp_pars = list(self.cp_emulators["pp"].parameters)
        if "h" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("h")] = "H0"
        if "omega_b" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("omega_b")] = "ombh2"
        if "omega_cdm" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("omega_cdm")] = "omch2"
        if "n_s" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("n_s")] = "ns"
        if "ln10^{10}A_s" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("ln10^{10}A_s")] = "logA"
        if "tau_reio" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("tau_reio")] = "tau"

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """

        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {p: None for p in self.cp_pars}

    def must_provide(self, **requirements):
        """Return dictionary of parameters that must be provided."""
        return {p: None for p in self.cp_pars}

    def get_can_provide(self):
        """Return list of quantities that can be provided."""
        return ["Cl", "Dl"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Calculate the CMB spectra. Calls the CosmoPower-JAX emulator models."""
        # Prepare hand-off to CosmoPower
        # numpy array creation is faster, but need jnp to get derivs
        # This is the order expected by CosmoPower-JAX

        pars_for_cp = [state["params"][p] for p in self.cp_pars]
        if "H0" in self.cp_pars:
            pars_for_cp[self.cp_pars.index("H0")] /= 100
        pars_for_cp = jnp.array(pars_for_cp)

        # Get CMB Cls
        state["Cl"] = {
            "pp": self.cp_emulators["pp"].predict(pars_for_cp).ravel(),
            "L": self.cp_emulators["pp"].modes,
        }
        state["Dl"] = {
            "pp": state["Cl"]["pp"]
            * self.cp_emulators["pp"].modes
            * self.cp_emulators["pp"].modes
            * (self.cp_emulators["pp"].modes + 1)
            * (self.cp_emulators["pp"].modes + 1)
            / (2 * jnp.pi),
            "L": self.cp_emulators["pp"].modes,
        }
        state["Dl"]["kk"] = state["Dl"]["pp"] * (2 * jnp.pi) / 4

        for spec_type in self.cp_emulators.keys():
            if spec_type != "pp":
                state["Cl"][spec_type] = (
                    self.cp_emulators[spec_type].predict(pars_for_cp).ravel()
                )
                state["Dl"][spec_type] = (
                    state["Cl"][spec_type]
                    * self.cp_emulators[spec_type].modes
                    * (self.cp_emulators[spec_type].modes + 1)
                    / (2 * jnp.pi)
                )
                state["Cl"]["ell"] = self.cp_emulators[spec_type].modes
                state["Dl"]["ell"] = self.cp_emulators[spec_type].modes

    def get_Cl(self, ell_factor=False, **kwargs):
        """Get the Cls or Dls."""
        if ell_factor:
            return self.current_state["Dl"].copy()
        else:
            return self.current_state["Cl"].copy()


class CobayaTheoryCosmoPowerLensing(cobaya_theory_Theory):
    """
    Wraps CosmoPower model for lensing into a cobaya theory code.
    See A. Spurio Mancini et al. 2021 for more (https://arxiv.org/abs/2106.03846).

    To initialise, pass 'emulator_filenames' in the relevant cobaya block:
    A dictionary of spectrum types and emulator file names as placed in cosmopower-jax's trained_models/ folder.

    This code is taken from the Cobaya example for custom theory codes and only slightly modified.
    See https://cobaya.readthedocs.io/en/latest/theories_and_dependencies.html.
    Torrado and Lewis, 2020 (https://arxiv.org/abs/2005.05290)

    Attributes
    -------------
    emulator_filenames : dict
        File names of emulators.
    cp_emulators : dict
        CosmoPower emulators.
    cp_pars : list
        List of parameters required by the emulators.
    provider : Provider
        Cobaya provider.

    descriptor : str
        A short descriptor.
    par_names : list
        Names of parameters involved in transformation.
    current_state : dict
        Dict containing current parameters and results.

    Methods
    ---------
    __init__ :
        Initialises an instance of the class.
    initialize :
        Complete set-up.
    initialize_with_provider :
        Initialization after other components initialised.
    get_requirements :
        Returns what the theory code needs to run.
    must_provide :
        Returns what the theory code needs to run.
    get_can_provide :
        Return what the theory code can supply.
    calculate :
        Carry out the calculation.
    get_Cl :
        Return result of calculation (Cls and Dls).
    """

    # I know having this be a class variable is not super elegant, but it's the easiest way to interface with Cobaya
    emulator_filenames: dict = {}

    def initialize(self):
        """Called from __init__ to initialise. Calls the CosmoPower emulator models."""

        self.cp_emulators = {}
        for spec_type in self.emulator_filenames.keys():
            if spec_type == "TE" or spec_type == "pp":
                self.cp_emulators[spec_type] = cp.cosmopower_PCAplusNN(
                    restore=True, restore_filename=self.emulator_filenames[spec_type]
                )
                self.cp_emulators[spec_type].get_prediction = self.cp_emulators[
                    spec_type
                ].predictions_np
            else:
                self.cp_emulators[spec_type] = cp.cosmopower_NN(
                    restore=True, restore_filename=self.emulator_filenames[spec_type]
                )
                self.cp_emulators[spec_type].get_prediction = self.cp_emulators[
                    spec_type
                ].ten_to_predictions_np

        # changing all of the parameter names
        self.cp_pars = list(self.cp_emulators["pp"].parameters)
        if "h" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("h")] = "H0"
        if "omega_b" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("omega_b")] = "ombh2"
        if "omega_cdm" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("omega_cdm")] = "omch2"
        if "n_s" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("n_s")] = "ns"
        if "ln10^{10}A_s" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("ln10^{10}A_s")] = "logA"
        if "tau_reio" in self.cp_pars:
            self.cp_pars[self.cp_pars.index("tau_reio")] = "tau"

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """

        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {p: None for p in self.cp_pars}

    def must_provide(self, **requirements):
        """Return dictionary of parameters that must be provided."""
        return {p: None for p in self.cp_pars}

    def get_can_provide(self):
        """Return list of quantities that can be provided."""
        return ["Cl", "Dl"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Calculate the CMB spectra. Calls the CosmoPower emulator models."""
        # Prepare hand-off to CosmoPower
        # numpy array creation is faster, but need jnp to get derivs
        # This is the order expected by CosmoPower-JAX

        pars_for_cp = [state["params"][p] for p in self.cp_pars]
        if "H0" in self.cp_pars:
            pars_for_cp[self.cp_pars.index("H0")] /= 100
        pars_for_cp = jnp.array(pars_for_cp)

        # Get CMB Cls and Dls
        state["Cl"] = {
            "pp": self.cp_emulators["pp"].get_predicion(pars_for_cp).ravel(),
            "L": self.cp_emulators["pp"].modes,
        }
        state["Dl"] = {
            "pp": state["Cl"]["pp"]
            * self.cp_emulators["pp"].modes
            * self.cp_emulators["pp"].modes
            * (self.cp_emulators["pp"].modes + 1)
            * (self.cp_emulators["pp"].modes + 1)
            / (2 * jnp.pi),
            "L": state["Cl"]["L"],
        }
        state["Dl"]["kk"] = state["Dl"]["pp"] * (2 * jnp.pi) / 4

        for spec_type in self.cp_emulators.keys():
            if spec_type != "pp":
                state["Cl"][spec_type] = (
                    self.cp_emulators[spec_type].get_prediction(pars_for_cp).ravel()
                )
                state["Dl"][spec_type] = (
                    state["Cl"][spec_type]
                    * self.cp_emulators[spec_type].modes
                    * (self.cp_emulators[spec_type].modes + 1)
                    / (2 * jnp.pi)
                )
                state["Cl"]["ell"] = self.cp_emulators[spec_type].modes
                state["Dl"]["ell"] = state["Cl"]["ell"]

    def get_Cl(self, ell_factor=False, **kwargs):
        """Get the Cls or Dls."""
        if ell_factor:
            return self.current_state["Dl"].copy()
        else:
            return self.current_state["Cl"].copy()


# --------------------------------------#
# LIKELIHOOD - COBAYA INTERFACE
# --------------------------------------#


class CandlCobayaLikelihood(cobaya_likelihood_Likelihood):
    """
    Wrapper for a candl likelihood into a cobaya.likelihood.Likelihood class.
    Based on example likelihood provided by Cobaya (https://cobaya.readthedocs.io/en/latest/likelihoods.html), Torrado and Lewis, 2020 (https://arxiv.org/abs/2005.05290).
    See the different attributes below that can be set when passing the class to Cobaya.

    Attributes
    ------------
    data_set_file : str
        Path of the data set info yaml file.
    variant : str
        Variant of the likelihood (if requested).
    clear_internal_priors : bool
        Whether to clear internal priors.
    clear_specific_priors : str or list
        String of list of strings of specific priors to clear. If a given parameter appears in an internal prior, the full prior will be cleared (beware for multi-dimensional priors). Doesn't do anything if clear_internal_priors is True.
    lensing : bool
        Whether to use the lensing likelihood.
    feedback : bool
        Whether to print feedback when initialising the likelihood.
    data_selection : any
        Data selection to be used. String, list of string, binary mask, or path to a mask are supported.
    candl_like : candl.Like or candl.LensLike
        Candl likelihood.
    wrapper: any
        Set if wanting to access a wrapper likelihood (e.g. clipy).
    additional_args : dict
        Additional arguments to be passed to the likelihood at initialisation.

    Methods
    -----------
    __init__ :
        Initialises an instance of the class.
    initialize :
        Internal set-up (load candl likelihood).
    get_requirements :
        Returns what the likelihood needs to run.
    logp :
        Evaluate the likelihood, calling candl under the hood.

    """

    data_set_file: str = "./"
    variant: str = None
    clear_internal_priors: bool = True
    clear_specific_priors: list = []
    lensing: bool = False
    feedback: bool = True
    data_selection: any = ...
    wrapper: any = None
    additional_args: dict = {}

    def initialize(self):
        """
        Called from __init__ to initialise and complete the setup.
        Loads the candl likelihood.
        """
        # Grab the correct data set
        if self.data_set_file.startswith("candl.data."):
            importlib.import_module("candl.data")
            self.data_set_file = eval(self.data_set_file)

        # Collect arguments
        init_args = {
            "variant": self.variant,
            "feedback": self.feedback,
        }
        if self.data_selection is not ...:
            init_args["data_selection"] = self.data_selection
        init_args = {
            **self.additional_args,
            **init_args,
        }  # add any additional args to be passed, giving priority to explicitly named arguments

        # Check if a candl wrapper to an external likelihood is requested
        if self.wrapper is not None:
            try:
                # Go through supported wrappers
                if self.wrapper == "clipy":
                    importlib.import_module("clipy")
                    self.candl_like = clipy.clik_candl(
                        self.data_set_file,
                        **self.additional_args,
                    )
                    # Add a list of parameter names that have priors applied
                    self.candl_like.required_prior_parameters = []
                    for p in list(self.candl_like._prior.keys()):
                        if (
                            type(self.candl_like._prior[p]) == list
                            or type(self.candl_like._prior[p]) == tuple
                        ):  # Sometimes clipy returns lists or tuples here
                            for pp in self.candl_like._prior[p]:
                                self.candl_like.required_prior_parameters.append(pp)
                        elif type(self.candl_like._prior[p]) == str:
                            self.candl_like.required_prior_parameters.append(p)
            except:
                raise Exception(
                    f"candl: wrapper likelihood {self.wrapper} could not be initialised!"
                )
            if self.candl_like is None:
                raise Exception(
                    f"candl: wrapper likelihood {self.wrapper} could not be initialised!"
                )
            else:
                return

        # Initialise the candl likelihood
        try:
            if self.lensing:
                self.candl_like = candl.LensLike(
                    self.data_set_file,
                    **init_args,
                )
            else:
                self.candl_like = candl.Like(
                    self.data_set_file,
                    **init_args,
                )
        except:
            raise Exception("candl: likelihood could not be initialised!")

        # by default clear internal priors and assume these are taken care off by Cobaya
        if self.clear_internal_priors:
            self.candl_like.priors = []
        else:
            if len(self.clear_specific_priors) > 0:
                if isinstance(self.clear_specific_priors, str):
                    self.clear_specific_priors = [self.clear_specific_priors]
                new_priors = []
                for ignore_par in self.clear_specific_priors:
                    for i, prior in enumerate(self.candl_like.priors):
                        if ignore_par in prior.par_names:
                            continue
                        else:
                            new_priors.append(prior)
                self.candl_like.priors = new_priors

    def get_requirements(self):
        """Return dictionary of parameters that are needed"""
        # Cls
        required_pars = {"Cl": {}}
        for spec in self.candl_like.unique_spec_types:
            if spec == "kk":
                required_pars["Cl"][
                    "pp"
                ] = (
                    self.candl_like.ell_max
                )  # Cobaya works with pp for lensing by default rather than kk
            else:
                required_pars["Cl"][spec] = self.candl_like.ell_max

        # Nuisance parameters
        for par in self.candl_like.required_nuisance_parameters:
            required_pars[par] = None

        # Any additional priors
        for par in self.candl_like.required_prior_parameters:
            if not par in list(required_pars.keys()):
                required_pars[par] = None

        return required_pars

    def logp(self, **params):
        """Calculate the log-likelihood by calling candl."""
        # Grab the theory spectra
        Dls = self.provider.get_Cl(
            ell_factor=True, units="muK2"
        )  # grab Dls (ell_factor=True)

        # Cover both lensing conventions if required
        if "pp" in Dls:
            Dls["kk"] = Dls["pp"] * jnp.pi / 2.0

        # Crop spectra to correct ell range
        start_ix = np.argwhere(Dls["ell"] == self.candl_like.ell_min)[0][0]
        stop_ix = np.argwhere(Dls["ell"] == self.candl_like.ell_max)[0][0] + 1

        # Assume spectra handed start at ell=0
        for ky in list(Dls.keys()):
            if ky != "ell":
                if ky in ["pp", "kk"]:
                    Dls[ky] = Dls[ky][start_ix:stop_ix]
                else:
                    Dls[ky.upper()] = Dls[ky][start_ix:stop_ix]
                    if ky.lower() == ky:
                        del Dls[ky]

        pars_to_pass = params
        pars_to_pass["Dl"] = Dls

        # Hand off to the likelihood
        logl = self.candl_like.log_like(pars_to_pass)

        return np.float32(logl)


def get_cobaya_info_dict_for_like(
    like,
    name="candl_like",
    data_selection=...,
    clear_internal_priors=True,
    clear_specific_priors=[],
    feedback=True,
    wrapper=None,
    additional_args={},
):
    """
    Thin wrapper for CandlCobayaLikelihood that returns the class with the requested data set in the format expected by Cobaya.
    Note that since Cobaya prefers to instantiate likelihoods itself, this will instantiate a new instance of the likelihood.
    That means, any modifications to the likelihood object you pass will not be reflected in the likelihood that Cobaya uses.
    Still, the wrapper will try to copy over any data selection - it's worth double checking this with the feedback printout.
    Use the optional keywords to pass arguments to the new Cobaya likelihood object during initialisation.
    Note that candl internal priors are cleared by default, be sure to either apply these in the Cobaya parameter block or override the default behaviour.

    Parameters
    ---------------
    like: candl.Like, candl.LensLike
        A candl likelihood to be used.
    name: str
        Name to give the likelihood in Cobaya
    data_selection : any
        Data selection to be used. None, string, list of string, binary mask, or path to a mask are supported.
    clear_internal_priors : bool
        Whether to clear internal priors.
    clear_specific_priors : str or list
        String or list of strings of specific priors to clear. If a given parameter appears in an internal prior, the full prior will be cleared (beware for multi-dimensional priors). Doesn't do anything if clear_internal_priors is True.
    feedback : bool
        Whether to print feedback when initialising the likelihood.
    wrapper : str
        Identifier of a supported external wrapper likelihood.
    additional_args: dict
        Additional arguments to be passed to the likelihood at initialisation.

    Returns
    ---------------
    dict:
        Dictionary to use in Cobaya's 'likelihood' entry.
    """

    # Throw up a warning about how Cobaya re-instantiates likelihoods
    print(
        "Warning: Cobaya will re-instantiate the likelihood object and earlier modifications (such as manually overwriting attributes, the data model, or the priors) you may have done to your instance will likely not apply unless you have specified them here again."
    )

    # Construct Cobaya dictionary
    cobaya_info = {
        name: {
            "external": CandlCobayaLikelihood,
            "data_set_file": like.data_set_file,
            "clear_internal_priors": clear_internal_priors,
            "clear_specific_priors": clear_specific_priors,
            "feedback": feedback,
            "wrapper": wrapper,
            "additional_args": additional_args,
        }
    }

    # Automatically detect some wrappers
    if like.data_set_file[-5:] == ".clik":
        if cobaya_info[name]["wrapper"] is None:
            cobaya_info[name]["wrapper"] = "clipy"

    # Try to copy over existing data selection
    if "data_selection" in like.data_set_dict:
        # Some data selection has already been done
        if data_selection is not ...:
            # Overwrite manually if requested
            print(
                "Warning: overwriting the data selection that was already set in the likelihood with the one provided to the cobaya dictionary generation function."
            )
            cobaya_info[name]["data_selection"] = data_selection
        else:
            cobaya_info[name]["data_selection"] = like.data_set_dict["data_selection"]
    else:
        # No data selection in the likelihood, but perhaps here
        if data_selection is not ...:
            cobaya_info[name]["data_selection"] = data_selection

    # Add flag for lensing likelihoods
    if isinstance(like, candl.LensLike):
        cobaya_info[name]["lensing"] = True

    return cobaya_info


# --------------------------------------#
# PACKAGE LIKELIHOOD FOR MONTEPYTHON
# --------------------------------------#


def get_montepython_nuisance_param_block_for_like(like):
    """
    Prints out info needed by Montepython .param file for nuisance parameters.

    Parameters
    ---------------
    like: candl.Like
        Likelihood to be used.

    Returns
    ---------------
    None

    """

    for nuisance_par in like.required_nuisance_parameters:
        for prior in like.priors:
            if nuisance_par in prior.par_names:
                prior_central = prior.central_value[prior.par_names.index(nuisance_par)]
                prior_width = np.sqrt(
                    np.diag(prior.prior_covariance)[prior.par_names.index(nuisance_par)]
                )
                mp_str = f"data.parameters['{nuisance_par}'] = [{prior_central}, -1, -1, {prior_width}, 1, 'nuisance']"
                print(mp_str)

    return


# --------------------------------------#
# PAR -> Dls SHORTCUTS FOR VARIOUS THEORY CODES
# --------------------------------------#


def get_CobayaTheory_pars_to_theory_specs_func(theory_calc):
    """
    Helper that returns a simple python function that moves from parameters to spectra using a cobaya.theory.Theory instances.

    Parameters
    ---------------
    theory_calc : cobaya.theory.Theory
        Theory code to calculate theory Dls.

    Returns
    ---------------
    func
        Function that takes a dictionary of parameter values, ell_max, and ell_min (optional) as input and returns a dictionary of CMB spectra (Dl).

    """

    def pars_to_theory_specs(pars, ell_high_cut, ell_low_cut=2):
        # Calculate theory Dls
        new_pars = {"params": deepcopy(pars)}
        theory_calc.calculate(new_pars)

        # Figure out ell range
        N_ell = ell_high_cut - ell_low_cut + 1

        theory_start_ix = (
            np.amax((new_pars["Dl"]["ell"][0], ell_low_cut)) - new_pars["Dl"]["ell"][0]
        )
        theory_stop_ix = (
            np.amin((new_pars["Dl"]["ell"][-1], ell_high_cut))
            + 1
            - new_pars["Dl"]["ell"][0]
        )

        like_start_ix = np.amax((new_pars["Dl"]["ell"][0], ell_low_cut)) - ell_low_cut
        like_stop_ix = (
            np.amin((new_pars["Dl"]["ell"][-1], ell_high_cut)) + 1 - ell_low_cut
        )

        # Slice spectra
        Dls = {"ell": np.arange(ell_low_cut, ell_high_cut + 1)}
        for ky in new_pars["Dl"]:
            if ky != "ell":
                Dls[ky] = jnp.zeros(N_ell)
                Dls[ky] = jax_optional_set_element(
                    Dls[ky],
                    np.arange(like_start_ix, like_stop_ix),
                    new_pars["Dl"][ky][theory_start_ix:theory_stop_ix],
                )
        return Dls

    return pars_to_theory_specs


def get_CosmoPowerJAX_pars_to_theory_specs_func(emulator_filenames):
    """
    Helper that returns a simple python function that moves from parameters to spectra using CosmoPower.
    See D. Piras, A. Spurio Mancini 2023 and A. Spurio Mancini et al. 2021 for more (https://arxiv.org/abs/2305.06347, https://arxiv.org/abs/2106.03846).
    Assumes that all emulators have the same input parameters in the same order.

    Parameters
    ---------------
    emulator_filenames : dict
        Dictionary of spectrum types and emulator file names (if they are placed in cosmopower-jax's trained_models/ folder), or the full path of the files, including in either case the file ending '.pkl' or '.npz'.

    Returns
    ---------------
    func
        Function that takes a dictionary of parameter values, ell_max, and ell_min (optional) as input and returns a dictionary of CMB spectra (Dl).


    """

    cp_emulators = {}
    for spec_type in list(emulator_filenames.keys()):
        if spec_type == "TE":
            try:
                cp_emulators[spec_type] = CPJ(
                    probe="custom_pca", filename=emulator_filenames[spec_type]
                )
            except:
                cp_emulators[spec_type] = CPJ(
                    probe="custom_pca", filepath=emulator_filenames[spec_type]
                )
        else:
            try:
                cp_emulators[spec_type] = CPJ(
                    probe="custom_log", filename=emulator_filenames[spec_type]
                )
            except:
                cp_emulators[spec_type] = CPJ(
                    probe="custom_log", filepath=emulator_filenames[spec_type]
                )

    # Grab input parameter order
    cp_pars = list(list(cp_emulators.values())[0].parameters)
    if "h" in cp_pars:
        cp_pars[cp_pars.index("h")] = "H0"

    def pars_to_theory_specs(pars, ell_high_cut, ell_low_cut=2):
        # Hand-off to CosmoPower-JAX
        pars_for_cp = jnp.zeros(len(cp_pars))
        for p in pars:
            if p in cp_pars:
                pars_for_cp = jax_optional_set_element(
                    pars_for_cp, cp_pars.index(p), pars[p]
                )

        if "H0" in cp_pars:
            pars_for_cp = jax_optional_set_element(
                pars_for_cp, cp_pars.index("H0"), pars["H0"] / 100
            )

        # Get CMB Dls
        Dls = {"ell": np.arange(ell_low_cut, ell_high_cut + 1)}
        for spec_type in list(cp_emulators.keys()):
            # Figure out ell range
            N_ell = ell_high_cut - ell_low_cut + 1

            theory_start_ix = (
                np.amax((cp_emulators[spec_type].modes[0], ell_low_cut))
                - cp_emulators[spec_type].modes[0]
            )
            theory_stop_ix = (
                np.amin((cp_emulators[spec_type].modes[-1], ell_high_cut))
                + 1
                - cp_emulators[spec_type].modes[0]
            )

            like_start_ix = (
                np.amax((cp_emulators[spec_type].modes[0], ell_low_cut)) - ell_low_cut
            )
            like_stop_ix = (
                np.amin((cp_emulators[spec_type].modes[-1], ell_high_cut))
                + 1
                - ell_low_cut
            )

            # Slice spectra
            this_Dl = (
                cp_emulators[spec_type].predict(pars_for_cp).ravel()
                * cp_emulators[spec_type].modes
                * (cp_emulators[spec_type].modes + 1)
                / (2 * jnp.pi)
            )
            Dls[spec_type] = jnp.zeros(N_ell)
            Dls[spec_type] = jax_optional_set_element(
                Dls[spec_type],
                np.arange(like_start_ix, like_stop_ix),
                this_Dl[theory_start_ix:theory_stop_ix],
            )

        return Dls

    return pars_to_theory_specs


def get_CosmoPower_pars_to_theory_specs_func(emulator_filenames):
    """
    Helper that returns a simple python function that moves from parameters to spectra using CosmoPower.
    See A. Spurio Mancini et al. 2021 for more (https://arxiv.org/abs/2106.03846).

    Parameters
    ---------------
    emulator_filenames : dict
        Dictionary of spectrum types and emulator file names.

    Returns
    ---------------
    func
        Function that takes a dictionary of parameter values, ell_max, and ell_min (optional) as input and returns a dictionary of CMB spectra (Dl).

    """

    # Load models and unify prediction methods
    cp_emulators = {}
    for spec_type in list(emulator_filenames.keys()):
        if spec_type == "TE":
            cp_emulators[spec_type] = cp.cosmopower_PCAplusNN(
                restore=True, restore_filename=emulator_filenames[spec_type]
            )
            cp_emulators[spec_type].get_prediction = cp_emulators[
                spec_type
            ].predictions_np
        else:
            cp_emulators[spec_type] = cp.cosmopower_NN(
                restore=True, restore_filename=emulator_filenames[spec_type]
            )
            cp_emulators[spec_type].get_prediction = cp_emulators[
                spec_type
            ].ten_to_predictions_np

    def pars_to_theory_specs(pars, ell_high_cut, ell_low_cut=2):
        # Hand-off to CosmoPower
        pars_for_cp = {str(p): [float(np.atleast_1d(pars[p])[0])] for p in pars}
        if "H0" in pars_for_cp and not "h" in pars_for_cp:
            pars_for_cp["h"] = [pars_for_cp["H0"][0] / 100]

        # Get CMB Dls
        Dls = {"ell": np.arange(ell_low_cut, ell_high_cut + 1)}
        for spec_type in list(cp_emulators.keys()):
            # Figure out ell range
            N_ell = ell_high_cut - ell_low_cut + 1

            theory_start_ix = (
                np.amax((cp_emulators[spec_type].modes[0], ell_low_cut))
                - cp_emulators[spec_type].modes[0]
            )
            theory_stop_ix = (
                np.amin((cp_emulators[spec_type].modes[-1], ell_high_cut))
                + 1
                - cp_emulators[spec_type].modes[0]
            )

            like_start_ix = (
                np.amax((cp_emulators[spec_type].modes[0], ell_low_cut)) - ell_low_cut
            )
            like_stop_ix = (
                np.amin((cp_emulators[spec_type].modes[-1], ell_high_cut))
                + 1
                - ell_low_cut
            )

            # Slice spectra
            this_Dl = (
                cp_emulators[spec_type].get_prediction(pars_for_cp).ravel()
                * cp_emulators[spec_type].modes
                * (cp_emulators[spec_type].modes + 1)
                / (2 * jnp.pi)
            )
            Dls[spec_type] = jnp.zeros(N_ell)
            Dls[spec_type] = jax_optional_set_element(
                Dls[spec_type],
                np.arange(like_start_ix, like_stop_ix),
                this_Dl[theory_start_ix:theory_stop_ix],
            )

        return Dls

    return pars_to_theory_specs


def get_PyCapse_pars_to_theory_specs_func(capse_base_path, specs=["TT", "TE", "EE"]):
    """
    Helper that returns a simple python function that moves from parameters to spectra using PyCapse.
    See Bonici, Bianchini, Ruiz-Zapatero 2023 for more (https://arxiv.org/abs/2307.14339).

    Parameters
    ---------------
    capse_base_path : str
        Path where the PyCapse is located.
    specs : list (optional)
        Which spectra (TT, TE, EE, BB) to try to load.

    Returns
    ---------------
    func
        Function that takes a dictionary of parameter values, ell_max, and ell_min (optional) as input and returns a dictionary of CMB spectra (Dl).

    """

    with open(capse_base_path + "nn_setup.json") as f:
        nn_setup = json.load(f)

    l = np.load(capse_base_path + "l.npy")
    pc_emulators = {}
    for spec_type in specs:
        weights = np.load(capse_base_path + f"weights_{spec_type}_lcdm.npy")
        trained_emu = pc.init_emulator(nn_setup, weights, pc.simplechainsemulator)
        emu = pc.cl_emulator(
            trained_emu,
            l,
            np.load(capse_base_path + "inMinMax_lcdm.npy"),
            np.load(capse_base_path + f"outMinMaxCℓ{spec_type}_lcdm.npy"),
        )
        pc_emulators[spec_type] = emu

    pc_pars_to_reg_pars = {
        "ln10As": "logA",
        "ns": "ns",
        "H0": "H0",
        "ωb": "ombh2",
        "ωc": "omch2",
        "τ": "tau",
    }

    def pars_to_theory_specs(pars, ell_high_cut, ell_low_cut=2):
        # Figure out ell range
        N_ell = ell_high_cut - ell_low_cut + 1

        theory_start_ix = np.amax((l[0], ell_low_cut)) - l[0]
        theory_stop_ix = np.amin((l[-1], ell_high_cut)) + 1 - l[0]

        like_start_ix = np.amax((l[0], ell_low_cut)) - ell_low_cut
        like_stop_ix = np.amin((l[-1], ell_high_cut)) + 1 - ell_low_cut

        # Hand-off to pycapse emulators
        Dls = {"ell": np.arange(ell_low_cut, ell_high_cut + 1)}
        for spec_type in specs:
            par_order = pc.get_parameters_list(pc_emulators[spec_type])
            pars_for_pc = np.array([pars[pc_pars_to_reg_pars[p]] for p in par_order])
            Dls[spec_type] = jnp.zeros(N_ell)
            Dls[spec_type] = jax_optional_set_element(
                Dls[spec_type],
                np.arange(like_start_ix, like_stop_ix),
                pc.compute_Cl(pars_for_pc, pc_emulators[spec_type])[
                    theory_start_ix:theory_stop_ix
                ],
            )

        return Dls

    return pars_to_theory_specs


def get_CAMB_pars_to_theory_specs_func(CAMB_pars):
    """
    Helper that returns a simple python function that moves from parameters to spectra using CAMB.

    Parameters
    ---------------
    CAMB_pars : camb.model.CAMBparams
        CAMBparams for the model. Defines accuracy, ell range, etc.

    Returns
    ---------------
    func
        Function that takes a dictionary of parameter values, ell_max, and ell_min (optional) as input and returns a dictionary of CMB spectra (Dl).

    """

    CAMB_ix = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}

    def pars_to_theory_specs(pars, ell_high_cut, ell_low_cut=2):
        # Set cosmological parameters
        CAMB_pars.set_cosmology(
            H0=pars["H0"],
            ombh2=pars["ombh2"],
            omch2=pars["omch2"],
            mnu=0.06,
            omk=0,
            tau=pars["tau"],
        )
        CAMB_pars.InitPower.set_params(
            As=np.exp(pars["logA"]) * 1e-10, ns=pars["ns"], r=0
        )

        # Calculate spectra
        results = camb.get_results(CAMB_pars)
        powers = results.get_cmb_power_spectra(CAMB_pars, CMB_unit="muK")

        # Figure out ell range
        CAMB_ells = jnp.arange(powers["total"].shape[0])

        # Figure out ell range
        N_ell = ell_high_cut - ell_low_cut + 1

        theory_start_ix = np.amax((CAMB_ells[0], ell_low_cut)) - CAMB_ells[0]
        theory_stop_ix = np.amin((CAMB_ells[-1], ell_high_cut)) + 1 - CAMB_ells[0]

        like_start_ix = np.amax((CAMB_ells[0], ell_low_cut)) - ell_low_cut
        like_stop_ix = np.amin((CAMB_ells[-1], ell_high_cut)) + 1 - ell_low_cut

        # Return as dictionary
        Dls = {
            "ell": np.arange(ell_low_cut, ell_high_cut + 1),
            "pp": powers["lens_potential"][theory_start_ix:theory_stop_ix, 0],
            "kk": powers["lens_potential"][theory_start_ix:theory_stop_ix, 0]
            * jnp.pi
            / 2.0,
        }
        for ky in list(CAMB_ix.keys()):
            Dls[ky] = jnp.zeros(N_ell)
            Dls[ky] = jax_optional_set_element(
                Dls[ky],
                np.arange(like_start_ix, like_stop_ix),
                powers["total"][theory_start_ix:theory_stop_ix, CAMB_ix[ky]],
            )

        return Dls

    return pars_to_theory_specs


def get_CLASS_pars_to_theory_specs_func(CLASS_cosmo):
    """
    Helper that returns a simple python function that moves from parameters to spectra using CLASS.

    Parameters
    ---------------
    CLASS_cosmo : classy.Class
        Class for the model to be evaluated. Need to set desired output options, accuracy settings, ell range, etc.

    Returns
    ---------------
    func
        Function that takes a dictionary of parameter values, ell_max, and ell_min (optional) as input and returns a dictionary of CMB spectra (Dl).

    """

    all_class_pars = [
        p.lower() for p in CLASS_cosmo.__dir__()
    ]  # Grab list of parameters that CLASS understands

    def pars_to_theory_specs(pars, ell_high_cut, ell_low_cut=2):

        # Only pass parameters understood by CLASS
        pars_for_class = {}
        for p in pars:
            if p.lower() in all_class_pars:
                pars_for_class[p] = pars[p]

        # Hand off to CLASS
        CLASS_cosmo.set(pars_for_class)
        CLASS_cosmo.compute()
        class_cls = CLASS_cosmo.lensed_cl(ell_high_cut)
        CLASS_cosmo.struct_cleanup()

        # Figure out ell range
        N_ell = ell_high_cut - ell_low_cut + 1

        theory_start_ix = (
            np.amax((class_cls["ell"][0], ell_low_cut)) - class_cls["ell"][0]
        )
        theory_stop_ix = (
            np.amin((class_cls["ell"][-1], ell_high_cut)) + 1 - class_cls["ell"][0]
        )

        like_start_ix = np.amax((class_cls["ell"][0], ell_low_cut)) - ell_low_cut
        like_stop_ix = np.amin((class_cls["ell"][-1], ell_high_cut)) + 1 - ell_low_cut

        # Return as dictionary of Dls
        Dls = {}
        for ky in list(class_cls.keys()):
            if ky == "ell":
                continue
            if ky == "pp":
                Dls[ky] = (
                    class_cls[ky]
                    * ((class_cls["ell"] * (class_cls["ell"] + 1)) ** 2.0)
                    / (2.0 * jnp.pi)
                )[theory_start_ix:theory_stop_ix]
                Dls["kk"] = Dls["pp"] * jnp.pi / 2.0
                continue
            Dls[ky.upper()] = jnp.zeros(N_ell)
            this_Dls = (
                class_cls[ky]
                * (CLASS_cosmo.T_cmb() ** 2)
                * 1e12
                * class_cls["ell"]
                * (class_cls["ell"] + 1)
                / (2 * np.pi)
            )
            Dls[ky.upper()] = jax_optional_set_element(
                Dls[ky.upper()],
                np.arange(like_start_ix, like_stop_ix),
                this_Dls[theory_start_ix:theory_stop_ix],
            )
        Dls["ell"] = np.arange(ell_low_cut, ell_high_cut + 1)

        return Dls

    return pars_to_theory_specs

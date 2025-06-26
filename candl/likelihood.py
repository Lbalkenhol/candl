"""
Main module containing the likelihood and prior classes.

Overview
-----------

Primary CMB likelihood class:

* :class:`Like`

Lensing likelihood class:

* :class:`LensLike`

Gaussian prior:

* :class:`GaussianPrior`

Misc:

* :func:`get_start_stop_ix`
* :func:`cholesky_decomposition`

"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
import candl.io
import candl.transformations.abstract_base
import os

# --------------------------------------#
# GRAB DIRECTORY
# --------------------------------------#

candl_path = os.path.dirname(os.path.realpath(__file__))

# --------------------------------------#
# LIKELIHOOD
# --------------------------------------#


class Like:
    """
    Likelihood class for primary CMB power spectra (TT/TE/EE/BB) used to move from theory Dls and parameter values to log likelihood value.

    Attributes
    --------------
    N_bins : array, int
        The number of band power bins of each spectrum.
    N_bins_total : int
        The total number of band power bins.
    N_ell_bins_theory : int
        The number of ell bins (same for each spectrum).
    N_spectra_total : int
        The number of spectra.
    bins_start_ix : array, int
        The start indices for each spectrum in the long data vector.
    bins_stop_ix : array, int
        The stop indices for each spectrum in the long data vector.
    covariance : array, float
        The band power covariance matrix.
    covariance_chol_dec : array, float
        The cholesky decomposition of the band power covariance matrix.
    crop_mask : array, bool
        A mask to be applies to the original data vector for any subselection.
    data_bandpowers : array, float
        The data band powers, blinded if requested.
    _data_bandpowers : array, float
        The unblinded data band powers - careful!
    _blinding_function : array, float
        ratio of blinded to unblinded band powers - careful!
    data_model : list of candl.transformations.abstract_base.Transformation
        List of transformations to be applied to the theory spectra.
    data_set_dict : dict
        Dictionary read in from data set .yaml file containing all the information needed to initialise the likelihood.
    data_set_file : str
        The file path of the data set .yaml file.
    effective_ells : array, float
        Bin centres for each spectrum.
    effective_frequencies : dict
        Dictionary containing effective frequency information from the corresponding .yaml file.
    ells : array, float
        Angular multipole moments for a single spectrum.
    long_ells : array, float
         Angular multipole moments for all spectra.
    name : str
        Name of the likelihood.
    priors : list, GaussianPrior
        List of priors to be applied.
    required_nuisance_parameters : list, str
        List of nuisance parameters required for the likelihood.
    spec_freqs : list
        List of lists giving the two frequencies of each spectrum.
    spec_order : list
        List of strings specifying the order of spectra in the long data and model vectors.
    spec_types : list
        List of strings specifying the spectrum type (TT, TE, EE).
    unique_spec_types : list
        List of strings specifying the spectrum type (TT, TE, EE) without duplicate entries in no particular order.
    tiled_ells : array, float
        2d array of ell values for unbinned model spectra.
    window_functions : list of arrays (float)
        Band power window functions in order of spectra. One list entry for each spectrum with size (N_theory_bins, N_bins).
        Window functions start at ell=2.

    Methods
    --------------
    __init__ :
        initialises an instance of the class.
    log_like :
        Calculates the log likelihood given a theory CMB spectrum and nuisance parameter values.
    log_like_for_bdp :
        Calculates the log likelihood given a theory CMB spectrum, nuisance parameter values, and band powers.
    prior_logl :
        Calculates the log_like contribution of the priors.
    gaussian_logl :
        Gaussian log likelihood (helper for log_like)
    chi_square :
        Calculates the chi square given a theory CMB spectrum and nuisance parameter values.
    get_model_specs :
        Returns the theory CMB spectra adjusted by the nuisance parameter model.
    bin_model_specs :
        Performs the binning.
    init_data_model :
        Initialises the data model.
    init_priors :
        Initialises the priors.
    crop_for_data_selection :
        Adjusts the attributes of the likelihood for an arbitrary selection of the data.
    generate_crop_mask :
        Generate a boolean mask based on which data to use.
    interpret_crop_hint :
        Interpret a string hint to generate a crop mask.
    get_ell_helpers :
        Returns useful variables surrounding the ells.
    blind_bandpowers :
        Blinds bandpowers.

    """

    def __init__(self, data_set_file, **kwargs):
        """
        Initialise a new instance of the Like class.

        Parameters
        --------------
        data_set_file : str
            The file path of a .yaml file that contains all the necessary information to initialise the likelihood.
            Alternatively, this can be an index file that then points to the data set file. In this case, use the keyword "variant" to specify which variant of the data set to use.
        **kwargs : dict
            Any additional keyword arguments to overwrite the information in the .yaml file.

        Returns
        --------------
        Like
            A new instance of the base likelihood class with all data read in and the set-up completed.
        """

        # Load in the data set yaml file, checking if a variant is requested
        if not "variant" in kwargs:
            kwargs["variant"] = None
        self.data_set_dict, self.data_set_file = candl.io.load_info_yaml(
            data_set_file, kwargs["variant"]
        )

        # Overwrite any info in the data set yaml file by passed kwargs
        for key in kwargs:
            self.data_set_dict[key] = kwargs[key]

        # Specify data path if necessary
        if not "data_set_path" in self.data_set_dict:
            self.data_set_dict["data_set_path"] = (
                "/".join(self.data_set_file.split("/")[:-1]) + "/"
            )

        # Grab fluffy descriptor data
        self.name = candl.io.read_meta_info_from_yaml(self.data_set_dict)

        # Grab spectrum info
        (
            self.spec_order,
            self.spec_types,
            self.spec_freqs,
            self.N_spectra_total,
            self.N_bins,
        ) = candl.io.read_spectrum_info_from_yaml(self.data_set_dict)

        self.unique_spec_types = list(
            np.unique(self.spec_types)
        )  # order is destroyed in this process

        # Grab total number of spectra and bins
        self.N_spectra_total = len(self.spec_order)
        self.N_bins_total = sum(self.N_bins)

        # Get start and stop indices of each spectrum
        self.bins_start_ix, self.bins_stop_ix = get_start_stop_ix(self.N_bins)

        # Load band powers and covariance
        self._data_bandpowers = candl.io.read_file_from_yaml(
            self.data_set_dict, "band_power_file"
        )
        self.covariance = candl.io.read_file_from_yaml(
            self.data_set_dict, "covariance_file"
        )

        # Load beam correlation if present
        self.beam_correlation = None
        if "beam_correlation_file" in self.data_set_dict:
            self.beam_correlation = candl.io.read_file_from_yaml(
                self.data_set_dict, "beam_correlation_file"
            )

        # Set likelihood method to be used
        self.logl_function = self.gaussian_logl
        if "likelihood_form" in self.data_set_dict:
            if self.data_set_dict["likelihood_form"] == "gaussian_beam_detcov":
                if not self.beam_correlation is None:
                    self.logl_function = self.gaussian_logl_beam_and_detcov
        else:
            self.data_set_dict["likelihood_form"] = "gaussian"

        # Load effective frequencies
        self.effective_frequencies = candl.io.read_effective_frequencies_from_yaml(
            self.data_set_dict
        )

        # Read in band passes
        self.bandpasses = {}
        if "bandpasses" in self.data_set_dict:
            for freq in self.data_set_dict["bandpasses"]:
                bandpass_arr = candl.io.read_file_from_path(
                    self.data_set_dict["data_set_path"]
                    + self.data_set_dict["bandpasses"][freq]
                )
                this_bandpass = candl.transformations.abstract_base.BandPass(
                    bandpass_arr
                )
                self.bandpasses[str(freq)] = this_bandpass

        # Load in window functions and grab number of theory ell bins
        self.window_functions = candl.io.read_window_functions_from_yaml(
            self.data_set_dict, self.spec_order, self.N_bins
        )
        self.N_ell_bins_theory = int(jnp.shape(self.window_functions[0])[0])

        # Set ell range
        self.ell_min = 2  # everything starts at ell=2
        self.ell_max = self.N_ell_bins_theory + self.ell_min - 1

        # Preliminary ell helper (important to understand any ell based cropping about to happen)
        self.effective_ells = jnp.array([])
        for i, spec in enumerate(self.spec_order):
            self.effective_ells = jnp.concatenate(
                (
                    self.effective_ells,
                    jnp.dot(
                        jnp.arange(2, self.N_ell_bins_theory + 2),
                        self.window_functions[i],
                    ).flatten(),
                )
            )

        # Blind the band powers if necessary
        self.data_bandpowers = self._data_bandpowers
        if "blinding" in list(self.data_set_dict.keys()):
            if (
                type(self.data_set_dict["blinding"]) == bool
                and self.data_set_dict["blinding"]
            ) or type(self.data_set_dict["blinding"]) == int:
                self.data_bandpowers, self._blinding_function = self.blind_bandpowers(
                    (
                        None
                        if type(self.data_set_dict["blinding"]) == bool
                        else self.data_set_dict["blinding"]
                    ),
                )

        # Get a mask according to any subselection of the data
        self.crop_mask = self.generate_crop_mask()

        # Apply crop mask if needed
        # True == use bin
        # False == do not use bin
        if np.any(self.crop_mask == False):
            self.crop_for_data_selection()

        # Grab cholesky decomposition of the covariance
        # Check if Hartlap correction is requested
        if "hartlap_correction" in list(self.data_set_dict.keys()):
            self.covariance_chol_dec = cholesky_decomposition(
                self.covariance,
                N_sims=self.data_set_dict["hartlap_correction"]["N_sims"],
                N_bins=self.N_bins_total,
            )
        else:
            self.covariance_chol_dec = cholesky_decomposition(self.covariance)

        # Define ell range and grab some helpers
        (
            self.ells,
            self.tiled_ells,
            self.long_ells,
            self.effective_ells,
        ) = self.get_ell_helpers()

        # Expand any transformation blocks in the data model section of the .yaml file
        expanded_data_model_list = []
        for i, data_model_entry in enumerate(self.data_set_dict["data_model"]):
            if len(data_model_entry) == 1 and list(data_model_entry.keys()) == [
                "Block"
            ]:
                expanded_transformations = candl.io.expand_transformation_block(
                    self.data_set_dict, data_model_entry["Block"]
                )
                expanded_data_model_list.extend(expanded_transformations)
            else:
                expanded_data_model_list.append(data_model_entry)
        self.data_set_dict["data_model"] = expanded_data_model_list

        # Initialise the data model
        self.data_model = self.init_data_model()

        # Initialise any requested priors
        self.priors = self.init_priors()

        # Get a list of all the nuisance parameters the likelihood requires
        self.required_nuisance_parameters = []
        for transformation in self.data_model:  # go through all transformations
            for nuisance_par in transformation.param_names:
                self.required_nuisance_parameters.append(nuisance_par)
        self.required_nuisance_parameters = [
            str(p) for p in np.unique(self.required_nuisance_parameters)
        ]

        # Similarly, get a list of all parameters with priors
        self.required_prior_parameters = []
        for prior in self.priors:
            for prior_par in prior.par_names:
                self.required_prior_parameters.append(prior_par)
        self.required_prior_parameters = list(np.unique(self.required_prior_parameters))

        # Output info on initialisation
        candl.io.like_init_output(self)

    @partial(jit, static_argnums=(0,))
    def log_like(self, params):
        """
        Returns the negative log likelihood for a given set of theory Dls and nuisance parameter values.

        Parameters
        --------------
        params : dict
            Dictionary containing theory Dls and nuisance parameter values. Required entries: "Dl", the theory Dls starting from ell=2 and going to ell=2+N_ell_bins_theory. Nuisance parameters for any requested models.

        Returns
        --------------
        float
            Negative log likelihood.
        """

        # Get model spectra
        modified_theory_Dls = self.get_model_specs(params)

        # Bin long data vector
        binned_theory_Dls = self.bin_model_specs(modified_theory_Dls)

        # Calculate logl
        logl = self.logl_function(self._data_bandpowers, binned_theory_Dls)

        # Apply priors
        prior_logl = self.prior_logl(params)

        return -(logl + prior_logl)

    @partial(jit, static_argnums=(0,))
    def log_like_for_bdp(self, params, bdp):
        """
        Returns the negative log likelihood for a given set of theory Dls and nuisance parameter values.
        Shortcut to input band powers is useful to jit up the likelihood function with more flexibility,
        e.g. when calculating the derivative for different mock data sets.

        Parameters
        --------------
        params : dict
            Dictionary containing theory Dls and nuisance parameter values. Required entries: "Dl", the theory Dls starting from ell=2 and going to ell=2+N_ell_bins_theory. Nuisance parameters for any requested models.

        Returns
        --------------
        float
            Negative log likelihood.
        """

        # Get model spectra
        modified_theory_Dls = self.get_model_specs(params)

        # Bin long data vector
        binned_theory_Dls = self.bin_model_specs(modified_theory_Dls)

        # Calculate logl
        logl = self.logl_function(bdp, binned_theory_Dls)

        # Apply priors
        prior_logl = self.prior_logl(params)

        return -(logl + prior_logl)

    @partial(jit, static_argnums=(0,))
    def prior_logl(self, params):
        """
        Returns the positive log likelihood of the priors.

        Parameters
        --------------
        params : dict
            Dictionary containing nuisance parameter values.

        Returns
        --------------
        float
            Positive log likelihood.
        """
        p_logl = 0
        for prior in self.priors:
            p_logl += prior.log_like(params)
        return p_logl

    def gaussian_logl(self, data_bandpowers, binned_theory_Dls):
        """
        Calculate the positive log likelihood, i.e.:
        logl = 0.5 * (x-m) @ C^-1 @ (x-m)
        Uses the cholesky decomposition of the covariance for speed.

        Parameters
        --------------
        data_bandpowers : array, float
            Data band powers
        binned_theory_Dls : array, float
            Model spectra

        Returns
        --------------
        float
            Positive log likelihood.
        """

        # Calculate difference between model and data
        delta_bdp = data_bandpowers - binned_theory_Dls

        # Calculate logl
        chol_fac = jnp.linalg.solve(self.covariance_chol_dec, delta_bdp)
        chisq = jnp.dot(
            chol_fac.T, chol_fac
        )  # equivalent to the straightforward method, i.e. delta @ C^-1 @ delta
        logl = chisq / 2

        return logl

    def gaussian_logl_beam_and_detcov(self, data_bandpowers, binned_theory_Dls):
        """
        Calculate the positive log likelihood, including determinant of the covariance i.e.:
        logl = 0.5 * (x-m) @ C^-1 @ (x-m) + log | cov |
        Adds beam covariance and re-computes the Cholesky decomposition.

        Parameters
        --------------
        data_bandpowers : array, float
            Data band powers
        binned_theory_Dls : array, float
            Model spectra

        Returns
        --------------
        float
            Positive log likelihood.
        """

        # Calculate difference between model and data
        delta_bdp = data_bandpowers - binned_theory_Dls

        # Add beam contribution, recompute Cholesky decomposition
        full_covariance_chol_dec = jnp.linalg.cholesky(
            self.covariance
            + self.beam_correlation * jnp.outer(binned_theory_Dls, binned_theory_Dls)
        )

        # Apply Hartlap correction if requested
        if "hartlap_correction" in self.data_set_dict:
            full_covariance_chol_dec /= np.sqrt(
                (
                    self.data_set_dict["hartlap_correction"]["N_sims"]
                    - self.N_bins_total
                    - 2
                )
                / (self.data_set_dict["hartlap_correction"]["N_sims"] - 1)
            )

        # Calculate logl
        chol_fac = jnp.linalg.solve(full_covariance_chol_dec, delta_bdp)
        chisq = jnp.dot(
            chol_fac.T, chol_fac
        )  # equivalent to the straightforward method, i.e. delta @ C^-1 @ delta
        logl = chisq / 2
        logl += jnp.sum(
            jnp.log(jnp.diag(full_covariance_chol_dec))
        )  # covariance determinant term (only equal to half, but we are adding to the logl, not the chisq, so this is correct)

        return logl

    def chi_square(self, params):
        """
        Returns the chi squared for a given set of theory Dls and nuisance parameter values.
        Adds beam covariance contribution to covariance if present.

        Parameters
        --------------
        params : dict
            Dictionary containing theory Dls and nuisance parameter values. Required entries: "Dl", the theory Dls starting from ell=2 and going to ell=2+N_ell_bins_theory. Nuisance parameters for any requested models.

        Returns
        --------------
        float
            Chi square value
        """

        # Get model spectra
        modified_theory_Dls = self.get_model_specs(params)

        # Bin long data vector
        binned_theory_Dls = self.bin_model_specs(modified_theory_Dls)

        # Calculate difference between model and data
        delta_bdp = self._data_bandpowers - binned_theory_Dls

        # Add beam covariance contribution if necessary
        if not self.beam_correlation is None:
            full_covariance_chol_dec = jnp.linalg.cholesky(
                self.covariance
                + self.beam_correlation
                * jnp.outer(binned_theory_Dls, binned_theory_Dls)
            )
            chol_fac = jnp.linalg.solve(full_covariance_chol_dec, delta_bdp)
        else:
            # Calculate chisq
            chol_fac = jnp.linalg.solve(self.covariance_chol_dec, delta_bdp)

        chisq = jnp.dot(
            chol_fac.T, chol_fac
        )  # equivalent to the straight forward method, i.e. delta @ C^-1 @ delta

        return chisq

    @partial(jit, static_argnums=(0,))
    def get_model_specs(self, params):
        """
        Returns the theory spectra adjusted for the foreground/nuisance model.

        Parameters
        --------------
        params : dict
            Dictionary containing theory Dls and nuisance parameter values. Required entries: "Dl", the theory Dls starting from ell=2 and going to ell=2+N_ell_bins_theory. Nuisance parameters for any requested models.


        Returns
        --------------
        array (float)
            Theory Dls adjusted for the nuisance and foreground model. Covering ell=2-(2+N_ell_bins_theory).
        """

        # Unpack theory Dls into long vector
        modified_theory_Dls = jnp.block([params["Dl"][st] for st in self.spec_types])

        # Apply nuisance modules
        for transformation in self.data_model:
            modified_theory_Dls = transformation.transform(modified_theory_Dls, params)

        return modified_theory_Dls

    @partial(jit, static_argnums=(0,))
    def bin_model_specs(self, model_specs):
        """
        Bin model spectra.

        Parameters
        --------------
        model_specs : array, float
            The spectra to be binned.

        Returns
        --------------
        array (float)
            The binned spectra.
        """

        # Bin long data vector
        binned_theory_Dls = jnp.array([])
        for i, spec in enumerate(self.spec_order):
            binned_theory_Dls = jnp.concatenate(
                (
                    binned_theory_Dls,
                    jnp.dot(
                        model_specs[
                            i
                            * self.N_ell_bins_theory : (i + 1)
                            * self.N_ell_bins_theory
                        ],
                        self.window_functions[i],
                    ),
                )
            )

        return binned_theory_Dls

    def init_data_model(self):
        """
        Initialise the data model, i.e. the series of transformations to be applied to the theory spectra.
        Crucially, this function initialises all the transformation instances and passes them the keywords they need.
        It steps through the attributes needed by the initialisation of each class and supplies these from attributes
        of the likelihood as arguments or info from the relevant section of the .yaml file where the transformation.
        Additionally, it creates the freq_info list of effective frequencies and completes the path of the
        template_file.

        Returns
        --------------
        list of candl.transformations.abstract_base.Transformation
            The initialised transformation instances in the intended order ready to be applied to the theory spectra.
        """

        # Load in the data model
        data_model = []
        if not "data_model" in self.data_set_dict:
            return data_model

        for i_tr in range(len(self.data_set_dict["data_model"])):
            # Get class of the requested transformation and any passed arguments
            full_tr_name, tr_passed_args = candl.io.read_transformation_info_from_yaml(
                self.data_set_dict, i_tr
            )

            # Check whether transformation is deliberately ignored through backdoor
            ignore_transformation = False
            if "force_ignore_transformations" in self.data_set_dict:
                # loop over all transformations (identified by their descriptor) that should be ignored
                for ignore_tr_descriptor in (
                    self.data_set_dict["force_ignore_transformations"].split(", ")
                    if isinstance(
                        self.data_set_dict["force_ignore_transformations"], str
                    )
                    else self.data_set_dict["force_ignore_transformations"]
                ):
                    # This casts the different acceptable formats into a list of strings
                    if "descriptor" in tr_passed_args:
                        if tr_passed_args["descriptor"] == ignore_tr_descriptor:
                            ignore_transformation = True
            if ignore_transformation:
                continue

            # Break the information into module and class name
            module_name = ".".join(full_tr_name.split(".")[:-1])
            tr_name = full_tr_name.split(".")[-1]

            # Load the required module - either candl library or user's own
            try:
                try:
                    tr_module = importlib.import_module(
                        "candl.transformations." + module_name
                    )
                except:
                    tr_module = importlib.import_module(module_name)

                # Check what arguments are required to initialise the Transformation
                tr_signature = inspect.signature(tr_module.__dict__[tr_name]).parameters
                tr_all_args = list(tr_signature)
            except:
                raise Exception(f"Could not locate transformation: {full_tr_name}")

            # Compile all arguments required to initialise the transformation class
            tr_arg_dict = tr_passed_args  # use all passed args
            for arg in tr_all_args:
                # Add attributes from likelihood
                if arg in self.__dict__:
                    tr_arg_dict[arg] = self.__dict__[arg]

                # Add effective frequency information
                if arg == "freq_info":
                    freq_info = []
                    for nu_1, nu_2 in self.spec_freqs:
                        freq_info.append(
                            [
                                self.effective_frequencies[
                                    tr_passed_args["effective_frequencies"]
                                ][nu_1],
                                self.effective_frequencies[
                                    tr_passed_args["effective_frequencies"]
                                ][nu_2],
                            ]
                        )
                    tr_arg_dict["freq_info"] = freq_info
                    del tr_arg_dict["effective_frequencies"]

                # Add bandpass information if necessary
                if arg == "bandpass_info":
                    # Ensure all band passes have pre-calculated the thermodynamic conversion for this
                    # reference frequency
                    for band in self.bandpasses:
                        for (
                            passed_arg
                        ) in (
                            tr_arg_dict
                        ):  # some modules might have more than one reference frequency (e.g. correlation terms)
                            if passed_arg[:6] == "nu_ref":
                                if (
                                    not tr_arg_dict[passed_arg]
                                    in self.bandpasses[band].thermo_conv
                                ):
                                    self.bandpasses[
                                        band
                                    ].calculate_thermodynamic_conversion(
                                        tr_arg_dict[passed_arg]
                                    )

                    # Create list of band pass instances in correct order for spectra
                    bandpass_info = []
                    for nu_1, nu_2 in self.spec_freqs:
                        bandpass_info.append(
                            [self.bandpasses[nu_1], self.bandpasses[nu_2]]
                        )
                    tr_arg_dict["bandpass_info"] = bandpass_info

                # Read in any templates
                if "template_arr" in arg:
                    try:
                        tr_arg_dict[arg] = candl.io.read_file_from_path(
                            f"{candl_path}/{tr_arg_dict[arg.replace('template_arr', 'template_file')]}"
                        )
                    except:
                        try:
                            tr_arg_dict[arg] = candl.io.read_file_from_path(
                                f"{self.data_set_dict['data_set_path']}{tr_arg_dict[arg.replace('template_arr', 'template_file')]}"
                            )
                        except:
                            tr_arg_dict[arg] = candl.io.read_file_from_path(
                                f"{tr_arg_dict[arg.replace('template_arr', 'template_file')]}"
                            )
                    del tr_arg_dict[arg.replace("template_arr", "template_file")]

                # Link any already initialised transformations
                if arg[:26] == "link_transformation_module":
                    # Loop over already initialised modules and see if it's available

                    req_module_name = ".".join(tr_passed_args[arg].split(".")[:-1])
                    req_tr_name = tr_passed_args[arg].split(".")[-1]

                    try:
                        req_module = importlib.import_module(
                            "candl.transformations." + req_module_name
                        )
                    except:
                        req_module = importlib.import_module(req_module_name)

                    for transformation in data_model:
                        if isinstance(transformation, req_module.__dict__[req_tr_name]):
                            tr_arg_dict[arg] = transformation
                            break  # use the first available

                    if isinstance(tr_arg_dict[arg], str):
                        raise Exception(
                            f"Failed to initialise {tr_name} transformation."
                            f"Could not find a {tr_passed_args[arg]} instance to link to."
                        )

            # Initialise the transformation
            this_transformation = tr_module.__dict__[tr_name](**tr_arg_dict)
            data_model.append(this_transformation)

        return data_model

    def init_priors(self):
        """
        Initialise the priors. Similar to init_data_model, this function steps through the priors requested by the user
        and initialises GaussianPrior instances using the passed information.

        Returns
        --------------
        list of GaussianPrior
            The initialised priors ready to be evaluated.
        """

        priors = []
        if "priors" in self.data_set_dict:
            for prior_block in self.data_set_dict["priors"]:
                prior_args = prior_block

                if "prior_std" in prior_args:
                    prior_args["prior_covariance"] = float(prior_args["prior_std"]) ** 2
                    del prior_args["prior_std"]

                # Check if covariance file is passed rather than numerical value
                if isinstance(prior_args["prior_covariance"], str):
                    prior_args["prior_covariance"] = jnp.array(
                        np.loadtxt(
                            self.data_set_dict["data_set_path"]
                            + prior_args["prior_covariance"]
                        )
                    )
                else:
                    # avoid trouble parsing values with scientific 'e' notation
                    prior_args["prior_covariance"] = float(
                        prior_args["prior_covariance"]
                    )

                priors.append(GaussianPrior(**prior_args))

        return priors

    def crop_for_data_selection(self):
        """
        Adjust all relevant attributes of the likelihood for an arbitrary subselection of the data. This function
        manipulates not only all the information the likelihood stores on the number of bins, spectrum types etc.,
        but also crops the band powers, covariance, and window functions.
        """

        # Check if any spectra are completely deselected and can be removed entirely
        spec_deselected = [
            jnp.all(
                self.crop_mask[self.bins_start_ix[i_spec] : self.bins_stop_ix[i_spec]]
                == False
            )
            for i_spec in range(len(self.spec_order))
        ]

        # Update number of bins for each spectrum
        self.N_bins = [
            sum(self.crop_mask[self.bins_start_ix[i_spec] : self.bins_stop_ix[i_spec]])
            for i_spec in range(len(self.spec_order))
            if not spec_deselected[i_spec]
        ]

        # Delete deselected spectra from spectra order
        self.spec_order = [
            self.spec_order[i_spec]
            for i_spec in range(len(self.spec_order))
            if not spec_deselected[i_spec]
        ]

        # Infer types, frequencies, and total number of bins
        self.spec_types = [s[:2] for s in self.spec_order]
        self.N_spectra_total = len(self.spec_order)
        self.N_bins_total = jnp.sum(self.crop_mask)
        self.spec_freqs = [s.split(" ")[1].split("x") for s in self.spec_order]
        self.unique_spec_types = list(np.unique(self.spec_types))

        # Crop band powers, covariance matrix, window functions, and beam covariance (if present)
        self._data_bandpowers = jnp.delete(
            self._data_bandpowers, jnp.invert(self.crop_mask)
        )
        self.data_bandpowers = jnp.delete(
            self.data_bandpowers, jnp.invert(self.crop_mask)
        )
        self.covariance = jnp.delete(
            self.covariance, jnp.invert(self.crop_mask), axis=0
        )
        self.covariance = jnp.delete(
            self.covariance, jnp.invert(self.crop_mask), axis=1
        )

        # Crop window functions
        window_list = []
        for i, wdw in enumerate(self.window_functions):
            if not spec_deselected[i]:
                window_list.append(
                    jnp.delete(
                        wdw,
                        jnp.invert(self.crop_mask)[
                            self.bins_start_ix[i] : self.bins_stop_ix[i]
                        ],
                        axis=1,
                    )
                )
        self.window_functions = window_list

        # Crop beam correlation if present
        if not self.beam_correlation is None:
            self.beam_correlation = jnp.delete(
                self.beam_correlation, jnp.invert(self.crop_mask), axis=0
            )
            self.beam_correlation = jnp.delete(
                self.beam_correlation, jnp.invert(self.crop_mask), axis=1
            )

        # Update start and stop indices
        self.bins_start_ix, self.bins_stop_ix = get_start_stop_ix(self.N_bins)

    def generate_crop_mask(self):
        """
        Generate a crop mask. A crop mask is understood to be a boolean mask to be applied to the uncropped data vector
        with True entries for bins that should be used and False entries for bins that should be ignored.
        The method accessed the information on data selection passed by the user interprets it.

        If multiple string hints are passed then only bins selected by the most hints are used.

        See also: interpret_crop_hint

        Returns
        --------------
        array : bool
            The mask to be applied to the uncropped long data vector.
        """

        # Check if crop is necessary
        if not "data_selection" in self.data_set_dict:
            return np.ones(self.N_bins_total) == 1

        if (
            isinstance(self.data_set_dict["data_selection"], str)
            and self.data_set_dict["data_selection"].lower() == "none"
        ):
            return np.ones(self.N_bins_total) == 1

        if self.data_set_dict["data_selection"] is None:
            return np.ones(self.N_bins_total) == 1

        if isinstance(self.data_set_dict["data_selection"], str):
            if " " in self.data_set_dict["data_selection"]:
                # Generate crop mask based on some string hint
                crop_mask = self.interpret_crop_hint(
                    self.data_set_dict["data_selection"]
                )
            else:
                # Interpret as a relative path to a mask
                crop_mask = candl.io.read_file_from_yaml(
                    self.data_set_dict, "data_selection"
                )
        elif isinstance(self.data_set_dict["data_selection"], list) and isinstance(
            self.data_set_dict["data_selection"][0], str
        ):
            # Generate crop mask based on multiple string hints
            cumulative_crop_mask = np.zeros(self.N_bins_total)
            for this_string_hint in self.data_set_dict["data_selection"]:
                cumulative_crop_mask += self.interpret_crop_hint(this_string_hint)
            crop_mask = np.array(
                cumulative_crop_mask == np.amax(cumulative_crop_mask), dtype=bool
            )
        elif isinstance(self.data_set_dict["data_selection"], list) or isinstance(
            self.data_set_dict["data_selection"], np.ndarray
        ):
            # Can either be series of spectra to use or a boolean mask
            if isinstance(self.data_set_dict["data_selection"][0], str):
                crop_mask = np.array(
                    [
                        1 if spec in self.data_set_dict["data_selection"] else 0
                        for spec in self.spec_order
                    ]
                )
            elif (
                isinstance(self.data_set_dict["data_selection"][0], bool)
                or isinstance(self.data_set_dict["data_selection"][0], np.bool_)
            ) or (
                isinstance(self.data_set_dict["data_selection"][0], int)
                or isinstance(self.data_set_dict["data_selection"][0], np.int64)
            ):
                crop_mask = np.array(self.data_set_dict["data_selection"], dtype=int)
            else:
                # Cannot interpret the list passed
                crop_mask = np.ones(self.N_bins_total)
        else:
            crop_mask = np.ones(self.N_bins_total)
        return crop_mask == 1  # cast to bool array

    def interpret_crop_hint(self, crop_hint, verbose=False):
        """
        Interpret a string hint that indicates a crop mask.
        The intended format is: "(data) (action)", where "(data)" specifies which part of the data is selected and
        "(action)" declares what to do with this selection.
        Understood options for "(data)" are:
        * a specific spectrum as specified in the input .yaml file, e.g. "EE 90x90"
        * spectrum types, e.g. "TT"
        * frequencies and frequency combinations, e.g. "90" or "150x220"
        * ell range, e.g. "ell<650" or "ell>1500"
        Understood options for "(action)" are:
        * "remove" remove this part
        * "only" only keep this part, removing all the rest

        Returns
        --------------
        array : bool
            The mask to be applied to the uncropped long data vector corresponding to this data hint.
        """

        # Split hint into data and action part
        hint_list = crop_hint.split(" ")
        action_hint = hint_list[-1]
        data_hint = " ".join(hint_list[:-1])

        affected_specs = []
        base_msk = np.ones(self.N_bins_total) == 0

        if data_hint in self.spec_order:
            # Specific spectrum targeted
            for i_spec, spec in enumerate(self.spec_order):
                if data_hint == spec:
                    base_msk[self.bins_start_ix[i_spec] : self.bins_stop_ix[i_spec]] = (
                        True
                    )
                    affected_specs.append(self.spec_order[i_spec])

        elif data_hint in self.spec_types:
            # Probably concerns spectra
            for i_spec, st in enumerate(self.spec_types):
                if data_hint == st:
                    base_msk[self.bins_start_ix[i_spec] : self.bins_stop_ix[i_spec]] = (
                        True
                    )
                    affected_specs.append(self.spec_order[i_spec])

        elif data_hint.split("x")[0] in [f for freqs in self.spec_freqs for f in freqs]:
            # Probably concerns frequencies
            for i_spec, freq_pair in enumerate(self.spec_freqs):
                if set(data_hint.split("x")) == set(freq_pair):
                    base_msk[self.bins_start_ix[i_spec] : self.bins_stop_ix[i_spec]] = (
                        True
                    )
                    affected_specs.append(self.spec_order[i_spec])

        elif "ell" in data_hint:
            # ell cut
            ell_cut = float(data_hint.split("ell")[1][1:])
            compare_operation = data_hint.split("ell")[1][0]
            base_msk[eval(f"self.effective_ells{compare_operation}ell_cut")] = True

        else:
            raise Exception(
                f"Did not recognise data hint: {data_hint}\n Using all bins."
            )
            return np.ones(self.N_bins_total) == 1

        # Go through action hints and check whether we need to invert the selection
        if action_hint == "only":
            pass
        elif action_hint == "remove":
            base_msk = np.invert(base_msk)
        else:
            raise Exception(
                f"Did not recognise action: {action_hint}\n Using all bins."
            )
            return np.ones(self.N_bins_total) == 1

        # Feedback
        if len(affected_specs) > 0:
            specs_to_list = affected_specs
            if action_hint == "remove":
                specs_to_list = [
                    spec for spec in self.spec_order if not spec in affected_specs
                ]
            if verbose:
                print(f"Made a mask that keeps only: {specs_to_list}")

        return base_msk

    def get_ell_helpers(self):
        """
        Calculate useful variables concerning the ell range and binning.

        Returns
        --------------
        array : float
            array running over the angular multipoles for one spectrum (2->N_ell_bins_theory+2)
        array : float
            2d array giving the angular multipole range for each spectrum in rows
        array : float
            long array giving the angular multipole range for each spectrum (concatenated)
        array : float
            long array effective containing the effective bin centres (concatenated according to spectra_order).
        """

        ells = jnp.arange(2, self.N_ell_bins_theory + 2)
        tiled_ells = jnp.tile(ells, (self.N_spectra_total, 1))
        long_ells = tiled_ells.flatten()

        effective_ells = jnp.array([])
        for i, spec in enumerate(self.spec_order):
            effective_ells = jnp.concatenate(
                (
                    effective_ells,
                    jnp.dot(
                        jnp.arange(2, self.N_ell_bins_theory + 2),
                        self.window_functions[i],
                    ).flatten(),
                )
            )

        return ells, tiled_ells, long_ells, effective_ells

    def blind_bandpowers(self, seed=None):
        """
        Applies blinding function to bandpowers by multiplying by a random sinusoid function and a slope.
        It's important that the same spectrum types get hit by the same blinding function, i.e. e.g. the same for all TT spectra regardless of frequency.

        Parameters
        ----------
        seed : int, optional
            random seed to use

        Returns
        -------
        array
            blinded band powers
        array
            blinding function
        """

        rng = np.random.default_rng(seed)

        blind_func_by_spec = {}
        for spec_type in np.unique(self.spec_types):
            ells = None
            for i, spec in enumerate(self.spec_order):
                if spec[:2] == spec_type:
                    ells = self.effective_ells[
                        self.bins_start_ix[i] : self.bins_stop_ix[i]
                    ]
                    break

            blind_func = 1.0
            blind_func += rng.uniform(low=-0.9, high=0.9) * (
                (rng.integers(0, np.amax(ells)) - ells) / np.amax(ells)
            ) + rng.uniform(
                low=-0.1, high=0.1
            )  # tilt
            blind_func += rng.uniform(low=-0.5, high=0.5) * np.sin(
                ells / rng.integers(50, 100) + rng.integers(-100, 100)
            )  # acoustic peaks

            blind_func_by_spec[spec_type] = blind_func
        blind_func = np.block(
            [blind_func_by_spec[spec[:2]] for spec in self.spec_order]
        )

        return jnp.array(self._data_bandpowers * blind_func), jnp.array(blind_func)


# --------------------------------------#
# LENSING LIKELIHOOD
# --------------------------------------#


class LensLike:
    """
    Lensing likelihood class used to move from theory Dls and parameter values to the log likelihood value.

    Attributes
    -----------------
    N_bins : array, int
        The number of band power bins of each spectrum.
    N_bins_total : int
        The total number of band power bins.
    N_ell_bins_theory : int
        The number of ell bins (same for each spectrum).
    N_spectra_total : int
        The number of spectra.
    bins_start_ix : array, int
        The start indices for each spectrum in the long data vector.
    bins_stop_ix : array, int
        The stop indices for each spectrum in the long data vector.
    covariance : array, float
        The band power covariance matrix.
    covariance_chol_dec : array, float
        The cholesky decomposition of the band power covariance matrix.
    crop_mask : array, bool
        A mask to be applies to the original data vector for any subselection.
    data_bandpowers : array, float
        The data band powers, blinded if requested.
    _data_bandpowers : array, float
        The unblinded data band powers - careful!
    _blinding_function : array, float
        ratio of blinded to unblinded band powers - careful!
    data_model : list of candl.transformations.abstract_base.Transformation
        List of transformations to be applied to the theory spectra.
    data_set_dict : dict
        Dictionary read in from data set .yaml file containing all the information needed to initialise the likelihood.
    data_set_file : str
        The file path of the data set .yaml file.
    effective_ells : array, float
        Bin centres for each spectrum.
    ells : array, float
        Angular multipole moments for a single spectrum.
    lensing_fiducial_correction : array, float
        M * Cfid, loaded from files.
    long_ells : array, float
         Angular multipole moments for all spectra.
    name : str
        Name of the likelihood.
    priors : list, GaussianPrior
        List of priors to be applied.
    required_nuisance_parameters : list, str
        List of nuisance parameters required for the likelihood.
    spec_order : list
        List of strings specifying the order of spectra in the long data and model vectors.
    spec_types : list
        List of strings specifying the spectrum type (pp).
    unique_spec_types : list
        List of strings specifying the spectrum type (TT, TE, EE) without duplicate entries in no particular order.
    tiled_ells : array, float
        2d array of ell values for unbinned model spectra.
    window_functions : list of arrays (float)
        Band power window functions. Each entry in the N_specs long list is and array with size (N_theory_bins, N_bins).

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    log_like :
        Calculates the log likelihood given a theory CMB spectrum and nuisance parameter values.
    log_like_for_bdp :
        Calculates the log likelihood given a theory CMB spectrum, nuisance parameter values, and band powers.
    prior_logl :
        Calculates the log_like contribution of the priors.
    gaussian_logl :
        Gaussian log likelihood (helper for log_like)
    chi_square :
        Calculates the chi square given a theory CMB spectrum and nuisance parameter values.
    get_model_specs :
        Returns the theory CMB spectra adjusted by the nuisance parameter model.
    bin_model_specs :
        Performs the binning.
    init_data_model :
        Initialises the data model.
    init_priors :
        Initialises the priors.
    crop_for_data_selection :
        Adjusts the attributes of the likelihood for an arbitrary selection of the data.
    generate_crop_mask :
        Generate a boolean mask based on which data to use.
    interpret_crop_hint :
        Interpret a string hint to generate a crop mask.
    get_ell_helpers :
        Returns useful variables surrounding the ells.
    blind_bandpowers :
        Blinds bandpowers.

    """

    def __init__(self, data_set_file, **kwargs):
        """
        Initialise a new instance of the LensLike class.

        Parameters
        --------------
        data_set_file : str
            The file path of a .yaml file that contains all the necessary information to initialise the likelihood.
            Alternatively, this can be an index file that then points to the data set file. In this case, use the keyword "variant" to specify which variant of the data set to use.
        kwargs : dict
            Any additional information to overwrite the information in the .yaml file.

        Returns
        --------------
        LensLike
            A new instance of the base lensing likelihood class with all data read in and the set-up completed.
        """

        # Load in the data set yaml file, checking if a variant is requested
        if not "variant" in kwargs:
            kwargs["variant"] = None
        self.data_set_dict, self.data_set_file = candl.io.load_info_yaml(
            data_set_file, kwargs["variant"]
        )

        # Overwrite any info in the data set yaml file by passed kwargs
        for key in kwargs:
            self.data_set_dict[key] = kwargs[key]

        # Specify data path if necessary
        if not "data_set_path" in self.data_set_dict:
            self.data_set_dict["data_set_path"] = (
                "/".join(self.data_set_file.split("/")[:-1]) + "/"
            )

        # Grab fluffy descriptor data
        self.name = candl.io.read_meta_info_from_yaml(self.data_set_dict)

        # Grab spectrum info
        (
            self.spec_order,
            self.spec_types,
            self.N_bins,
        ) = candl.io.read_spectrum_info_from_yaml(self.data_set_dict, lensing=True)

        self.unique_spec_types = list(
            np.unique(self.spec_types)
        )  # order is destroyed in this process

        # Grab total number of spectra and bins
        self.N_spectra_total = len(self.spec_order)
        self.N_bins_total = sum(self.N_bins)

        # Get start and stop indices of each spectrum
        self.bins_start_ix, self.bins_stop_ix = get_start_stop_ix(self.N_bins)

        # Load band powers and covariance
        self._data_bandpowers = candl.io.read_file_from_yaml(
            self.data_set_dict, "band_power_file"
        )
        self.covariance = candl.io.read_file_from_yaml(
            self.data_set_dict, "covariance_file"
        )

        # Load in window functions and grab number of theory ell bins
        self.window_functions = candl.io.read_window_functions_from_yaml(
            self.data_set_dict, self.spec_order, self.N_bins
        )
        self.N_ell_bins_theory = int(jnp.shape(self.window_functions[0])[0])

        # Set ell range
        self.ell_min = 2  # everything starts at ell=2
        self.ell_max = self.N_ell_bins_theory + self.ell_min - 1

        # Preliminary ell helper (important to understand any ell based cropping about to happen)
        self.effective_ells = jnp.array([])
        for i, spec in enumerate(self.spec_order):
            self.effective_ells = jnp.concatenate(
                (
                    self.effective_ells,
                    jnp.dot(
                        jnp.arange(2, self.N_ell_bins_theory + 2),
                        self.window_functions[i],
                    ).flatten(),
                )
            )

        # Blind the band powers if necessary
        self.data_bandpowers = self._data_bandpowers
        if "blinding" in list(self.data_set_dict.keys()):
            if (
                type(self.data_set_dict["blinding"]) == bool
                and self.data_set_dict["blinding"]
            ) or type(self.data_set_dict["blinding"]) == int:
                self.data_bandpowers, self._blinding_function = self.blind_bandpowers(
                    (
                        None
                        if type(self.data_set_dict["blinding"]) == bool
                        else self.data_set_dict["blinding"]
                    ),
                )

        # Get a mask according to any subselection of the data
        self.crop_mask = self.generate_crop_mask()

        # Apply crop mask if needed
        # True == use bin
        # False == do not use bin
        if np.any(self.crop_mask == False):
            self.crop_for_data_selection()

        # Grab cholesky decomposition of the covariance
        if "hartlap_correction" in list(self.data_set_dict.keys()):
            self.covariance_chol_dec = cholesky_decomposition(
                self.covariance,
                N_sims=self.data_set_dict["hartlap_correction"]["N_sims"],
                N_bins=self.N_bins_total,
            )
        else:
            self.covariance_chol_dec = cholesky_decomposition(self.covariance)

        # Define ell range and grab some helpers
        (
            self.ells,
            self.tiled_ells,
            self.long_ells,
            self.effective_ells,
        ) = self.get_ell_helpers()

        # Initialise the data model
        self.data_model = self.init_data_model()

        # Initialise any requested priors
        self.priors = self.init_priors()

        # Get a list of all the nuisance parameters the likelihood requires
        self.required_nuisance_parameters = []
        for transformation in self.data_model:  # go through all transformations
            for nuisance_par in transformation.param_names:
                self.required_nuisance_parameters.append(nuisance_par)
        self.required_nuisance_parameters = [
            str(p) for p in np.unique(self.required_nuisance_parameters)
        ]

        # Similarly, get a list of all parameters with priors
        self.required_prior_parameters = []
        for prior in self.priors:
            for prior_par in prior.par_names:
                self.required_prior_parameters.append(prior_par)
        self.required_prior_parameters = list(np.unique(self.required_prior_parameters))

        # Assert Likelihood form
        self.data_set_dict["likelihood_form"] = "gaussian"

        # Output info on initialisation
        candl.io.like_init_output(self)

    @partial(jit, static_argnums=(0,))
    def log_like(self, params):
        """
        Returns the negative log likelihood for a given set of theory Dls and nuisance parameter values.

        Parameters
        --------------
        params : dict
            Dictionary containing theory Dls and nuisance parameter values. Required entries: "Dl", the theory Dls starting from ell=2 and going to ell=2+N_ell_bins_theory. Nuisance parameters for any requested models.


        Returns
        --------------
        float
            Negative log likelihood.
        """

        # Get model spectra
        binned_theory_Dls = self.get_model_specs(params)

        # Calculate difference between model and data
        delta_bdp = self._data_bandpowers - binned_theory_Dls

        # Calculate logl
        logl = self.gaussian_logl(delta_bdp)

        # Apply priors
        prior_logl = self.prior_logl(params)

        return -(logl + prior_logl)

    @partial(jit, static_argnums=(0,))
    def log_like_for_bdp(self, params, bdp):
        """
        Returns the negative log likelihood for a given set of theory Dls and nuisance parameter values.
        Shortcut to input band powers is useful to jit up the likelihood function with more flexibility,
        e.g. when calculating the derivative for different mock data sets.

        Parameters
        --------------
        params : dict
            Dictionary containing theory Dls and nuisance parameter values. Required entries: "Dl", the theory Dls starting from ell=2 and going to ell=2+N_ell_bins_theory. Nuisance parameters for any requested models.


        Returns
        --------------
        float
            Negative log likelihood.
        """

        # Get model spectra
        binned_theory_Dls = self.get_model_specs(params)

        # Calculate difference between model and data
        delta_bdp = bdp - binned_theory_Dls

        # Calculate logl
        logl = self.gaussian_logl(delta_bdp)

        # Apply priors
        prior_logl = self.prior_logl(params)

        return -(logl + prior_logl)

    @partial(jit, static_argnums=(0,))
    def prior_logl(self, params):
        """
        Returns the positive log likelihood of the priors.

        Parameters
        --------------
        params : dict
            Dictionary containing nuisance parameter values.

        Returns
        --------------
        float
            Positive log likelihood.
        """
        p_logl = 0
        for prior in self.priors:
            p_logl += prior.log_like(params)
        return p_logl

    def gaussian_logl(self, delta_bdp):
        """
        Calculate the positive log likelihood, i.e.:
        logl = 0.5 * (x-m) @ C^-1 @ (x-m)
        Uses the cholesky decomposition of the covariance for speed.

        Parameters
        --------------
        delta_bdp : array, float
            Difference between data and model band powers

        Returns
        --------------
        float
            Positive log likelihood.
        """

        # Calculate logl
        chol_fac = jnp.linalg.solve(self.covariance_chol_dec, delta_bdp)
        chisq = jnp.dot(
            chol_fac.T, chol_fac
        )  # equivalent to the straightforward method, i.e. delta @ C^-1 @ delta
        logl = chisq / 2

        return logl

    def chi_square(self, params):
        """
        Returns the chi squared for a given set of theory Dls and nuisance parameter values.
        Intentionally not reusing the gaussian_logl function as we might add a beam covariance contribution
        at a later date.

        Parameters
        --------------
        params : dict
            Dictionary containing theory Dls and nuisance parameter values. Required entries: "Dl", the theory Dls starting from ell=2 and going to ell=2+N_ell_bins_theory. Nuisance parameters for any requested models.


        Returns
        --------------
        float
            Chi square value
        """

        # Get model spectra
        binned_theory_Dls = self.get_model_specs(params)

        # Calculate difference between model and data
        delta_bdp = self._data_bandpowers - binned_theory_Dls

        # Calculate chisq
        chol_fac = jnp.linalg.solve(self.covariance_chol_dec, delta_bdp)
        chisq = jnp.dot(
            chol_fac.T, chol_fac
        )  # equivalent to the straight forward method, i.e. delta @ C^-1 @ delta

        return chisq

    @partial(jit, static_argnums=(0,))
    def get_model_specs(self, params):
        """
        Returns the theory spectra adjusted for the foreground/nuisance model.

        Parameters
        --------------
        params : dict
            Dictionary containing theory Dls and nuisance parameter values. Required entries: "Dl", the theory Dls starting from ell=2 and going to ell=2+N_ell_bins_theory. Nuisance parameters for any requested models.


        Returns
        --------------
        array (float)
            Theory Dls adjusted for the nuisance and foreground model. Covering ell=2-(2+N_ell_bins_theory).
        """

        # Unpack theory Dls into long vector
        theory_Dls = jnp.block([params["Dl"][st] for st in self.spec_types])

        # Lensing foreground is already binned
        # Bin long data vector
        modified_theory_Dls = self.bin_model_specs(theory_Dls)

        # Apply nuisance modules
        for transformation in self.data_model:
            modified_theory_Dls = transformation.transform(modified_theory_Dls, params)

        return modified_theory_Dls

    @partial(jit, static_argnums=(0,))
    def bin_model_specs(self, model_specs):
        """
        Bin model spectra.

        Parameters
        --------------
        model_specs : array, float
            The spectra to be binned.

        Returns
        --------------
        array (float)
            The binned spectra.
        """

        # Bin long data vector
        binned_theory_Dls = jnp.array([])
        for i, spec in enumerate(self.spec_order):
            binned_theory_Dls = jnp.concatenate(
                (
                    binned_theory_Dls,
                    jnp.dot(
                        model_specs[
                            i
                            * self.N_ell_bins_theory : (i + 1)
                            * self.N_ell_bins_theory
                        ],
                        self.window_functions[i],
                    ),
                )
            )

        return binned_theory_Dls

    def init_data_model(self):
        """
        Initialise the data model, i.e. the series of transformations to be applied to the theory spectra.
        Crucially, this function initialises all the transformation instances and passes them the keywords they need.
        It steps through the attributes needed by the initialisation of each class and supplies these from attributes
        of the likelihood as arguments or info from the relevant section of the .yaml file where the transformation.
        Additionally, it creates the freq_info list of effective frequencies and completes the path of the
        template_file.

        Returns
        --------------
        list of candl.transformations.abstract_base.Transformation
            The initialised transformation instances in the intended order ready to be applied to the theory spectra.
        """

        # Load in the data model
        data_model = []
        if not "data_model" in self.data_set_dict:
            return data_model

        for i_tr in range(len(self.data_set_dict["data_model"])):
            # Get class of the requested transformation and any passed arguments
            full_tr_name, tr_passed_args = candl.io.read_transformation_info_from_yaml(
                self.data_set_dict, i_tr
            )

            # Check whether transformation is deliberately ignored through backdoor
            ignore_transformation = False
            if "force_ignore_transformations" in self.data_set_dict:
                # loop over all transformations (identified by their descriptor) that should be ignored
                for ignore_tr_descriptor in (
                    self.data_set_dict["force_ignore_transformations"].split(", ")
                    if isinstance(
                        self.data_set_dict["force_ignore_transformations"], str
                    )
                    else self.data_set_dict["force_ignore_transformations"]
                ):
                    # This casts the different acceptable formats into a list of strings
                    if "descriptor" in tr_passed_args:
                        if tr_passed_args["descriptor"] == ignore_tr_descriptor:
                            ignore_transformation = True
            if ignore_transformation:
                continue

            # Break the information into module and class name
            module_name = ".".join(full_tr_name.split(".")[:-1])
            tr_name = full_tr_name.split(".")[-1]

            # Load the required module - either candl library or user's own
            try:
                try:
                    tr_module = importlib.import_module(
                        "candl.transformations." + module_name
                    )
                except:
                    tr_module = importlib.import_module(module_name)

                # Check what arguments are required to initialise the Transformation
                tr_signature = inspect.signature(tr_module.__dict__[tr_name]).parameters
                tr_all_args = list(tr_signature)
            except:
                raise Exception(f"Could not locate transformation: {full_tr_name}")

            # Compile all arguments required to initialise the transformation class
            tr_arg_dict = tr_passed_args  # use all passed args
            for arg in tr_all_args:
                # Add attributes from likelihood
                if arg in self.__dict__:
                    tr_arg_dict[arg] = self.__dict__[arg]

                # Read in any templates
                if "template_arr" in arg:
                    try:
                        tr_arg_dict[arg] = candl.io.read_file_from_path(
                            f"{candl_path}/{tr_arg_dict[arg.replace('template_arr', 'template_file')]}"
                        )
                    except:
                        try:
                            tr_arg_dict[arg] = candl.io.read_file_from_path(
                                f"{self.data_set_dict['data_set_path']}{tr_arg_dict[arg.replace('template_arr', 'template_file')]}"
                            )
                        except:
                            tr_arg_dict[arg] = candl.io.read_file_from_path(
                                f"{tr_arg_dict[arg.replace('template_arr', 'template_file')]}"
                            )
                    del tr_arg_dict[arg.replace("template_arr", "template_file")]

                # Read in M matrices and fiducial correction, if needed
                if arg == "M_matrices":

                    # Start with M matrices, looping over requested corrections
                    M_matrices = dict()
                    for s in tr_arg_dict["Mmodes"]:
                        if s in ["TT", "TE", "EE", "BB", "pp", "kk"]:
                            M_matrices[s] = candl.io.read_lensing_M_matrices_from_yaml(
                                self.data_set_dict["data_set_path"]
                                + tr_arg_dict["M_matrices_folder"],
                                Mtype=s,
                            )
                    tr_arg_dict["M_matrices"] = M_matrices

                    # Now read in fiducial correction
                    fiducial_correction = candl.io.read_file_from_path(
                        self.data_set_dict["data_set_path"]
                        + tr_arg_dict["fiducial_correction_file"]
                    )

                    # If a matrix was passed, use corrections only for the requested spectra
                    if np.ndim(fiducial_correction) == 2:
                        spec_ix = {
                            "TT": 0,
                            "TE": 1,
                            "EE": 2,
                            "BB": 3,
                            "pp": 4,
                            "kk": 4,
                        }  # order expected in M_matrices and in fiducial corrections file
                        sum_msk = np.zeros(5, dtype=bool)
                        for s in list(spec_ix.keys()):
                            if s in tr_arg_dict["Mmodes"]:
                                sum_msk[spec_ix[s]] = True
                        fiducial_correction = jnp.sum(
                            fiducial_correction, axis=1, where=sum_msk
                        )

                    tr_arg_dict["fiducial_correction"] = fiducial_correction

                    # Clean up arguments
                    del tr_arg_dict["M_matrices_folder"]
                    del tr_arg_dict["Mmodes"]
                    del tr_arg_dict["fiducial_correction_file"]

            # Initialise the transformation
            this_transformation = tr_module.__dict__[tr_name](**tr_arg_dict)
            data_model.append(this_transformation)

        return data_model

    def init_priors(self):
        """
        Initialise the priors. Similar to init_data_model, this function steps through the priors requested by the user
        and initialises GaussianPrior instances using the passed information.

        Returns
        --------------
        list of GaussianPrior
            The initialised priors ready to be evaluated.
        """

        priors = []
        if "priors" in self.data_set_dict:
            for prior_block in self.data_set_dict["priors"]:
                prior_args = prior_block
                if "prior_std" in prior_args:
                    prior_args["prior_covariance"] = float(prior_args["prior_std"]) ** 2
                    del prior_args["prior_std"]

                # Check if covariance file is passed rather than numerical value
                if isinstance(prior_args["prior_covariance"], str):
                    prior_args["prior_covariance"] = jnp.array(
                        np.loadtxt(
                            self.data_set_dict["data_set_path"]
                            + prior_args["prior_covariance"]
                        )
                    )
                else:
                    # avoid trouble parsing values with scientific 'e' notation
                    prior_args["prior_covariance"] = float(
                        prior_args["prior_covariance"]
                    )

                priors.append(GaussianPrior(**prior_args))

        return priors

    def crop_for_data_selection(self):
        """
        Adjust all relevant attributes of the likelihood for an arbitrary subselection of the data. This function
        manipulates not only all the information the likelihood stores on the number of bins, spectrum types etc.,
        but also crops the band powers, covariance, and window functions.
        """

        # Check if any spectra are completely deselected and can be removed entirely
        spec_deselected = [
            jnp.all(
                self.crop_mask[self.bins_start_ix[i_spec] : self.bins_stop_ix[i_spec]]
                == False
            )
            for i_spec in range(len(self.spec_order))
        ]

        # Update number of bins for each spectrum
        self.N_bins = [
            sum(self.crop_mask[self.bins_start_ix[i_spec] : self.bins_stop_ix[i_spec]])
            for i_spec in range(len(self.spec_order))
            if not spec_deselected[i_spec]
        ]

        # Delete deselected spectra from spectra order
        self.spec_order = [
            self.spec_order[i_spec]
            for i_spec in range(len(self.spec_order))
            if not spec_deselected[i_spec]
        ]

        # Infer types, frequencies, and total number of bins
        self.spec_types = [s[:2] for s in self.spec_order]
        self.N_spectra_total = len(self.spec_order)
        self.N_bins_total = jnp.sum(self.crop_mask)
        self.unique_spec_types = list(np.unique(self.spec_types))

        # Crop band powers, covariance matrix, window functions
        self._data_bandpowers = jnp.delete(
            self._data_bandpowers, jnp.invert(self.crop_mask)
        )
        self.data_bandpowers = jnp.delete(
            self.data_bandpowers, jnp.invert(self.crop_mask)
        )
        self.covariance = jnp.delete(
            self.covariance, jnp.invert(self.crop_mask), axis=0
        )
        self.covariance = jnp.delete(
            self.covariance, jnp.invert(self.crop_mask), axis=1
        )

        # Crop window functions
        window_list = []
        for i, wdw in enumerate(self.window_functions):
            if not spec_deselected[i]:
                window_list.append(
                    jnp.delete(
                        wdw,
                        jnp.invert(self.crop_mask)[
                            self.bins_start_ix[i] : self.bins_stop_ix[i]
                        ],
                        axis=1,
                    )
                )
        self.window_functions = window_list

        # Update start and stop indices
        self.bins_start_ix, self.bins_stop_ix = get_start_stop_ix(self.N_bins)

    def generate_crop_mask(self):
        """
        Generate a crop mask. A crop mask is understood to be a boolean mask to be applied to the uncropped data vector
        with True entries for bins that should be used and False entries for bins that should be ignored.
        The method accessed the information on data selection passed by the user interprets it.

        If multiple string hints are passed then only bins selected by the most hints are used.

        See also: interpret_crop_hint

        Returns
        --------------
        array : bool
            The mask to be applied to the uncropped long data vector.
        """

        # Check if crop is necessary
        if not "data_selection" in self.data_set_dict:
            return np.ones(self.N_bins_total) == 1

        if (
            isinstance(self.data_set_dict["data_selection"], str)
            and self.data_set_dict["data_selection"].lower() == "none"
        ):
            return np.ones(self.N_bins_total) == 1

        if self.data_set_dict["data_selection"] is None:
            return np.ones(self.N_bins_total) == 1

        if isinstance(self.data_set_dict["data_selection"], str):
            # Generate crop mask based on some string hint
            crop_mask = self.interpret_crop_hint(self.data_set_dict["data_selection"])
        elif isinstance(self.data_set_dict["data_selection"], list) and isinstance(
            self.data_set_dict["data_selection"][0], str
        ):
            # Generate crop mask based on multiple string hints
            cumulative_crop_mask = np.zeros(self.N_bins_total)
            for this_string_hint in self.data_set_dict["data_selection"]:
                cumulative_crop_mask += self.interpret_crop_hint(this_string_hint)
            crop_mask = np.array(
                cumulative_crop_mask == np.amax(cumulative_crop_mask), dtype=bool
            )
        elif isinstance(self.data_set_dict["data_selection"], list):
            # Can either be series of spectra to use or a boolean mask
            if isinstance(self.data_set_dict["data_selection"][0], str):
                crop_mask = np.array(
                    [
                        1 if spec in self.data_set_dict["data_selection"] else 0
                        for spec in self.spec_order
                    ]
                )
            elif isinstance(self.data_set_dict["data_selection"][0], bool):
                crop_mask = np.array(self.data_set_dict["data_selection"], dtype=int)
            else:
                # Cannot interpret the list passed
                crop_mask = np.ones(self.N_bins_total)
        else:
            crop_mask = np.ones(self.N_bins_total)
        return crop_mask == 1  # cast to bool array

    def interpret_crop_hint(self, crop_hint, verbose=False):
        """
        Interpret a string hint that indicates a crop mask.
        The intended format is: "(data) (action)", where "(data)" specifies which part of the data is selected and
        "(action)" declares what to do with this selection.
        Understood options for "(data)" are:
        * spectrum types, e.g. "pp"
        * ell range, e.g. "ell<650" or "ell>1500"
        Understood options for "(action)" are:
        * "remove" remove this part
        * "only" only keep this part, removing all the rest

        Returns
        --------------
        array : bool
            The mask to be applied to the uncropped long data vector corresponding to this data hint.
        """

        # Split hint into data and action part
        data_hint, action_hint = crop_hint.split(" ")

        affected_specs = []
        base_msk = np.ones(self.N_bins_total) == 0
        if data_hint in self.spec_types:
            # Probably concerns spectra
            for i_spec, st in enumerate(self.spec_types):
                if data_hint == st:
                    base_msk[self.bins_start_ix[i_spec] : self.bins_stop_ix[i_spec]] = (
                        True
                    )
                    affected_specs.append(self.spec_order[i_spec])

        elif "ell" in data_hint:
            # ell cut
            ell_cut = float(data_hint.split("ell")[1][1:])
            compare_operation = data_hint.split("ell")[1][0]
            base_msk[eval(f"self.effective_ells{compare_operation}ell_cut")] = True

        else:
            raise Exception(
                f"Did not recognise data hint: {data_hint}\n Using all bins."
            )
            return np.ones(self.N_bins_total) == 1

        # Go through action hints and check whether we need to invert the selection
        if action_hint == "only":
            pass
        elif action_hint == "remove":
            base_msk = np.invert(base_msk)
        else:
            raise Exception(
                f"Did not recognise action: {action_hint}\n Using all bins."
            )
            return np.ones(self.N_bins_total) == 1

        # Feedback
        if len(affected_specs) > 0:
            specs_to_list = affected_specs
            if action_hint == "remove":
                specs_to_list = [
                    spec for spec in self.spec_order if not spec in affected_specs
                ]
            if verbose:
                print(f"Made a mask that keeps only: {specs_to_list}")

        return base_msk

    def get_ell_helpers(self):
        """
        Calculate useful variables concerning the ell range and binning.

        Returns
        --------------
        array : float
            array running over the angular multipoles for one spectrum (2->N_ell_bins_theory+2)
        array : float
            2d array giving the angular multipole range for each spectrum in rows
        array : float
            long array giving the angular multipole range for each spectrum (concatenated)
        array : float
            long array effective containing the effective bin centres (concatenated according to spectra_order).
        """

        ells = jnp.arange(2, self.N_ell_bins_theory + 2)
        tiled_ells = jnp.tile(ells, (self.N_spectra_total, 1))
        long_ells = tiled_ells.flatten()

        effective_ells = jnp.array([])
        for i, spec in enumerate(self.spec_order):
            effective_ells = jnp.concatenate(
                (
                    effective_ells,
                    jnp.dot(
                        jnp.arange(2, self.N_ell_bins_theory + 2),
                        self.window_functions[i],
                    ).flatten(),
                )
            )

        return ells, tiled_ells, long_ells, effective_ells

    def blind_bandpowers(self, seed=None):
        """
        Applies blinding function to bandpowers by multiplying by a random sinusoid function and a slope.
        It's important that the same spectrum types get hit by the same blinding function, i.e. e.g. the same for all TT spectra regardless of frequency.

        Parameters
        ----------
        seed : int, optional
            random seed to use

        Returns
        -------
        array
            blinded band powers
        array
            blinding function
        """

        rng = np.random.default_rng(seed)

        blind_func_by_spec = {}
        for spec_type in np.unique(self.spec_types):
            ells = None
            for i, spec in enumerate(self.spec_order):
                if spec[:2] == spec_type:
                    ells = self.effective_ells[
                        self.bins_start_ix[i] : self.bins_stop_ix[i]
                    ]
                    break

            blind_func = 1.0
            blind_func += rng.uniform(low=-0.9, high=0.9) * (
                (rng.integers(0, np.amax(ells)) - ells) / np.amax(ells)
            ) + rng.uniform(
                low=-0.1, high=0.1
            )  # tilt
            blind_func += rng.uniform(low=-0.5, high=0.5) * np.sin(
                ells / rng.integers(50, 100) + rng.integers(-100, 100)
            )  # acoustic peaks

            blind_func_by_spec[spec_type] = blind_func
        blind_func = np.block(
            [blind_func_by_spec[spec[:2]] for spec in self.spec_order]
        )

        return jnp.array(self._data_bandpowers * blind_func), jnp.array(blind_func)


# --------------------------------------#
# PRIORS
# --------------------------------------#


class GaussianPrior:
    """
    Base class for Gaussian priors.

    Attributes
    -----------------
    central_value : array, float
        The central value of the prior.
    par_names : list, str
        The names of the parameters this prior acts on.
    prior_covariance : array, float
        The covariance matrix.
    prior_covariance_chol : array, float
        Cholesky decomposition of the covariance matrix.

    Methods
    ----------------
    __init__ :
        initialises an instance of the class.
    log_like :
        Calculates the positive log likelihood of the prior.

    """

    def __init__(self, central_value, prior_covariance, par_names):
        """
        Initialise a new instance of the GaussianPrior class.
        Note that the order of parameters across arguments is expected to be the same, i.e. the [i] central value
        corresponds to the [i,i] entry in the covariance and the [i] parameter name.

        Parameters
        --------------
        central_value : float or array (float)
            Central values of the prior
        prior_covariance : float or array (float)
            Covariance of the prior
        par_names : str or list (str)
            List of names the prior acts on.

        Returns
        --------------
        GaussianPrior :
            A new instance of the GaussianPrior class with the set-up completed.
        """

        self.central_value = jnp.atleast_1d(
            central_value
        )  # Make sure central value is a vector
        self.prior_covariance = jnp.atleast_2d(
            prior_covariance
        )  # Make sure covariance is a matrix
        self.prior_covariance_chol = jnp.linalg.cholesky(self.prior_covariance)

        # par names needs to be a list of strings
        if type(par_names) is str:
            self.par_names = [par_names]
        else:
            self.par_names = par_names

    def log_like(self, sampled_pars):
        """
        Calculates the positive log likelihood of the prior.

        Parameters
        --------------
        sampled_pars : dict
            Dictionary that holds the values of all parameters in par_names.

        Returns
        --------------
        float :
            The positive log likelihood of the prior evaluated for the sampled_params.
        """

        delta_pars = (
            jnp.atleast_1d([sampled_pars[par_name] for par_name in self.par_names])
            - self.central_value
        )
        chol_fac = jnp.linalg.solve(self.prior_covariance_chol, delta_pars)
        logl = (
            jnp.dot(chol_fac.T, chol_fac) / 2
        )  # equivalent to the chisq method, i.e. 0.5*(delta @ C^-1 @ delta)
        return logl


# --------------------------------------#
# HELPER FUNCTIONS
# --------------------------------------#


def get_start_stop_ix(N_bins):
    """
    Generates a list of start anad stop indices given the number of bins in each spectrum.

    Parameters
    --------------
    N_bins : array (int)
        List with an int entry for each spectrum giving the number of bins.

    Returns
    --------------
    array (int) :
        The start indices of each spectrum in a long vector.
    array (int) :
        The stop indices of each spectrum in a long vector.
    """

    bins_stop_ix = np.cumsum(N_bins)
    bins_start_ix = np.insert(bins_stop_ix[:-1], 0, 0)

    return bins_start_ix, bins_stop_ix


def cholesky_decomposition(covariance, N_sims=None, N_bins=None):
    """
    Performs the Ccholesky decomposition of the covariance matrix. Stops the program if unsuccessful.

    Parameters
    --------------
    covariance : array (float)
        The matrix to be decomposed
    N_sims : int, optional
        Number of simulations (used for the Hartlap correction).
    N_bins : int, optional
        Number of bins in the data vector (used for the Hartlap correction).

    Returns
    --------------
    array (float) :
        Cholesky decomposition of the input matrix.
    """

    covariance_chol_dec = None

    # Compute cholesky decomposition, make sure it works
    try:
        covariance_chol_dec = jnp.linalg.cholesky(covariance)
    except:
        raise Exception(
            "Band power covariance matrix is not positive definite! Stopping."
        )
        exit(1)

    # Check no nans are present from insufficient precision
    if np.isnan(covariance_chol_dec).any():
        raise Exception(
            "candl: cholesky decomposition contains 'nan'! Check file and try switching on double precision in JAX. Stopping."
        )
        exit(1)

    # Apply Hartlap correction if requested
    if N_sims is not None and N_bins is not None:
        covariance_chol_dec /= np.sqrt((N_sims - N_bins - 2) / (N_sims - 1))

    return covariance_chol_dec

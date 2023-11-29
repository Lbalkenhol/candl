# --------------------------------------#
# IMPORTS
# --------------------------------------#

from montepython.likelihood_class import Likelihood
import candl
import numpy as np
from io_mp import dictitems
import importlib

# --------------------------------------#
# MP INTERFACE
# --------------------------------------#


class candl_mp(Likelihood):
    def __init__(self, path, data, command_line):
        # Regular MontePython initialisation
        Likelihood.__init__(self, path, data, command_line)

        # Grab the correct data set
        if self.data_set_file.startswith("candl.data."):
            importlib.import_module("candl.data")
            self.data_set_file = eval(self.data_set_file)

        # Initialise the candl likelihood
        try:
            if self.lensing:
                self.candl_like = candl.LensLike(
                    self.data_set_file,
                    feedback=self.feedback,
                    data_selection=self.data_selection,
                )
            else:
                self.candl_like = candl.Like(
                    self.data_set_file,
                    feedback=self.feedback,
                    data_selection=self.data_selection,
                )
        except:
            raise Exception("candl: likelihood could not be initialised!")

        # by default clear internal priors and assume these are taken care off by MontePython
        if self.clear_internal_priors:
            self.candl_like.priors = []

        # Communicate to CLASS what spectra need
        required_specs = "lCl"
        if "TT" in self.candl_like.spec_types:
            required_specs += " tCl"
        if "TE" or "EE" in self.candl_like.spec_types:
            required_specs += " pCl"
        if self.lensing:
            required_specs += " tCl pCl"  # usually needed for M matrix corrections of lensing likelihoods
        self.need_cosmo_arguments(data, {"lensing": "yes", "output": required_specs})

        # Set ell_max
        self.need_cosmo_arguments(data, {"l_max_scalars": self.candl_like.ell_max + 1})

        # Set nuisance parameters
        self.nuisance = self.candl_like.required_nuisance_parameters

    def loglkl(self, cosmo, data):
        # Grab theory Cls, convert to right units and shuffle dict keys
        class_Cls = self.get_cl(cosmo)
        like_Cls = {"ell": class_Cls["ell"][2 : self.candl_like.ell_max + 1]}
        for ky in class_Cls:
            if ky != "ell":
                like_Cls[ky.upper()] = (
                    class_Cls[ky]
                    * class_Cls["ell"]
                    * (class_Cls["ell"] + 1.0)
                    / (2.0 * np.pi)
                )
                like_Cls[ky.upper()] = like_Cls[ky.upper()][
                    2 : self.candl_like.ell_max + 1
                ]

        # Grab all parameter values
        pars_for_like = {}
        for key, value in dictitems(data.mcmc_parameters):
            pars_for_like[key] = value["current"] * value["scale"]

        # Deal with tau renaming
        pars_for_like["tau"] = pars_for_like["tau_reio"]

        # Hand off to like
        pars_for_like["Dl"] = like_Cls

        logl = self.candl_like.log_like(pars_for_like)
        return logl

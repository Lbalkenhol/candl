try:
    from candl.lib import *
    import candl
    import candl.data
except ImportError:
    raise RuntimeError('Can not find candl. Try running: pip install candl-like')

from cosmosis.datablock import names, SectionOptions
import numpy as np
import os

class CandlCosmoSISLikelihood:
    """
    A thin wrapper to use candl likelihoods in CosmoSIS.
    """

    def __init__(self, options):
        """
        Read in options from CosmoSIS ini file and initialise requested candl likelihood.
        """

        # Read requested data set from ini and grab a name under which the logl value will be recorded
        data_set_str = options.get_string("data_set", default="")
        if data_set_str[:11] == "candl.data.":
            # This looks like a short-cut to a pre-installed data set, e.g. "candl.data.SPT3G_2018_Lens"
            self.data_set_file = eval(data_set_str)
            self.name = "candl." + data_set_str.split(".")[-1]
        else:
            # Assume this to be a path pointing directly to a data set .yaml file
            self.data_set_file = data_set_str
            self.name = "candl." + self.data_set_file.split("/")[-1][:-5]

        # Read in other options from ini
        self.lensing = options.get_bool('lensing', default=True)
        self.clear_1d_internal_priors = options.get_bool('clear_1d_internal_priors', default=True)
        self.clear_nd_internal_priors = options.get_bool('clear_nd_internal_priors', default=False)# CosmoSIS only has 1d priors implemented
        self.feedback = options.get_bool('feedback', default=True)

        # Optional entries
        init_args = {"feedback": self.feedback}
        
        # Data selection
        if options.has_value("data_selection"):
            self.data_selection = options.get_string("data_selection", default="...")
            data_selection_requested = True
            if isinstance(self.data_selection, str):
                if self.data_selection == "...":
                    data_selection_requested = False
            if data_selection_requested:
                init_args["data_selection"] = self.data_selection

        # Likelihood variant
        if options.has_value("variant"):
            self.variant_str = options.get_string("variant", default=None)
            init_args["variant"] = self.variant_str

        # Force ignore transformations
        if options.has_value("force_ignore_transformations"):
            self.force_ignore_transformations = options.get_string("force_ignore_transformations", default=None)
            init_args["force_ignore_transformations"] = self.force_ignore_transformations

        # Initialise the likelihood
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

        # By default clear internal priors and assume these are taken care off by CosmoSIS
        keep_prior_ix = []
        for i, prior in enumerate(self.candl_like.priors):
            if prior.prior_covariance.shape[0] == 1 and not self.clear_1d_internal_priors:
                keep_prior_ix.append(i)
            elif prior.prior_covariance.shape[0] > 1 and not self.clear_nd_internal_priors:
                keep_prior_ix.append(i)
        self.candl_like.priors = [self.candl_like.priors[i] for i in keep_prior_ix]


    def reformat(self,block):
        """
        Converting from CosmoSIS to candl format
        """

        model_dict = {}
        
        # Load all cosmological parameters and add extra params so that candl understands
        # These should not be needed by candl, but it doesn't hurt in case anyone builds a likelihood that directly depends on cosmological parameters
        cos_par_names      = [param_name for param_sec, param_name in block.keys() if param_sec == 'cosmological_parameters']
        model_dict = {par: block[('cosmological_parameters',par)] for par in cos_par_names}
        model_dict['H0']   = model_dict['h0']*100
        model_dict['ns']   = model_dict['n_s']
        model_dict['logA'] = model_dict['log1e10as']
        
        # Load all nuisance parameters
        nuisance_par_names      = [param_name for param_sec, param_name in block.keys() if param_sec == 'nuisance_parameters']

        # Match any nuisance parameters in candl and restore right cases
        like_nuisance_pars_lowered = [p.lower() for p in self.candl_like.required_nuisance_parameters]
        for i, par in enumerate(nuisance_par_names):
            if par in like_nuisance_pars_lowered:
                nuisance_par_names[i] = self.candl_like.required_nuisance_parameters[like_nuisance_pars_lowered.index(par)]

        for par in nuisance_par_names:
            model_dict[par] = block[('nuisance_parameters',par)]# CosmoSIS doesn't care about cases, so putting them in is easy
        
        # Read in Cls from CosmoSIS and save them in dict.
        # CosmoSIS outputs CAMB Cls in unit of l(l+1)/(2pi)
        # For pp it's ell * (ell + 1) / (2 * np.pi) - i.e. missing the customary extra ell * (ell + 1) wrt eg the CAMB standard
        # This matches candl expectations for primary CMB - but we need to convert this for lensing
        ell = block[names.cmb_cl, 'ell']
        cl_tt = block[names.cmb_cl, 'tt']
        cl_ee = block[names.cmb_cl, 'ee']
        cl_te = block[names.cmb_cl, 'te']
        cl_bb = block[names.cmb_cl, 'bb']
        cl_pp = block[names.cmb_cl, 'pp'] * ell * (ell + 1)
        cl_kk = cl_pp * np.pi / 2.0

        # Figure out ell range of supplied spectra w.r.t. the expectation of the likelihood
        N_ell = self.candl_like.ell_max - self.candl_like.ell_min + 1

        theory_start_ix = (
            np.amax((ell[0], self.candl_like.ell_min))
            - ell[0]
        )
        theory_stop_ix = (
            np.amin((ell[-1], self.candl_like.ell_max))
            + 1
            - ell[0]
        )

        like_start_ix = (
            np.amax((ell[0], self.candl_like.ell_min)) - self.candl_like.ell_min
        )
        like_stop_ix = (
            np.amin((ell[-1], self.candl_like.ell_max))
            + 1
            - self.candl_like.ell_min
        )

        # Slot supplied CMB spectra into an array of zeros of the correct length
        # candl will optionally import JAX which should ensure the two methods below run
        model_dict['Dl'] = {'ell': np.arange(self.candl_like.ell_min, self.candl_like.ell_max + 1)}
        for spec_type, spec in zip(["pp", "kk", "TT", "EE", "BB", "TE"], [cl_pp, cl_kk, cl_tt, cl_ee, cl_bb, cl_te]):
            model_dict['Dl'][spec_type] = jnp.zeros(N_ell)
            model_dict['Dl'][spec_type] = jax_optional_set_element(
                model_dict['Dl'][spec_type],
                np.arange(like_start_ix, like_stop_ix),
                spec[theory_start_ix:theory_stop_ix],
            )

        return model_dict
    

    def likelihood(self, block):
        '''Computes loglike'''
        logl  = self.candl_like.log_like(self.reformat(block))
        return float(logl)


def setup(options):
    options = SectionOptions(options)
    return CandlCosmoSISLikelihood(options)

def execute(block, config):
    like    = config.likelihood(block)
    block[names.likelihoods, "%s_like"%config.name] = like
    return 0
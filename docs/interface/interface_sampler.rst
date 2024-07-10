Samplers and Minimisers
=================================================

We provide tools to interface the likelihood with the two most commonly used MCMC samplers in cosmology: `Cobaya <https://github.com/CobayaSampler/cobaya>`_ and `MontePython <https://baudren.github.io/montepython.html>`_.
Moreover, we show how the likelihood can be interfaced with tools that explicitly exploit its differentiability: `BlackJAX <https://github.com/blackjax-devs/blackjax>`_ and `Optax <https://github.com/google-deepmind/optax>`_.

.. warning::

    When using MCMC samplers, be careful to only apply priors once.
    For some data sets candl includes Gaussian priors on cosmological and nuisance parameters, to ensure a fully differentiable pipeline.
    When interfacing candl with Cobaya and MontePython these will be turned off by default and you are expected to specify them within the relevant framework.
    However, this may not be available for all samplers (e.g. BlackJAX).
    Look out for ``clear_internal_priors`` flags in the interface functions and take care not to double count information.

Cobaya
-------------------------------------------------

Interactive
^^^^^^^^^^^^^

To interface with Cobaya interactively, you can use the supplied interface code.
For example:

.. code-block:: python

    import candl
    import candl.data
    import candl.interface
    
    candl_like = candl.Like(candl.data.SPT3G_2018_TTTEEE)
    cobaya_dict = {"likelihood": candl.interface.get_cobaya_info_dict_for_like(candl_like)}

This will interface the likelihood with Cobaya and register all of its requirements.
You can then proceed to populate ``cobaya_dict`` with the the parameters to be sampled etc. and run Cobaya as usual.
By default the candl internal priors are not applied, add ``"clear_internal_priors": False`` to the relevant likelihood block in the dictionary if you want to use them.
Note that Cobaya prefers to initialise the likelihood itself, hence any modifications of ``candl_like`` won't be reflected in the Cobaya likelihood.

.. autofunction:: candl.interface.get_cobaya_info_dict_for_like

Command-Line
^^^^^^^^^^^^^

In order to run Cobaya from the command line it sufficies to include the following lines in the likelihood block.

.. code-block:: yaml

    likelihood:
        candl.interface.CandlCobayaLikelihood:
            data_set_file: candl.data.SPT3G_2018_TTTEEE # data set or path to .yaml file
            lensing: False # Switch on for lensing likelihoods
            feedback: False # Switch on to request feedback from candl initialisation
            data_selection: None # Select a subset of the data set
            clear_internal_priors: True # Switch off to use candl internal priors

Only ``data_set_file`` is required, the other arguments are optional.
See :ref:`here <Data Selection>` for info on data selection.
Again, by default the candl internal priors are not applied, set ``clear_internal_priors: False`` if you want to use them.

Connecting Theory Codes to Cobaya
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the ``interface`` module we also supply an interface to allow for the use of CosmoPower, CosmoPower-JAX, and Capse.jl with Cobaya.
This works through custom ``cobaya.theory.Theory`` subclasses.
For example, to use CosmoPowerJAX in an interactive session with Cobaya, you can use the following code:

.. code-block:: python

    import candl.interface

    cp_emulator_filenames = {"TT": "your_desired_emu_model"}
    cobaya_dict = {"theory": {"CosmoPowerJAX": {"external": candl.interface.CobayaTheoryCosmoPowerJAX,
                                                "emulator_filenames": cp_emulator_filenames}}}

You can then add the other info needed to ``cobaya_dict`` and initialise Cobaya. You can similarly use this interface when launching Cobaya from the command line.

MontePython
-------------------------------------------------

In order to interface candl with Montepython, copy the ``candl_mp`` folder into your ``montepython/likelihoods`` directory.
The folder contains the wrapper in the ``__init__.py`` file and options can be set in the ``candl_mp.data`` file.
The provided template for the latter looks like this:

.. code-block::

    candl_mp.data_set_file = #"candl.data.the_data_set_you_want" or "path/to/info.yaml"
    candl_mp.lensing = False
    candl_mp.feedback = True
    candl_mp.data_selection = None
    candl_mp.clear_internal_priors = True

You have to insert the data set you want in the first line: either the name of a released data set or a path to a .yaml file.
The lensing flag must be set to ``True`` for lensing data sets.

.. note::
    By default the internal priors declared in candl's data set ``.yaml`` file are ignored. It is assumed that Montepython handles these.
    If you want to use the internal priors, set ``clear_internal_priors = False`` in ``candl_mp.data``.

You can then add ``candl_mp`` to the likelihoods in your ``.param`` file.
Here, MontePython also requires you to add all nuisance parameters for the likelihood.
This can be a little tedious, so we provide a helper function to do this for you.
For example, for the SPT-3G 2018 TTTEEE likelihood, run the following python code:

.. code-block:: python
    
    import candl
    import candl.data
    import candl.interface

    candl_like = candl.Like(candl.data.SPT3G_2018_TTTEEE)
    candl.interface.get_montepython_nuisance_param_block_for_like(candl_like)

This will print the nuisance parameter block to the terminal, which you can then copy over to your ``.param`` file.

.. autofunction:: candl.interface.get_montepython_nuisance_param_block_for_like


BlackJAX
-------------------------------------------------

`BlackJAX <https://github.com/blackjax-devs/blackjax>`__ gradient-based samplers for JAX.
We show how to interface with BlackJAX and run NUTS chains in :ref:`the differentiability tutorial <Tutorials and Use>`.
Below is a quick example using the ACT DR4 likelihood and CosmoPower-JAX.
Note that you need to have downloaded the CosmoPower SPT high-accuracy models (available `here <https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/SPT_high_accuracy>`_) and placed them in your ``cosmopower_jax/trained_models`` folder.
First, we initialise the likelihood and the theory code:

.. code-block:: python

    import candl
    import candl.data
    import candl.interface
    import candl.tools

    candl_like = candl.Like(candl.data.ACT_DR4_TTTEEE)
    cp_emulator_filenames = {"TT": "cmb_spt_TT_NN",
                            "TE": "cmb_spt_TE_PCAplusNN",
                            "EE": "cmb_spt_EE_NN"}
    pars_to_theory_specs = candl.interface.get_CosmoPowerJAX_pars_to_theory_specs_func(cp_emulator_filenames)
    like = candl.tools.get_params_to_logl_func(candl_like, pars_to_theory_specs)

In the last line we obtain a function that moves from a dictionary of cosmological parameters to the log likelihood value.
For BlackJAX, it's easier to work with a normalised parameter vector as an input.
So next, we define the Planck 18 best-fit point as our reference and use the Fisher matrix (from this point) to normalise the parameters.

.. code-block:: python

    import jax
    import numpy as np

    fid_pars = {'H0': 67.37, 'ombh2': 0.02233, 'omch2': 0.1198, 'logA': 3.043, 'ns': 0.9652, 'tau': 0.054, 'yp': 1.0}
    fid_pars_vec = np.array([fid_pars[p] for p in par_order])
    par_cov, par_order = candl.tools.get_fisher_matrix(pars_to_theory_specs, candl_like, fid_pars, par_order=None, return_par_order=True)

    def norm_pars(pars):
        return (pars-fid_pars_vec)/np.sqrt(np.diag(par_cov))

    def denorm_pars(pars):
        return pars*np.sqrt(np.diag(par_cov))+fid_pars_vec

    @jax.jit
    def like_normed_vec(input_vec):
        denormed_vec = denorm_pars(input_vec)
        return like({par_order[i]: denormed_vec[i] for i in range(len(par_order))})

Now we have a function that takes a normalised parameter vector and returns the log likelihood value.
Following the NUTS example provided by BlackJAX (`here <https://blackjax-devs.github.io/blackjax/examples/quickstart.html>`_) we define the inference loop.

.. code-block:: python

    # This code is taken from the BlackJAX tutorial
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state
        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states

Finally we feed this into BlackJAX and run the sampler.

.. code-block:: python

    import blackjax
    from datetime import date

    rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))# grab random number for starting point
    nuts = blackjax.nuts(like_normed_vec, step_size=0.1, inverse_mass_matrix=np.ones(len(par_order)))
    initial_state = nuts.init(np.zeros(len(par_order)))
    states = inference_loop(rng_key,
                            nuts.step,
                            initial_state,
                            100)# Number of desired samples
    NUTS_samples = jax.numpy.apply_along_axis(denorm_pars, 1, states.position)

Where in the last line we move from the normalised parameter space back to our familiar one (i.e. get the right units).

Optax
-------------------------------------------------

`Optax <https://github.com/google-deepmind/optax>`__ is a library for gradient-based optimisation written for JAX.
We show how to interface with Optax and run the ADAM minimiser in :ref:`the differentiability tutorial <Tutorials and Use>`.
Below is a quick example using the ACT DR4 likelihood and CosmoPower-JAX.
Note that you need to have downloaded the CosmoPower SPT high-accuracy models (available `here <https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/SPT_high_accuracy>`_) and placed them in your ``cosmopower_jax/trained_models`` folder.
Like before, we initialise the likelihood and the theory code:

.. code-block:: python

    import candl
    import candl.data
    import candl.interface
    import candl.tools

    candl_like = candl.Like(candl.data.ACT_DR4_TTTEEE)
    cp_emulator_filenames = {"TT": "cmb_spt_TT_NN",
                            "TE": "cmb_spt_TE_PCAplusNN",
                            "EE": "cmb_spt_EE_NN"}
    pars_to_theory_specs = candl.interface.get_CosmoPowerJAX_pars_to_theory_specs_func(cp_emulator_filenames)
    like = candl.tools.get_params_to_logl_func(candl_like, pars_to_theory_specs)

Optax also prefers normalised input parameters, so as before we define the Planck 18 best-fit point as our reference and use the Fisher matrix (from this point) to normalise the parameters.

.. code-block:: python

    import jax
    import numpy as np

    fid_pars = {'H0': 67.37, 'ombh2': 0.02233, 'omch2': 0.1198, 'logA': 3.043, 'ns': 0.9652, 'tau': 0.054, 'yp': 1.0}
    par_cov, par_order = candl.tools.get_fisher_matrix(pars_to_theory_specs, candl_like, fid_pars, par_order=None, return_par_order=True)
    par_scales = {par_order[i]: np.sqrt(par_cov[i,i]) for i in range(len(par_order))}

    def transform_to_zero_mean_unit_var(par_dict):
        new_par_dict = {}
        for p in par_dict:
            if p in fid_pars and p in par_order:
                new_par_dict[p] = (par_dict[p] - fid_pars[p])/par_scales[p]
        return new_par_dict

    def transform_from_zero_mean_unit_var(par_dict):
        new_par_dict = {}
        for p in par_dict:
            if p in fid_pars and p in par_order:
                new_par_dict[p] = fid_pars[p] + par_dict[p]*par_scales[p]
        return new_par_dict

    like_normed = jax.jit(lambda p: -1.0*like(transform_from_zero_mean_unit_var(p)))
    like_normed_deriv = jax.jit(jax.jacfwd(like_normed))

Note the minus sign in the normalised likelihood function; Optax uses a different sign convention (expeting the logl to be positive).
We are now ready to initialise and run the ADAM minimiser.

.. code-block:: python

    import optax
    from copy import deepcopy

    # Initialise the ADAM minimiser and starting point
    adam_optimiser = optax.adam(learning_rate = 0.75)
    starting_pars = transform_to_zero_mean_unit_var(fid_pars)
    opt_state = adam_optimiser.init(starting_pars)

    # Minimise!
    this_pars = deepcopy(starting_pars)
    for i_adam in range(100):

        # Get like value and gradient
        this_logl = like_normed(this_pars)
        this_grad = like_normed_deriv(this_pars)

        # Pass information to optax and update
        updates, opt_state = adam_optimiser.update(this_grad,
                                                opt_state,
                                                this_pars)
        this_pars = optax.apply_updates(this_pars,
                                        updates)

    bf_point = transform_from_zero_mean_unit_var(this_pars)

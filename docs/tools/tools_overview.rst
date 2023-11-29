Auxiliary Tools
=================================================

.. toctree::
   :hidden:
   :titlesonly:
   :maxdepth: 1

   Documentation <tools_docs>

We provide a set of auxiliary tools in the ``tools`` module, designed to simplify common analysis tasks.
These include among others:

* Fisher matrix calculation (:func:`candl.tools.get_fisher_matrix`).
* The generation of mock data (:func:`candl.tools.generate_mock_data`).
* A simple Newton-Raphson minimiser (:func:`candl.tools.newton_raphson_minimiser`, :func:`candl.tools.newton_raphson_minimiser_bdp`).
* Ways to bundle theory codes and the likelihood together to move from parameters to the log likelihood value in one step (:func:`candl.tools.get_params_to_logl_func`, :func:`candl.tools.get_params_to_chi_square_func`).
* Inter-frequency consistency tests (:func:`candl.tools.make_MV_combination`, :func:`candl.tools.make_frequency_conditional`, :func:`candl.tools.make_difference_spectra`).
* Short-cuts to obtain model spectra (:func:`candl.tools.pars_to_model_specs`, :func:`candl.tools.get_foreground_contributions`).

See the full documentation for details.

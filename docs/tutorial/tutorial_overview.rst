Tutorials and Use
=================================================

.. toctree::
   :hidden:
   :titlesonly:
   :maxdepth: 1

   tutorial_list
   usage_tips

Below, you can find an example of how to quickly get started with candl.
Be sure to check out the :ref:`tutorials<Tutorials>` for helpful examples and look at the :ref:`usage tips<Usage Tips>` for some important information on how to use candl.

Quickstart
------------------------------

To initialise the likelihood, point it to the data set ``yaml`` file.
Short cuts exist for data sets released through the :code:`candl_data` `repository <https://github.com/Lbalkenhol/candl_data>`__.
Simply clone the repo to get the data and run :code:`pip install .` to install the shortcuts.
More details can be found in the :code:`candl_data` `readme <https://github.com/Lbalkenhol/candl_data>`__.

Let's say we want to work with the SPT-3G 2018 TT/TE/EE data set.

.. code-block:: python

    import candl
    import candl_data
    
    candl_like = candl.Like(candl_data.SPT3G_2018_TTTEEE)

and that's it! You can now access aspects of the data, for example the band powers ``candl_like.data_bandpowers`` and the covariance matrix ``candl_like.covariance``.

.. tip::
    Want to use only a part of the data? You can pass e.g. ``data_selection = "TT only"`` when initialising the likelihood. See :ref:`Data Selection` for more information.

If you have a dictionary of parameter values and CMB spectra you can then go ahead and calculate the :math:`\chi^2`:

.. code-block:: python

    params = {"Dl": {"TT": .., "EE": .., "TE": ..}, ...}# followed by nuisance parameters
    chi2 = candl_like.chi_square(params)

.. note::

    The likelihood operates in :math:`D_\ell` space, i.e. on :math:`\ell (\ell + 1) C_\ell / (2 \pi)`, in units of :math:`\mu K_{\mathrm{CMB}}^2`.
    Theory spectra start at :math:`\ell=2`.

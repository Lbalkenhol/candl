Usage Tips
=================================================

Below is a collection of information and tips on how to use candl, ranging from good-to-know assumptions to helpful backdoors.


Conventions
----------------------------------------------------------

Primary CMB likelihoods operate in :math:`D_\ell` space, i.e. on :math:`C_\ell \ell (\ell + 1) / (2 \pi)`, in units of :math:`\mu K_{\mathrm{CMB}}^2`.
Theory spectra passed to the likelihood start at :math:`\ell=2`.
The interface module prefers spectrum identifiers (e.g. ``TT``, ``TE``, ``EE``) to be in uppercase.

For lensing likelihoods, theory spectra also start at :math:`L=2`.
The interface module accounts for spectrum identifiers ``pp`` and ``kk`` (in lowercase), expecting the normalisations :math:`C^{\phi\phi}_L \left[ L (L + 1) \right]^2 / (2 \pi)` (such that the value of the spectrum at :math:`L=100` is approximately :math:`10^{-7}`) and :math:`C^{\phi\phi}_L \left[ L (L + 1) \right]^2 / 4`, respectively.


Instantiating Likelihoods
----------------------------------------------------------

There are three options for instantiating likelihoods. In ``candl.Like()`` or ``candl.LensLike()`` you can:

1. point directly to the data set info ``.yaml`` file.
2. use the short cut for released data sets, e.g. ``candl.data.SPT3G_2018_TTTEEE``.
3. point to an index file and specify the variant, e.g. ``candl.data.SPT3G_2018_Lens`` with ``variant = 'lens_only'``.


Working With Instantiated Likelihoods
----------------------------------------------------------

In general, modifications to instantiated likelihood objects are only correctly propagated, if they are done immediately after initialisation.
This has to do with how JAX's jit works with class methods (more details can be found here `here <https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods>`_).
While this may change in the future, exercise caution for now.

.. warning::

   Changes to the attributes of candl likelihoods (e.g. the band powers or the data model) must be made immediately after initialisation.
   Once a jitted method has been called (e.g. the likelihood has been evaluated), changes to attributes are no longer tracked.
   Therefore, it is advised to perform any customisation immediately after initialisation (or by modifying the underlying .yaml file directly - see :ref:`here<Data Structure>` for more info on how to do that).


Combining Multiple Likelihoods
----------------------------------------------------------

You can combine multiple likelihoods by defining a function that returns the sum of the individual likelihoods.
However, for this approach the data sets in question need to be independent of one another.
Consult the literature and in particular the relevant release papers to verify that this is true.
If this is not the case, you need to create a new data set, with a long data vector comprising of the individual data sets and account for the correlation between the data sets in the covariance matrix.
See :ref:`Data Structure` for more details on the structure of data sets.
Even if your data sets are independent, be sure to check that you are not applying any priors twice (e.g. on :math:`\tau`).


Overriding Keywords at Initialisation
----------------------------------------------------------

When instantiating a likelihood, you can pass keywords that override the options in the yaml file.
For example, if you actually want to use a different band power file compared to what is listed in the ``.yaml`` file (and you don't feel like changing the ``.yaml`` file), you can pass ``band_power_file = 'a_different_bdp_file.txt'``.
This can also be handy to clear priors if you don't want to apply them; pass ``priors = []`` in this case.


Ignoring Transformations
----------------------------------------------------------

You can pass ``force_ignore_transformations = <transformations_to_ignore>`` when initialising the likelihood to ignore select transformations from the data model.
Transformations are identified by their ``descriptor`` in the data set info file.
Hence, be sure that when using this option you name all the transformations in the ``.yaml`` file with unique descriptors.
The accepted formats for ``<transformations_to_ignore>`` are:

* a single string for a single transformation to be ignored, e.g. ``force_ignore_transformations = 'calibration'``,
* a single string for multiple transformation to be ignored with transformations separated by a comma and a space, e.g. ``force_ignore_transformations = 'calibration, cirrus'``,
* a list of strings for multiple transformations to be ignored, e.g. ``force_ignore_transformations = ['calibration', 'cirrus']``.

Data Sets
=================================================

The following data sets are available for use with candl.
For the default data sets that ship with the release there are shortcuts available to access them easily (see below).
You can also use :func:`candl.data.print_all_shortcuts` to list all available shortcuts.
New data sets will be uploaded as they become available.

Default Data Sets
-------------------------------------------------

The pip installation of candl ships with the data sets below.
These are by default in your python :code:`site-packages/` folder along with the code.
You may choose to move them to a more convenient location (this will break the default short cuts, though you can override them).

SPT-3G 2018 TT/TE/EE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |spt3g18ttteee| image:: https://img.shields.io/badge/arXiv-2212.05642-b31b1b.svg
   :target: https://arxiv.org/abs/2212.05642

:Paper(s):
   | L. Balkenhol, D. Dutcher, A. Spurio Mancini, A. Dussot, K. Benabed, S. Galli et al. (SPT-3G Collaboration)
   | |spt3g18ttteee|

:Type:
   Primary power spectrum measurement (:math:`TT/TE/EE`)

:Website:
   `SPT-3G Website <https://pole.uchicago.edu/public/data/balkenhol22/>`__

:LAMBDA:
   `NASA Archive <https://lambda.gsfc.nasa.gov/product/spt/spt3g_likelihood_v2_get.html>`__

:Short cut(s):
   ``candl.data.SPT3G_2018_TTTEEE``

SPT-3G 2018 PP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |spt3g18pp| image:: https://img.shields.io/badge/arXiv-2308.11608-b31b1b.svg
   :target: https://arxiv.org/abs/2308.11608

:Paper(s):
   | Z. Pan, F. Bianchini, W. L. K. Wu et al. (SPT-3G Collaboration)
   | |spt3g18pp|

:Type:
   Lensing power spectrum measurement (:math:`\phi\phi`)

:Short cut(s):
   ``candl.data.SPT3G_2018_Lens``, ``candl.data.SPT3G_2018_Lens_and_CMB``

*Note*: this data set uses the lensing power spectrum in :math:`\phi\phi`.
Use ``candl.data.SPT3G_2018_Lens`` when only working with lensing data, use ``candl.data.SPT3G_2018_Lens_and_CMB`` when combining lensing and primary CMB data.

ACT DR4 TT/TE/EE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |actdr4ttteee_aiola| image:: https://img.shields.io/badge/arXiv-2007.07288-b31b1b.svg
   :target: https://arxiv.org/abs/2007.07288

.. |actdr4ttteee_choi| image:: https://img.shields.io/badge/arXiv-2007.07289-b31b1b.svg
   :target: https://arxiv.org/abs/2007.07289

:Paper(s):
   | S. Aiola, E. Calabrese, L. Maurin, S. Naess, B. L. Schmitt et al. (ACT Collaboration)
   | |actdr4ttteee_aiola|
   | S. K. Choi, M. Hasselfield, S.-P. P. Ho, B. Koopman, M. Lungu et al. (ACT Collaboration)
   | |actdr4ttteee_choi|

:Type:
   Primary power spectrum measurement (:math:`TT/TE/EE`)

:LAMBDA:
   `NASA archive <https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_get.html>`__

:Short cut(s):
   ``candl.data.ACT_DR4_TTTEEE``
   
*Note*: This is the CMB-only, foreground marginalised version of the likelihood.
The likelihood refers to the deep data as ``dxd`` and the wide data as ``wxw``.

ACT DR6 PP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |actdr4pp_madhavacheril| image:: https://img.shields.io/badge/arXiv-2304.05203-b31b1b.svg
   :target: https://arxiv.org/abs/2304.05203

.. |actdr4pp_qu| image:: https://img.shields.io/badge/arXiv-2304.05202-b31b1b.svg
   :target: https://arxiv.org/abs/2304.05202

:Paper(s):
   | M. S. Madhavacheril, F. J. Qu, B. D. Sherwin, N. MacCrann, Y. Li et al. (ACT Collaboration)
   | |actdr4pp_madhavacheril|
   | F. J. Qu, B. D. Sherwin, M. S. Madhavacheril, D. Han, K. T. Crowley et al. (ACT Collaboration)
   | |actdr4pp_qu|

:Type:
   Lensing power spectrum measurement (:math:`\phi\phi`)

:Website: `Github <https://github.com/ACTCollaboration/act_dr6_lenslike>`__

:Short cut(s):
   ``candl.data.ACT_DR6_Lens``, ``candl.data.ACT_DR6_Lens_and_CMB``

*Note*: this data set uses the lensing power spectrum in :math:`\kappa\kappa`.
For the ACT + Planck lensing combination see also `Carron, Mirmelstein, Lewis 2023 <https://arxiv.org/abs/2206.07773>`_.
Use ``candl.data.ACT_DR6_Lens`` when only working with lensing data, use ``candl.data.ACT_DR6_Lens_and_CMB`` when combining lensing and primary CMB data.

Adding Data Sets
-------------------------------------------------

If you wish to install data sets separately from the code, please download the desired folders individually from the GitHub repo.
You can place these wherever you like.
It simply suffices to point to the ``.yaml`` file of a likelihood to initialise it.
If you wish to build your own data sets, please consult the information :ref:`here<Data Structure>`.

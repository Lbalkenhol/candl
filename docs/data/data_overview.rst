Data Sets
=================================================

The following data sets are available. New data sets will be made available as they are released.
Use the short cuts supplied point to each data sets ``.yaml`` file and can be used to initialise the likelihood.

SPT-3G 2018 TT/TE/EE
-------------------------------------------------

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

:Short cut:
   ``candl.data.SPT3G_2018_TTTEEE``

*Note:* this data set is not differentiable due to the functional form of the tSZ-CIB correlation term.

SPT-3G 2018 PP
-------------------------------------------------

.. |spt3g18pp| image:: https://img.shields.io/badge/arXiv-2308.11608-b31b1b.svg
   :target: https://arxiv.org/abs/2308.11608

:Paper(s):
   | Z. Pan, F. Bianchini, W. L. K. Wu et al. (SPT-3G Collaboration)
   | |spt3g18pp|

:Type:
   Lensing power spectrum measurement (:math:`\phi\phi`)

:Short cut:
   ``candl.data.SPT3G_2018_Lens``

*Note*: this data set uses the lensing power spectrum in :math:`\phi\phi`.

ACT DR4 TT/TE/EE
-------------------------------------------------

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

:Short cut:
   ``candl.data.ACT_DR4_TTTEEE``
   
*Note*: This is the CMB-only, foreground marginalised version of the likelihood.
The likelihood refers to the deep data as ``dxd`` and the wide data as ``wxw``.

ACT DR6 PP
-------------------------------------------------

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

:Short cut:
   ``candl.data.ACT_DR6_Lens``

*Note*: this data set uses the lensing power spectrum in :math:`\kappa\kappa`.
For the ACT + Planck lensing combination see also `Carron, Mirmelstein, Lewis 2023 <https://arxiv.org/abs/2206.07773>`_.

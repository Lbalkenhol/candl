Data Sets
=================================================

The following data sets are available for use with candl.
For the default data sets that ship with the release there are shortcuts available to access them easily (see below).
For data sets that have multiple variants (for the moment lensing data sets), you can either point to the specific data set you want or you can point to the (more generally named) index file and add ``variant = <your_desired_variant>`` during initialisation.
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
   ````
   | ``candl.data.SPT3G_2018_TTTEEE`` (index file),
   | ``candl.data.SPT3G_2018_TTTEEE_multifreq`` (or ``variant = 'multifreq'``),
   | ``candl.data.SPT3G_2018_TTTEEE_lite`` (or ``variant = 'lite'``)

:Latest version:
   ``v0``

.. tip::

    Though the multi-frequency likelihood is loaded by default, be sure to check out the much faster lite version with ``variant = 'lite'`` introduced in `Balkenhol 2024 <https://arxiv.org/abs/2412.00826>`__.
    If you are interested in how the lite likelihood is constructed, see the associated notebook in the :ref:`tutorials<Tutorials>`.    

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
   | ``candl.data.SPT3G_2018_Lens`` (index file),
   | ``candl.data.SPT3G_2018_Lens_only`` (or ``variant = 'lens_only'``),
   | ``candl.data.SPT3G_2018_Lens_and_CMB`` (or ``variant = 'use_CMB'``)

:Latest version:
   ``v0``

*Note*: this data set uses the lensing power spectrum in :math:`\phi\phi`.
Use ``candl.data.SPT3G_2018_Lens`` with ``variant = 'lens_only'`` or ``candl.data.SPT3G_2018_Lens_only`` when only working with lensing data, use ``candl.data.SPT3G_2018_Lens`` with ``variant = 'use_CMB'`` or ``candl.data.SPT3G_2018_Lens_and_CMB`` when combining lensing and primary CMB data.

ACT DR6 TT/TE/EE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |actdr6ttteee_naess| image:: https://img.shields.io/badge/arXiv-2503.14451-b31b1b.svg
   :target: https://arxiv.org/abs/2503.14451

.. |actdr6ttteee_louis| image:: https://img.shields.io/badge/arXiv-2503.14452-b31b1b.svg
   :target: https://arxiv.org/abs/2503.14452

.. |actdr6ttteee_calabrese| image:: https://img.shields.io/badge/arXiv-2503.14454-b31b1b.svg
   :target: https://arxiv.org/abs/2503.14454

:Paper(s):
   | Naess, Guan, Duivenvoorden, Hasselfield, Wang et al. (ACT Collaboration)
   | |actdr6ttteee_naess|
   | Louis, La Posta, Atkins, Jense et al. (ACT Collaboration)
   | |actdr6ttteee_louis|
   | Calabrese, Hill, Jense, La Posta et al. (ACT Collaboration)
   | |actdr6ttteee_calabrese|

:Type:
   Primary power spectrum measurement (:math:`TT/TE/EE`)

:Website:
   `ACT Website <https://act.princeton.edu/act-dr6-data-products>`__
   `ACT DR6 Notebooks <https://github.com/ACTCollaboration/DR6_Notebooks/tree/main>`__

:LAMBDA:
   `NASA archive <https://lambda.gsfc.nasa.gov/product/act/actadv_prod_table.html>`__

:Short cut(s):
   ``candl.data.ACT_DR6_TTTEEE``

:Latest version:
   ``v0``

*Note*: This is the CMB-only, foreground marginalised (lite) version of the likelihood.
The ACT collaboration suggests to combine with the sroll2 likelihood to constrain the optical depth to reionisation; this implementation of the likelihood contains by default their suggested alternative :math:`\tau` prior of :math:`0.0566 \pm 0.0058`.
If you use ACT data via candl, please see the attribution instructions in the ACT DR6 notebooks `README <https://github.com/ACTCollaboration/DR6_Notebooks/tree/main>`__.


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

:Latest version:
   ``v0``

*Note*: This is the CMB-only, foreground marginalised (lite) version of the likelihood.
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
   | ``candl.data.ACT_DR6_Lens`` (index file),
   | ``candl.data.ACT_DR6_Lens_only`` (or ``variant = 'lens_only'``),
   | ``candl.data.ACT_DR6_Lens_and_CMB`` (or ``variant = 'use_CMB'``)

:Latest version:
   ``v0``

*Note*: this data set uses the lensing power spectrum in :math:`\kappa\kappa`.
For the ACT + Planck lensing combination see also `Carron, Mirmelstein, Lewis 2023 <https://arxiv.org/abs/2206.07773>`_.
Use ``candl.data.ACT_DR6_Lens`` with ``variant = 'lens_only'`` or ``candl.data.ACT_DR6_Lens_only`` when only working with lensing data, use ``candl.data.ACT_DR6_Lens`` with ``variant = 'use_CMB'`` or ``candl.data.ACT_DR6_Lens_and_CMB`` when combining lensing and primary CMB data.


Planck
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

candl comes with ``clipy``, a pure Python implementation of the Planck likelihoods.
See the `clipy website <https://github.com/benabed/clipy>`_ to see which specific likelihoods supported and be sure to download the respective data files from the `Planck Legacy Archive <https://pla.esac.esa.int/pla>`_.

The Planck likelihoods are not implemented as native candl likelihoods, but as wrappers.
While they are differentiable and work with a lot of the candl tools and interface code, they do not support the full functionality.
See the :ref:`clipy x candl tutorial<Tutorials>` for a demonstration of what's possible.


Adding Data Sets
-------------------------------------------------

If you wish to install data sets separately from the code, please download the desired folders individually from the GitHub repo.
You can place these wherever you like.
It simply suffices to point to the ``.yaml`` file of a likelihood to initialise it.
If you wish to build your own data sets, please consult the information :ref:`here<Data Structure>`.

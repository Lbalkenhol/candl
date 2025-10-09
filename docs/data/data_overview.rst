Data Sets
=================================================

Data sets for candl are kept separately from the code.
There currently exist three library repositories with compatible data and you can find detailed information on them below.
You can still use candl without any data library for the different tools supplied or to create your own likelihoods.

.. warning::

    For versions :code:`v1.*` of candl data came directly with the pip installation.
    This is no longer the case (due to the growing size of amazing CMB data we have!) and the data need to be installed separately of the code.

``spt_candl_data``
-------------------------------------------------

`Official repository <https://github.com/SouthPoleTelescope/spt_candl_data>`__ of the South Pole Telescope collaboration featuring the latest SPT-3G data.
Simply clone the repo to get the data and run :code:`pip install .` inside the folder to install.
You can use :code:`spt_candl_data.print_all_shortcuts` to list all available shortcuts to data sets.
More details can be found in the :code:`spt_candl_data` `readme <https://github.com/SouthPoleTelescope/spt_candl_data>`__.

SPT-3G D1 T&E
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |spt3gd1tne| image:: https://img.shields.io/badge/arXiv-2506.20707-b31b1b.svg
   :target: https://arxiv.org/abs/2506.20707


:Paper(s):
   | E. Camphuis, W. Quan, L. Balkenhol, A. R. Khalife, F. Ge, F. Guidi, N. Huang, G. P. Lynch, Y. Omori, C. Trendafilova et al. 2025 (SPT-3G Collaboration)
   | |spt3gd1tne|
   | W. Quan et al. (SPT-3G Collaboration), in prep.

:Type:
   Primary power spectrum measurement (:math:`TT/TE/EE`)

:Website:
   `SPT Website <https://pole.uchicago.edu/public/data/camphuis25/>`__

:LAMBDA:
   To come.

:Short cut(s):
   ````
   | ``spt_candl_data.SPT3G_D1_TnE`` (index file),
   | ``spt_candl_data.SPT3G_D1_TnE_multifreq`` (or ``variant = 'multifreq'``),
   | ``spt_candl_data.SPT3G_D1_TnE_lite`` (or ``variant = 'lite'``)

:Latest version:
   ``v0``

.. tip::

    For this likelihood, please use dedicated variants if you are only fitting a subset of the data (e.g. only EE spectra or only 90GHz) and only use the candl functionality :code:`data_selection` for multipole cuts. See all available variants via :code:`spt_candl_data.print_all_shortcuts()`.

SPT-3G D1 BB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |spt3gd1bb| image:: https://img.shields.io/badge/arXiv-2505.02827-b31b1b.svg
   :target: https://arxiv.org/abs/2505.02827

:Paper(s):
   | J. A. Zebrowski, C. L. Reichardt et al. 2025 (SPT-3G Collaboration)
   | |spt3gd1bb|

:Type:
   Primary power spectrum measurement (:math:`BB`)

:Website:
   `SPT Website <https://pole.uchicago.edu/public/data/zebrowski25/>`__

:LAMBDA:
   To come.

:Short cut(s):
   ````
   | ``spt_candl_data.SPT3G_D1_BB``

:Latest version:
   ``v0``

.. tip::

    This likelihood uses the `Hamimeche-Lewis <https://arxiv.org/abs/0801.0554>`__ approximation. As such, not all subsets of the data set can be run. You may also consider the foreground-marginalised (CMB-only) version of this data set in :code:`candl_data`.


``candl_data``
-------------------------------------------------

The data below are made available through the :code:`candl_data` `repository <https://github.com/Lbalkenhol/candl_data>`__.
Simply clone the repo to get the data and run :code:`pip install .` inside the folder to install.
You can use :func:`candl_data.print_all_shortcuts` to list all available shortcuts to data sets.
More details can be found in the :code:`candl_data` `readme <https://github.com/Lbalkenhol/candl_data>`__.

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
   `SPT Website <https://pole.uchicago.edu/public/data/balkenhol22/>`__

:LAMBDA:
   `NASA Archive <https://lambda.gsfc.nasa.gov/product/spt/spt3g_likelihood_v2_get.html>`__

:Short cut(s):
   ````
   | ``candl_data.SPT3G_2018_TTTEEE`` (index file),
   | ``candl_data.SPT3G_2018_TTTEEE_multifreq`` (or ``variant = 'multifreq'``),
   | ``candl_data.SPT3G_2018_TTTEEE_lite`` (or ``variant = 'lite'``)

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
   | ``candl_data.SPT3G_2018_Lens`` (index file),
   | ``candl_data.SPT3G_2018_Lens_only`` (or ``variant = 'lens_only'``),
   | ``candl_data.SPT3G_2018_Lens_and_CMB`` (or ``variant = 'use_CMB'``)

:Latest version:
   ``v0``

*Note*: this data set uses the lensing power spectrum in :math:`\phi\phi`.
Use ``candl_data.SPT3G_2018_Lens`` with ``variant = 'lens_only'`` or ``candl_data.SPT3G_2018_Lens_only`` when only working with lensing data, use ``candl_data.SPT3G_2018_Lens`` with ``variant = 'use_CMB'`` or ``candl_data.SPT3G_2018_Lens_and_CMB`` when combining lensing and primary CMB data.

SPT-3G D1 BB lite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |spt3gd1bblite| image:: ARXIVTODO
   :target: ARXIVTODO

:Paper(s):
   | J. A. Zebrowski, C. L. Reichardt et al. 2025 (SPT-3G Collaboration)
   | |spt3gd1bb|
   | L. Balkenhol, A. Coerver, C. L. Reichardt, and J. A. Zebrowski 2025
   | [ARXIVTODO]

:Type:
   Primary power spectrum measurement (:math:`BB`)

:Website:
   `SPT Website <https://pole.uchicago.edu/public/data/zebrowski25/>`__

:Short cut(s):
   ````
   | ``candl_data.SPT3G_D1_BB_lite``

:Latest version:
   ``v0``

.. tip::

*Note*: Foreground-marginalised (CMB-only) version of the ``spt_candl_data.SPT3G_D1_BB`` likelihood. This data set is not suited for constraining primordial signals that do not match the CMB black body SED.

SPTpol BB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |sptpolbb| image:: https://img.shields.io/badge/arXiv-1910.05748-b31b1b.svg
   :target: https://arxiv.org/abs/1910.05748

:Paper(s):
   | J. T. Sayre, C. L. Reichardt, J. W. Henninget al. 2020 (SPTpol Collaboration)
   | |sptpolbb|
   | L. Balkenhol, A. Coerver, C. L. Reichardt, and J. A. Zebrowski 2025
   | [ARXIVTODO]

:Type:
   Primary power spectrum measurement (:math:`BB`)

:Website:
   `SPT Website <https://pole.uchicago.edu/public/data/sayre19/>`__

:LAMBDA:
   `NASA archive <https://lambda.gsfc.nasa.gov/product/spt/sptpol_bblh_2019_info.html>`__

:Short cut(s):
   ````
   | ``candl_data.SPTpol_BB`` (index file),
   | ``candl_data.SPTpol_BB_multifreq`` (or ``variant = 'multifreq'``),
   | ``candl_data.SPTpol_BB_lite`` (or ``variant = 'lite'``)

:Latest version:
   ``v0``

.. tip::

*Note*: This is an implementation of the original Fortran likelihood. Please see Balkenhol et al. 2025 [ARXIVTODO]. for details, especially when analysing :math:`\chi^2` values in detail. Note that the CMB-only version of data set is not suited for constraining primordial signals that do not match the CMB black body SED.


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
   ``candl_data.ACT_DR6_TTTEEE``

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
   ``candl_data.ACT_DR4_TTTEEE``

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
   | ``candl_data.ACT_DR6_Lens`` (index file),
   | ``candl_data.ACT_DR6_Lens_only`` (or ``variant = 'lens_only'``),
   | ``candl_data.ACT_DR6_Lens_and_CMB`` (or ``variant = 'use_CMB'``)

:Latest version:
   ``v1``

*Note*: this data set uses the lensing power spectrum in :math:`\kappa\kappa`.
For the ACT + Planck lensing combination see also `Carron, Mirmelstein, Lewis 2023 <https://arxiv.org/abs/2206.07773>`_.
Use ``candl_data.ACT_DR6_Lens`` with ``variant = 'lens_only'`` or ``candl_data.ACT_DR6_Lens_only`` when only working with lensing data, use ``candl_data.ACT_DR6_Lens`` with ``variant = 'use_CMB'`` or ``candl_data.ACT_DR6_Lens_and_CMB`` when combining lensing and primary CMB data.


Planck
-------------------------------------------------

candl comes with ``clipy``, a pure Python implementation of the 2018 Planck likelihoods.
See the `clipy website <https://github.com/benabed/clipy>`_ to see which specific likelihoods supported and be sure to download the respective data files from the `Planck Legacy Archive <https://pla.esac.esa.int/pla>`_.

The Planck likelihoods are not implemented as native candl likelihoods, but as wrappers.
While they are differentiable and work with a lot of the candl tools and interface code, they do not support the full functionality.
See the :ref:`clipy x candl tutorial<Tutorials>` for a demonstration of what's possible.


Adding Your Own Data Sets
-------------------------------------------------

If you wish to install data sets separately from the code, please download the desired folders individually from the GitHub repo.
You can place these wherever you like.
It simply suffices to point to the ``.yaml`` file of a likelihood to initialise it.
If you wish to build your own data sets, please consult the information :ref:`here<Data Structure>`.

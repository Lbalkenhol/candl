Transformations
=================================================

Transformation make up the data model.
They modify the theory spectra to account for foregrounds, calibration, or any other effect.
Programmatically, transformations are their own python classes that get initialised along with the likelihood.
Their code is in the ``transformations/`` folder, which contains abstract base classes, as well as concrete implementations.
See the :ref:`full documentation<transformations_api>` for details.


Adding a Transformation to the Data Model
-------------------------------------------------

In order for a transformation to be applied to the model spectra, it needs to be listed in the data set info ``.yaml`` file.
At a minimum, the library and name of the transformation must be declared, but beyond this, the specific information that needs to be passed varies.
You can find the relevant info in the doc string of each transformation class.
Here is an example for adding Poisson power using the class ``PoissonPower`` from the ``common`` library:

.. code-block:: yaml

      data_model:
        - Module: "common.PoissonPower"
          spec_param_dict:
            TT 150x150: "TT_Poisson_150x150"
            EE 150x150: "EE_Poisson_150x150"
          ell_ref: 3000

In this example we add Poisson power to the ``TT 150x150`` and ``EE 150x150`` spectra, with free amplitude parameters ``TT_Poisson_150x150`` and ``EE_Poisson_150x150``, respectively.
The amplitude parameters are normalised at ``ell_ref = 3000``.


Transformation Blocks
-------------------------------------------------

To keep things tidy, you may chose to separate out a series of transformations into a block and store them in a separate file.
For example, you could have ``TT_foregrounds.yaml`` file containing the list transformations to be applied to temperature spectra.
In your main ``.yaml`` file you would then include this block as follows:

.. code-block:: yaml

      data_model:
        - Block: "TT_foregrounds.yaml"


Available Transformations
----------------------------------------

Before writing your own transformation, check the libraries in ``transformations/`` that it is not already available, the classes already implemented cover everything that is needed for the data sets listed on the `data sets page <../data/data_overview.html>`_.
Moreover, note that limited support for the SO library `FGSpectra <https://github.com/simonsobs/fgspectra/tree/main>`_ exists.
In particular, the class ``FactorizedCrossSpectrum``` is made available (see the documentation of ``transformations/common.FGSpectraInterfaceFactorizedCrossSpectrum`` for details).

.. tip::

   You can also load transformations that are not part of candl's library by pointing directly to the specific class, e.g.: ``my_transformations_module.MyTransformation``.
   candl will then import the class under the hood, given that ``my_transformations_module`` is an importable module.


Writing Your Own Transformation
----------------------------------------

Before writing your own transformation, check the libraries in ``transformations/`` that it is not already available, the classes already implemented cover everything that is needed for the data sets listed on the `data sets page <../data/data_overview.html>`_.
If you are sure you want to create your own transformation class, you can use the abstract base classes in ``transformations/transform.py`` as a starting point.
Crucially, all transformation classes must have an ``transform(self, model_Dl, params)`` method that applies it to the model spectra, ``model_Dl``, for the parameter values ``params`` and returns the modified model vector.

Transformation classes are programatically instantiated by the likelihood.
In this process any information from the corresponding block in the ``.yaml`` info file is passed.
If the initialisation of the transformation requires any keywords that match attributes of the likelihood, these are also passed.
Additionally, the following keywords in the transformations intitialisation are recognised by the likelihood:

* ``freq_info`` (primary CMB only): requires the use of ``effective_frequencies: <source>`` in the ``.yaml`` info file. A list of lists is then passed to the transformation, containing the two effective frequencies for each spectrum.
* ``bandpass_info`` (primary CMB only): if requested and band passes are supplied, a list of lists is passed to the transformation, containing the two band passes for each spectrum. See info on the band pass class for details.
* ``template_arr``: if a `template_file: <relative file path>` is listed in the likelihood block containing the angular multipole moments and corresponding template values, this file is read in and passed as ``template_arr`` to the transformation for initialisation.
* ``link_transformation_module`` (primary CMB only): allows to link transformations together. If blocks starting with ``link_transformation_module`` are listed in the ``.yaml`` info file, the likelihood looks for already initialised transformations matching the desired class. See SPT-3G 2018 tSZ-CIB module as an example.
* ``M_matrices`` (lensing likelihood only): reads in M matrices specified by ``Mmodes: <list of types of spectra>`` from passed ``M_matrices_folder: <relative path to M matrix folder>``.
* ``fiducial_correction`` (lensing likelihood only): reads in and passes fiducial correction from ``fiducial_correction_file: <relative path to fiducial correction file>``.

It is then up to you to deal with this information in your transformation class and initialise it.
Note that for primary CMB likelihoods transformations act on the unbinned model spectra, while for lensing likelihoods they act on the binned model spectra.

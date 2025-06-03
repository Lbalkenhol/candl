"""
IO module that handles reading in data and providing feedback to the user.

Overview:
--------------

* :func:`read_meta_info_from_yaml`
* :func:`load_info_yaml`
* :func:`read_spectrum_info_from_yaml`
* :func:`read_file_from_yaml`
* :func:`read_file_from_path`
* :func:`read_effective_frequencies_from_yaml`
* :func:`read_window_functions_from_yaml`
* :func:`read_transformation_info_from_yaml`
* :func:`read_lensing_M_matrices_from_yaml`
* :func:`like_init_output`
"""

# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
from astropy.io import fits

# --------------------------------------#
# PROCESS YAML INPUT
# --------------------------------------#


def read_meta_info_from_yaml(data_set_dict):
    """
    Read any meta information.

    Parameters
    --------------
    data_set_dict : dict
        The data set dictionary containing all the information from the input yaml file.

    Returns
    --------------
    str :
        Name of the likelihood.
    """

    return data_set_dict["name"]


def load_info_yaml(yaml_file, variant="default"):
    """
    Read the data set yaml file that contains all the information needed to instantiate the likelihood.
    Navigates through index files if necessary.

    Parameters
    --------------
    yaml_file : str
        Path of the data set yaml file or index file
    variant : str
        Optional, to be used for index files. Indicates the variant of the data set to be read in.

    Returns
    --------------
    dict :
        Data set dictionary.
    str :
        Path of the data set yaml file.
    """

    # Check if this is an index file or a complete data set file already
    dataset_file = yaml_file
    if "index.yaml" in yaml_file:

        # Read index file
        with open(yaml_file, "r") as f:
            index_dict = yaml.load(f, Loader=yaml.loader.SafeLoader)

        # Grab file corresponding to desired variant
        if variant is None:
            variant = "default"
        elif variant not in index_dict:
            print(
                f"Variant '{variant}' not found in index file '{yaml_file}'. Running with default '{index_dict['default']}'."
            )
            variant = "default"

        dataset_file = "/".join(yaml_file.split("/")[:-1]) + "/" + index_dict[variant]

    # Read in the dataset yaml file
    with open(dataset_file, "r") as f:
        data_set_dict = yaml.load(f, Loader=yaml.loader.SafeLoader)

    return data_set_dict, dataset_file


def expand_transformation_block(data_set_dict, block_file):
    """
    Read a .yaml file containing a list of transformation entries.

    Parameters
    --------------
    data_set_dict : dict
        The data set dictionary containing all the information from the input yaml file.
    block_file : str
        Full path of the .yaml block file containing the transformations to be read.

    Returns
    --------------
    list :
        List of transformations form the block file.
    """

    with open(data_set_dict["data_set_path"] + block_file, "r") as f:
        transformation_block = yaml.load(f, Loader=yaml.loader.SafeLoader)

    # Check if any paths are given in here - these should be adjusted so that they are relative to the info .yaml file again
    block_file_path_prefix = "/".join(block_file.split("/")[:-1]) + "/"
    for i, tr in enumerate(transformation_block):
        for tr_attribute in list(tr.keys()):
            if "file" in tr_attribute and "." in tr[tr_attribute]:
                # This is most likely a file path that needs to be adjusted
                transformation_block[i][tr_attribute] = (
                    block_file_path_prefix + tr[tr_attribute]
                )

    return transformation_block


def read_spectrum_info_from_yaml(data_set_dict, lensing=False):
    """
    Read spectrum info.
    If lensing == True only returns first, second, and fourth item from usual returns.

    Parameters
    --------------
    data_set_dict : dict
        The data set dictionary containing all the information from the input yaml file.

    Returns
    --------------
    list :
        List of strings of spectrum identifiers.
    list :
        List of strings of spectrum types.
    list :
        List of lists with two entries giving frequencies for each spectrum.
    int :
        Number of total spectra.
    list :
        List of ints giving number of bins for each spectrum.
    """

    # Extract spectrum order, number of bins, spectrum types, and spectrum frequencies
    spec_order = [list(spec.keys())[0] for spec in data_set_dict["spectra_info"]]
    N_bins = [int(list(spec.values())[0]) for spec in data_set_dict["spectra_info"]]

    # Lensing likelihoods only need some of this info
    if lensing == True:
        return spec_order, spec_order, N_bins

    spec_types = [s[:2] for s in spec_order]
    spec_freqs = [s.split(" ")[1].split("x") for s in spec_order]
    N_spectra_total = len(spec_order)
    N_bins = [int(list(spec.values())[0]) for spec in data_set_dict["spectra_info"]]

    return spec_order, spec_types, spec_freqs, N_spectra_total, N_bins


def read_file_from_yaml(data_set_dict, file_kw):
    """
    Read in a file from the data_set dict.

    Parameters
    --------------
    data_set_dict : dict
        The data set dictionary containing all the information from the input yaml file.
    file_kw : str
        A string that corresponds to a keyword in data_set_dict that specifies the file location relative to
        the base path.

    Returns
    --------------
    array :
        File read into array format.
    """

    # Hand off to file reader
    arr = read_file_from_path(data_set_dict["data_set_path"] + data_set_dict[file_kw])

    return arr


def read_file_from_path(full_path):
    """
    Read in a file (array) from a path.
    Method used to read in band powers and covariance matrix.
    Can read four types:
    (1) Text files. Ending must be ".txt" or ".dat".
    (2) Binary files. These must end in ".bin" and be stored as float64s. Will be turned into a square array if possible.
    (3) Numpy files. These must end in ".npy" and will be read in as a numpy array.
    (4) FITS files. These must end in ".fits".

    Parameters
    --------------
    full_path : str
        The absolute path of the file.

    Returns
    --------------
    array :
        File read into array format.
    """

    arr = None
    file_ending = full_path.split(".")[-1]
    if file_ending == "txt" or file_ending == "dat":
        # plain text file
        arr = jnp.array(np.loadtxt(full_path))
    elif file_ending == "bin":
        # binary file
        arr = jnp.array(np.fromfile(full_path, "float64"))
        square_dim = np.sqrt(float(len(arr)))
        if square_dim.is_integer():
            arr = arr.reshape((int(square_dim), int(square_dim)))
    elif file_ending == "npy":
        # numpy file
        arr = jnp.array(np.load(full_path))
    elif file_ending == "fits":
        with fits.open(full_path) as hdul:
            arr = jnp.array(hdul[1].data)

    return arr


def read_effective_frequencies_from_yaml(data_set_dict):
    """
    Read in effective frequency information from data set dictionary.

    Parameters
    --------------
    data_set_dict : dict
        The data set dictionary containing all the information from the input yaml file.

    Returns
    --------------
    dict :
        Dictionary containing keys for all source types given in the effective frequency yaml file. Each entry is a dictionary with frequency identifiers as keys and effective frequencies as values.
    """

    # Check if effective_frequencies exists (might not be needed for all likelihoods depending on spectra/fg models)
    if not "effective_frequencies" in data_set_dict:
        return None

    # Load effective frequencies, casting frequency id's to strings and effective frequencies to floats
    with open(
        data_set_dict["data_set_path"] + data_set_dict["effective_frequencies"], "r"
    ) as f:
        effective_freq_dict = yaml.load(f, Loader=yaml.loader.SafeLoader)
    effective_frequencies = {}
    for spec_source in list(effective_freq_dict.keys()):
        effective_frequencies[spec_source] = {}
        for ky in list(effective_freq_dict[spec_source].keys()):
            effective_frequencies[spec_source][str(ky)] = float(
                effective_freq_dict[spec_source][ky]
            )

    return effective_frequencies


def read_window_functions_from_yaml(data_set_dict, spec_order, N_bins):
    """
    Read band power window functions using data set dictionary. There are two allowed formats:

    (1) Window functions are saved by spectrum as "{spec}_window_functions.txt"
        The files are arrays of (ell, N_bins+1) size, where the first column gives the theory ell.
    (2) Window functions are saved by bin as "window_{i}.txt" starting at i=0.
        The files are arrays of (ell, N_specs+1) size, where the first column gives the theory ell.
    Generally, the first format is preferred as it allows for spectra of different length.

    Parameters
    --------------
    data_set_dict : dict
        The data set dictionary containing all the information from the input yaml file.
    spec_order : list
        List specifying the order of spectra.
    N_bins : list (int)
        Number of bins for each spectrum.

    Returns
    --------------
    list :
        Band power window functions as a list of N_spectra with (N_ell_theory, N_bins) arrays. Start at ell=2.
    """

    # Check what format the window functions are saved in depending on what files are in the passed folder
    files_in_dir = os.listdir(
        data_set_dict["data_set_path"] + data_set_dict["window_functions_folder"]
    )
    bin_expected_files = [f"window_{i + 1}.txt" for i in range(np.amax(N_bins))]
    spec_expected_files = [
        s.replace(" ", "_") + "_window_functions.txt" for s in spec_order
    ]

    if set(spec_expected_files).issubset(set(files_in_dir)):
        # Format (1): window functions are saved by spectra as "{spec}_window_functions.txt"
        window_functions = []
        for i, spec in enumerate(spec_order):
            this_window = np.loadtxt(
                data_set_dict["data_set_path"]
                + data_set_dict["window_functions_folder"]
                + f"{spec.replace(' ', '_')}_window_functions.txt"
            )
            start_ix = np.argwhere(this_window[:, 0] == 2)[0, 0]  # start at ell = 2
            window_functions.append(this_window[start_ix:, 1:])  # remove ell column

    elif set(bin_expected_files).issubset(set(files_in_dir)):
        # Format (2): window functions are saved as "window_{i}.txt"
        # Read in first window to understand file sizes
        first_window = np.loadtxt(
            data_set_dict["data_set_path"]
            + data_set_dict["window_functions_folder"]
            + "window_1.txt"
        )
        ell_column = first_window[:, 0]
        ell_2_ix = np.argwhere(ell_column == 2)[0, 0]
        N_ell_theory = len(ell_column) - ell_2_ix

        window_functions = [
            np.zeros((N_ell_theory, N_bins[i])) for i in range(len(spec_order))
        ]
        for i in range(np.amax(N_bins)):
            wdw = np.loadtxt(
                data_set_dict["data_set_path"]
                + data_set_dict["window_functions_folder"]
                + f"window_{i + 1}.txt"
            )
            for j, spec in enumerate(spec_order):
                window_functions[j][:, i] = wdw[
                    ell_2_ix:, j + 1
                ]  # starting at ell = 2 and ignore ell column

    else:
        raise Exception("Did not recognise window function format.")
        return None

    return [jnp.array(w) for w in window_functions]


def read_transformation_info_from_yaml(data_set_dict, i_tr):
    """
    Read in information about a specific transformation from data set yaml file.

    Parameters
    --------------
    data_set_dict : dict
        The data set dictionary containing all the information from the input yaml file.
    i_tr : int
        Index of the transformation to be read.

    Returns
    --------------
    str :
        Name of the transformation class.
    dict :
        Dictionary of information passed by the user, required as instantiation Parameters for the specified class.
    """

    tr_name = f"{data_set_dict['data_model'][i_tr]['Module']}"
    tr_passed_args = {
        key: data_set_dict["data_model"][i_tr][key]
        for key in data_set_dict["data_model"][i_tr]
        if key != "Module"
    }

    return tr_name, tr_passed_args


def read_lensing_M_matrices_from_yaml(full_path, Mtype="pp"):
    """
    Read lensing M matrices.
    This function assumes that window functions are saved in a specific format:  window{n}.dat, where n runs over the bin numbers.
    Each file specifies the response function at all L values.
    All bins are read in, any cropping is performed later.

    Parameters
    --------------
    full_path : str
        The absolute path of the folder.
    Mtype : str
        Which M matrices to load, "TT", "TE", "EE", "BB", "pp", or "kk".

    Returns
    --------------
    array :
        M matrices as a (number_of_ells, N_files_total) array.
    """

    # Load in window functions

    # Figure out how many files (i.e. bins) there are to read
    N_files_total = len(
        [
            name
            for name in os.listdir(full_path)
            if name[:7] == "window_" and name[-4:] == ".txt"
        ]
    )

    # find highest L from the first bin's window
    first_window = np.loadtxt(full_path + "window_0.txt")
    last_L = int(first_window[-1, 0])
    number_of_ells = last_L - 1  # starting at ell = 2
    start_ix = np.argwhere(first_window[:, 0] == 2)[0, 0]  # start at ell = 2
    this_window = np.zeros([number_of_ells, N_files_total])

    # Spectrum order
    ix_to_access = {"TT": 0, "TE": 1, "EE": 2, "BB": 3, "pp": 4, "kk": 4}

    # each bin has its own file
    for n in range(N_files_total):
        a_window = np.loadtxt(full_path + "window_" + str(n) + ".txt")
        this_window[:, n] = a_window[
            start_ix:, ix_to_access[Mtype] + 1
        ]  # offset by 1 due to ell column in files

    return this_window


# --------------------------------------#
# FEEDBACK FOR INITIALISATION
# --------------------------------------#


def like_init_output(like):
    """
    Print details after successful initialisation of the likelihood.

    Parameters
    -----------------
    like:
        candl.Like or candl.LensLike

    Returns
    --------------
        None

    """

    # Skip if no feedback is requested
    if "feedback" not in like.data_set_dict:
        return
    if not like.data_set_dict["feedback"]:
        return

    if "log_file" in like.data_set_dict:
        logging.basicConfig(
            filename=like.data_set_dict["like_folder_path"]
            + like.data_set_dict["log_file"],
            encoding="utf-8",
            level=logging.INFO,
            format="%(message)s",
        )
        write_msg = lambda msg: logging.info(msg)
    else:
        write_msg = lambda msg: print(msg)

    line_width = 80

    # Header
    start_str = (
        f"Successfully initialised candl likelihood '{like.data_set_dict['name']}' (type: {type(like)}).\n"
        f"Data loaded from '{like.data_set_dict['data_set_path']}'.\n"
        f"Functional likelihood form: {like.data_set_dict['likelihood_form']}"
    )
    write_msg(start_str)
    write_msg(line_width * "-")

    # Spectrum Order
    spec_str = f"It will analyse the following spectra:\n"
    write_msg(spec_str)
    for i, spec in enumerate(like.spec_order):
        write_msg(
            "{:<12}".format(spec)
            + f" ({like.N_bins[i]} bins, bin centres spanning ell = {sigfig.round(float(like.effective_ells[like.bins_start_ix[i]]), decimals=1)} - {sigfig.round(float(like.effective_ells[like.bins_stop_ix[i]-1]), decimals=1)})"
        )
    write_msg(line_width * "-")

    # Data Model
    if len(like.data_model) == 0:
        dat_model_str = f"No transformation have been initialised in the data model.\n"
    else:
        dat_model_str = f"A data model consisting of {len(like.data_model)} transformations has been initialised.\n"
        dat_model_str += f"The following transformations will be applied to the theory spectra in this order:\n"

    write_msg(dat_model_str)
    for i, tr in enumerate(like.data_model):
        write_msg(f"({i+1}) Name: {tr.descriptor}")
        write_msg(len(str(i)) * " " + f"   Type: {type(tr)}")
    write_msg(line_width * "-")

    # Priors
    if len(like.priors) == 0:
        prior_str = f"No priors will be added to the likelihood.\n"
    else:
        prior_str = f"A total of {len(like.priors)} Gaussian priors will be added to the likelihood.\n"
        prior_str += f"The priors affect the following parameters:\n"

    write_msg(prior_str)
    for i, prior in enumerate(like.priors):
        write_msg(", ".join(prior.par_names))
    write_msg(line_width * "-")

    return

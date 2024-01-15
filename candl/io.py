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

# --------------------------------------#
# PROCESS YAML INPUT
# --------------------------------------#


def read_meta_info_from_yaml(dataset_dict):
    """
    Read any meta information.

    Parameters
    --------------
    dataset_dict : dict
        The data set dictionary containing all the information from the input yaml file.

    Returns
    --------------
    str :
        Name of the likelihood.
    """

    return dataset_dict["name"]


def load_info_yaml(dataset_file):
    """
    Read the data set yaml file that contains all the information needed to instantiate the likelihood.

    Parameters
    --------------
    dataset_file : str
        Path of the data set yaml file

    Returns
    --------------
    dict :
        Data set dictionary.
    """

    # Read in the dataset yaml file
    with open(dataset_file, "r") as f:
        dataset_dict = yaml.load(f, Loader=yaml.loader.SafeLoader)

    # Any potential path modification can go here

    return dataset_dict


def read_spectrum_info_from_yaml(dataset_dict, lensing=False):
    """
    Read spectrum info.
    If lensing == True only returns first, second, and fourth item from usual returns.

    Parameters
    --------------
    dataset_dict : dict
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
    spec_order = [list(spec.keys())[0] for spec in dataset_dict["spectra_info"]]
    N_bins = [int(list(spec.values())[0]) for spec in dataset_dict["spectra_info"]]

    # Lensing likelihoods only need some of this info
    if lensing == True:
        return spec_order, spec_order, N_bins

    spec_types = [s[:2] for s in spec_order]
    spec_freqs = [s.split(" ")[1].split("x") for s in spec_order]
    N_spectra_total = len(spec_order)
    N_bins = [int(list(spec.values())[0]) for spec in dataset_dict["spectra_info"]]

    return spec_order, spec_types, spec_freqs, N_spectra_total, N_bins


def read_file_from_yaml(dataset_dict, file_kw):
    """
    Read in a file from the data_set dict.

    Parameters
    --------------
    dataset_dict : dict
        The data set dictionary containing all the information from the input yaml file.
    file_kw : str
        A string that corresponds to a keyword in dataset_dict that specifies the file location relative to
        the base path.

    Returns
    --------------
    array :
        File read into array format.
    """

    # Hand off to file reader
    arr = read_file_from_path(dataset_dict["data_set_path"] + dataset_dict[file_kw])

    return arr


def read_file_from_path(full_path):
    """
    Read in a file (array) from a path.
    Method used to read in band powers and covariance matrix.
    Can read two types:
    (1) Text files. Ending must be ".txt" or ".dat".
    (2) Binary files. These must end in ".bin" and be stored as float64s. Will be turned into a square array if possible.

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

    return arr


def read_effective_frequencies_from_yaml(dataset_dict):
    """
    Read in effective frequency information from data set dictionary.

    Parameters
    --------------
    dataset_dict : dict
        The data set dictionary containing all the information from the input yaml file.

    Returns
    --------------
    dict :
        Dictionary containing keys for all source types given in the effective frequency yaml file. Each entry is a dictionary with frequency identifiers as keys and effective frequencies as values.
    """

    # Check if effective_frequencies exists (might not be needed for all likelihoods depending on spectra/fg models)
    if not "effective_frequencies" in dataset_dict:
        return None

    # Load effective frequencies, casting frequency id's to strings and effective frequencies to floats
    with open(
        dataset_dict["data_set_path"] + dataset_dict["effective_frequencies"], "r"
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


def read_window_functions_from_yaml(dataset_dict, spec_order, N_bins):
    """
    Read band power window functions using data set dictionary. There are two allowed formats:

    (1) Window functions are saved by spectrum as "{spec}_window_functions.txt"
        The files are arrays of (ell, N_bins+1) size, where the first column gives the theory ell.
    (2) Window functions are saved by bin as "window_{i}.txt" starting at i=0.
        The files are arrays of (ell, N_specs+1) size, where the first column gives the theory ell.
    Generally, the first format is preferred as it allows for spectra of different length.

    Parameters
    --------------
    dataset_dict : dict
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
        dataset_dict["data_set_path"] + dataset_dict["window_functions_folder"]
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
                dataset_dict["data_set_path"]
                + dataset_dict["window_functions_folder"]
                + f"{spec.replace(' ', '_')}_window_functions.txt"
            )
            start_ix = np.argwhere(this_window[:, 0] == 2)[0, 0]  # start at ell = 2
            window_functions.append(this_window[start_ix:, 1:])  # remove ell column

    elif set(bin_expected_files).issubset(set(files_in_dir)):
        # Format (2): window functions are saved as "window_{i}.txt"
        # Read in first window to understand file sizes
        first_window = np.loadtxt(
            dataset_dict["data_set_path"]
            + dataset_dict["window_functions_folder"]
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
                dataset_dict["data_set_path"]
                + dataset_dict["window_functions_folder"]
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


def read_transformation_info_from_yaml(dataset_dict, i_tr):
    """
    Read in information about a specific transformation from data set yaml file.

    Parameters
    --------------
    dataset_dict : dict
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

    tr_name = f"{dataset_dict['data_model'][i_tr]['Module']}"
    tr_passed_args = {
        key: dataset_dict["data_model"][i_tr][key]
        for key in dataset_dict["data_model"][i_tr]
        if key != "Module"
    }

    return tr_name, tr_passed_args


def read_lensing_M_matrices_from_yaml(full_path, N_bins_total, Mtype="pp"):
    """
    Read lensing M matrices.
    This function assumes that window functions are saved in a specific format:  window{n}.dat, where n runs over the bin numbers.
    Each file specifies the response function at all L values.

    Parameters
    --------------
    full_path : str
        The absolute path of the folder.
    N_bins_total : int
        The number of bins.
    Mtype : str
        Which M matrices to load, "TT", "TE", "EE", "BB", "pp", or "kk".

    Returns
    --------------
    array :
        M matrices as a (number_of_ells, N_bins_total) array.
    """

    # Load in window functions

    # find highest L from the first bin's window
    first_window = np.loadtxt(full_path + "window_0.txt")
    last_L = int(first_window[-1, 0])
    number_of_ells = last_L - 1  # starting at ell = 2
    start_ix = np.argwhere(first_window[:, 0] == 2)[0, 0]  # start at ell = 2
    this_window = np.zeros([number_of_ells, N_bins_total])

    # Spectrum order
    ix_to_access = {"TT": 0, "TE": 1, "EE": 2, "BB": 3, "pp": 4, "kk": 4}

    # each bin has its own file
    for n in range(N_bins_total):
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
    if "feedback" not in like.dataset_dict:
        return
    if not like.dataset_dict["feedback"]:
        return

    if "log_file" in like.dataset_dict:
        logging.basicConfig(
            filename=like.dataset_dict["like_folder_path"]
            + like.dataset_dict["log_file"],
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
        f"Successfully initialised candl likelihood '{like.dataset_dict['name']}' (type: {type(like)}).\n"
        f"Data loaded from '{like.dataset_dict['data_set_path']}'.\n"
        f"Functional likelihood form: {like.dataset_dict['likelihood_form']}"
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
    dat_model_str = f"A data model consisting of {len(like.data_model)} transformations has been initialised.\n"
    if len(like.data_model) > 0:
        dat_model_str += f"The following transformations will be applied to the theory spectra in this order:\n"

    write_msg(dat_model_str)
    for i, tr in enumerate(like.data_model):
        write_msg(f"({i+1}) Name: {tr.descriptor}")
        write_msg(len(str(i)) * " " + f"   Type: {type(tr)}")
    write_msg(line_width * "-")

    return

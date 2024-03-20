"""
Short cuts to ``.yaml`` files for released data sets.

Overview:
------------------------

* :const:`SPT3G_2018_TTTEEE`: SPT3G 2018 TTTEEE data set.
* :const:`SPT3G_2018_Lens`: SPT3G 2018 Lensing data set.
* :const:`SPT3G_2018_Lens_and_CMB`: SPT3G 2018 Lensing data set, with transformations for combined constraints with primary CMB.
* :const:`ACT_DR4_TTTEEE`: ACT DR4 TTTEEE data set.
* :const:`ACT_DR6_Lens`: ACT DR6 Lensing data set.
* :const:`ACT_DR6_Lens_and_CMB`: ACT DR6 Lensing data set, with transformations for combined constraints with primary CMB.
"""

# Grab Data Path
import os

data_path = os.path.dirname(os.path.realpath(__file__))

# Data Sets
SPT3G_2018_TTTEEE = f"{data_path}/SPT3G_2018_TTTEEE_v0/SPT3G_2018_TTTEEE.yaml"
SPT3G_2018_Lens = f"{data_path}/SPT3G_2018_Lens_v0/SPT3G_2018_PP_lens_only.yaml"
SPT3G_2018_Lens_and_CMB = (
    f"{data_path}/SPT3G_2018_Lens_v0/SPT3G_2018_PP_lens_and_CMB.yaml"
)
ACT_DR4_TTTEEE = f"{data_path}/ACT_DR4_CMB_only_v0/ACT_DR4_CMB_only.yaml"
ACT_DR6_Lens = f"{data_path}/ACT_DR6_Lens_v0/ACT_DR6_KK_lens_only.yaml"
ACT_DR6_Lens_and_CMB = f"{data_path}/ACT_DR6_Lens_v0/ACT_DR6_KK_lens_and_CMB.yaml"

shortcuts = {
    "SPT-3G 2018 TT/TE/EE": "SPT3G_2018_TTTEEE",
    "SPT-3G 2018 Lensing": "SPT3G_2018_Lens",
    "SPT-3G 2018 Lensing (for combining with primary CMB)": "SPT3G_2018_Lens_and_CMB",
    "ACT DR4 TT/TE/EE": "ACT_DR4_TTTEEE",
    "ACT DR6 Lensing": "ACT_DR6_Lens",
    "ACT DR6 Lensing (for combining with primary CMB)": "ACT_DR6_Lens_and_CMB",
}


def print_all_shortcuts():
    """
    Prints all available shortcuts to data sets.
    """
    for ky in list(shortcuts.keys()):
        print(
            f"{ky}: 'candl.data.{shortcuts[ky]}'\n(data files located at: {globals()[shortcuts[ky]]})"
        )

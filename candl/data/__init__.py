"""
candl.data directory

Contains short cuts to released data sets.
"""

# Grab Data Path
import os

data_path = os.path.dirname(os.path.realpath(__file__))

# Data Sets
SPT3G_2018_TTTEEE = f"{data_path}/SPT3G_2018_TTTEEE/SPT3G_2018_TTTEEE.yaml"
SPT3G_2018_Lens = f"{data_path}/SPT3G_2018_Lens/SPT3G_2018_PP_lens_only.yaml"
ACT_DR4_TTTEEE = f"{data_path}/ACT_DR4_CMB_only/ACT_DR4_CMB_only.yaml"
ACT_DR6_Lens = f"{data_path}/ACT_DR6_Lens/ACT_DR6_KK_lens_only.yaml"

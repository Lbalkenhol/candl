"""
candl.constants module

A series of useful constants.
"""

# Physical constants
T_CMB = 2.72548  # CMB temperature
PLANCK_h = 6.62606957e-34  # Planck's constant
BOLTZMANN_kB = 1.3806488e-23  # Boltzmann constant
GHz_KELVIN = (
    PLANCK_h / BOLTZMANN_kB * 1e9
)  # Useful shortcut for black-body related functions

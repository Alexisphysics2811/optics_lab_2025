"""
Physical constants and common values for optics experiments.

All values in SI units unless otherwise specified.
"""

import numpy as np

# Speed of light (m/s)
SPEED_OF_LIGHT = 2.99792458e8

# Planck's constant (J·s)
PLANCK_CONSTANT = 6.62607015e-34

# Common laser wavelengths (m)
WAVELENGTH_RED_LASER = 632.8e-9  # HeNe red laser
WAVELENGTH_GREEN_LASER = 532e-9  # Frequency-doubled Nd:YAG
WAVELENGTH_BLUE_LASER = 405e-9   # Violet diode laser

# Common laser wavelengths (nm) for convenience
WAVELENGTH_RED_LASER_NM = 632.8
WAVELENGTH_GREEN_LASER_NM = 532.0
WAVELENGTH_BLUE_LASER_NM = 405.0

# Conversion factors
NM_TO_M = 1e-9
MM_TO_M = 1e-3
UM_TO_M = 1e-6

# Angular conversions
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

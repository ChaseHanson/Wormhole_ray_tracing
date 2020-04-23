# Program to set our initial conditions for building our map
import numpy as np


def calculate_r(a, rho, ell, M):
    """
    Calculate r using Equation (5) from James et al. (2015)

    Parameters
    ----------
    a : float
        Half the height of the wormhole cylinder
    rho : float
        Radius of the wormhole cylinder
    ell : float
        Proper distance in radial direction
    M : float
        Black hole mass

    Returns
    -------
    r : float
        Distance used for calculations as a function of ell
    """
    x = 2 * (np.abs(ell) - a) / (np.pi * M)  # Equation 5b
    # Outside wormhole
    if np.abs(ell) > a:
        r = rho + M * (x * np.arctan(x) - 0.5 * np.log(1 + x**2))  # Equation 5a
    # Inside wormhole
    else:
        r = rho  # Equation 5c
    return r

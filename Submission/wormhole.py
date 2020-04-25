import numpy as np


class Wormhole:
    def __init__(self, a, rho, W):
        self.a = a
        self.rho = rho
        self.W = W
        self.M = self.W / 1.4295268191803117  # Footnote 19


    def calc_r(self, ell):
        """
        Calculate r using Equation (5) from James et al. (2015)

        Parameters
        ----------
        ell : float
            Proper distance in radial direction

        Returns
        -------
        r : float
            Distance used for calculations as a function of ell
        """
        x = 2 * (np.abs(ell) - self.a) / (np.pi * self.M)  # 5b
        # Outside wormhole
        if np.abs(ell) > self.a:
            r = (self.rho + self.M * (x * np.arctan(x) -
                 0.5 * np.log(1 + x**2)))  # 5a
        # Inside wormhole
        else:
            r = self.rho  # 5c
        return r


    def calc_drdl(ell, M):
        """
        Calculate dr/dl with the equation given in footnote 19

        Parameters
        ----------
        ell : float
            Proper distance in radial direction

        Returns
        -------
        dr/dl : float

        """
        return 2 / np.pi * np.arctan((2 * ell) / (np.pi * M))

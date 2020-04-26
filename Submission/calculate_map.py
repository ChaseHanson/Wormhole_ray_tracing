import numpy as np
from scipy.integrate import solve_ivp
from wormhole import Wormhole


def calc_camera_unit_vector(theta_cs, phi_cs):
    """
    Calculate the Cartesian unit vector in the camera's
    local sky with Equation (A9a)

    Parameters
    ----------
    theta_cs : float
        The zenith angle in the camera sky
    phi_cs : float
        The azimuth angle in the camera sky

    Returns
    -------
    N_x, N_y, N_z : float
        The x, y, and z components of the unit vector
    """
    # Equation (A9a)
    N_x = np.sin(theta_cs) * np.cos(phi_cs)
    N_y = np.sin(theta_cs) * np.sin(phi_cs)
    N_z = np.cos(theta_cs)
    return N_x, N_y, N_z


def calc_propagation_vector(N_x, N_y, N_z):
    """
    Calculate the unit vector describing the direction of
    propagation of the ray with Equation (A9b)

    Parameters
    ----------
    N_x : float
        The x-component of the unit vector in the camera sky
    N_y : float
        The y-component of the unit vector in the camera sky
    N_z : float
        The z-component of the unit vector in the camera sky

    Returns
    -------
    n_ell, n_phi, n_theta : float
        The ell, phi, and theta components of the vector
        describing the direction of propagation
    """
    # Equation (A9b)
    n_ell, n_phi, n_theta = -N_x, -N_y, N_z
    return n_ell, n_phi, n_theta


def calc_canonical_momenta(r, theta, n_ell, n_theta, n_phi):
    """
    Calculate the incoming light ray's canonical momenta in the
    ell, theta, and phi directions with Equation (A9c)

    Parameters
    ----------
    N_x : float
        The x-component of the unit vector in the camera sky
    N_y : float
        The y-component of the unit vector in the camera sky
    N_z : float
        The z-component of the unit vector in the camera sky

    Returns
    -------
    p_ell, p_theta, p_phi : float
        The light ray's canonical momenta in the ell, theta,
        and phi directions
    """
    # Equation (A9c)
    p_ell = n_ell
    p_theta = r * n_theta
    p_phi = r * np.sin(theta) * n_phi
    return p_ell, p_theta, p_phi


def calc_b_Bsquared(p_phi, r, n_theta, n_phi):
    """
    Calculate the light ray's constants of motion with Equation (A9d)

    Parameters
    ----------
    p_phi : float
        The ray's momentum in the phi direction
    r : float
        The ray's radial distance from the wormhole's axis of symmetry
    n_theta : float
        The direction of propagation of the incoming ray in the theta
        direction
    n_phi : float
        The direction of propagation of the incoming ray in the phi
        direction

    Returns
    -------
    b, B_squared : float
        The ray's constants of motion
    """
    # Equation (A9d)
    b = p_phi
    B_squared = r**2 * (n_theta**2 + n_phi**2)
    return b, B_squared


def geo_eqns(p_l,p_theta,ell,theta,phi,wormhole):
    N_x = np.sin(theta)*np.cos(phi)
    N_y = np.sin(theta)*np.sin(phi)
    N_z = np.cos(theta)
    n_l     = -N_x
    n_phi   = -N_y
    n_theta =  N_z
    r        = wormhole.calc_r(ell)
    drdl     = wormhole.calc_drdl(ell)
    p_l     = n_l
    p_theta = r * np.sin(theta) * n_theta
    b     = p_phi
    Bsqrd = r**2*(n_theta**2 + n_phi**2)
    geo = np.array([p_l, p_theta/r**2,b/(r**2*(np.sin(theta))**2),Bsqrd*(drdl/r**3),(b**2/r**2)*(np.cos(theta)/(np.sin(theta)**3))])
    return geo


def integrate_geo_equations(t_end=-100):
    # Wormhole parameters
    a = 1  # Half the height of wormhole's cylinder interior
           # in the embedding space
    rho = 200 * a  # Radius of the cylinder
    W = 0.05 * rho  # Black hole lensing width, related to black hole mass
    wormhole = Wormhole(a, rho, W)  # Create a wormhole object

    # Camera and camera sky parameters
    ell_camera = 6.25 * rho + a  # Radial distance of the camera's location
    theta_camera = np.pi/2  # Zenith angle of the camera's location,
                            # in the equatorial plane
    phi_camera = 0  # Azimuthal angle of the camera's location
    theta_cs, phi_cs = np.pi / 8, np.pi / 6  # A location in the camera sky

    # Calculate the unit vector in the camera sky
    N_x, N_y, N_z = calc_camera_unit_vector(theta_cs, phi_cs)

    # Calculate the vector describing the ray's direction of propagation
    n_ell, n_phi, n_theta = calc_propagation_vector(N_x, N_y, N_z)

    # Calculate the light ray's initial distance from the wormhole's
    # axis of symmetry
    r_init = wormhole.calc_r(ell_camera)

    # Calculate the incoming light ray's canonical momenta
    p_ell, p_theta, p_phi = calc_canonical_momenta(r_init, theta_cs,
                                                   n_ell, n_theta, n_phi)

    # The ray's constants of motion
    b, B_squared = calc_b_Bsquared(p_phi, r_init, n_theta, n_phi)

    # Initial conditions for (ell, theta, phi, p_ell, p_theta)
    # The ray starts at the camera's location, we will integrate
    # backwards in time
    ray_arr = np.array([ell_camera, theta_camera, phi_camera,
                        p_ell, p_theta])

    # The right hand side of the ray equations
    def f(t, y):
        return geo_eqns(y, wormhole)

    # Numerically integrate the ray equation with scipy.integrate.solve_ivp
    t_range = (0, t_end)
    map = solve_ivp(f, t_range, ray_arr, method='RK45')


# def integrate_geo_eqns():
#     # y = np.zeros()
#     def f(t, y):
#         return geo_eqns(y[3], y[4], y[0], y[1], y[2], wormhole)
#
#     print(map.y)
#     print(np.shape(map.y))


if __name__ == '__main__':
    integrate_geo_eqns()

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


def geo_eqns(ray_arr, n_theta, n_phi, wormhole):
    # Sort out the quantities in the ray array
    ell, theta, phi = ray_arr[0], ray_arr[1], ray_arr[2]
    p_ell, p_theta = ray_arr[3], ray_arr[4]

    # The ray's distance from the wormhole's axis of symmetry
    r = wormhole.calc_r(ell)
    # And how quickly the ray's distance is changing
    dr_dl     = wormhole.calc_drdl(ell)

    # The ray's constants of motion
    p_phi = r * np.sin(theta) * n_phi  # Equation (A9c)
    b, B_squared = calc_b_Bsquared(p_phi, r, n_theta, n_phi)

    # Set up the right hand side of the system with Equations (A7a) - (A7e)
    dl_dt = p_ell  # A7a
    dtheta_dt = p_theta / r**2  # A7b
    dphi_dt = b / (r * np.sin(theta))**2  # A7c
    dpl_dt = B_squared * dr_dl / r**3  # A7d
    dptheta_dt = (b / r)**2 * np.cos(theta) / np.sin(theta)**3 # A7e

    return np.array([dl_dt, dtheta_dt, dphi_dt, dpl_dt, dptheta_dt])


def integrate_geo_equations(t_end=-100, return_map=False):
    # Wormhole parameters
    a = 0.01  # Half the height of wormhole's cylinder interior
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
    f = lambda t, y: geo_eqns(y, n_theta, n_phi, wormhole)

    # Numerically integrate the ray equations with scipy.integrate.solve_ivp
    t_range = (0, t_end)
    map = solve_ivp(f, t_range, ray_arr, method='RK45')

    if return_map:
        return map


if __name__ == '__main__':
    map = integrate_geo_equations(t_end=-1e5, return_map=True)
    print('Shape of map: {}\n'.format(map.y.shape))
    print('ell: {}\n'.format(map.y[0]))
    print('theta: {}\n'.format(map.y[1]))
    print('phi: {}\n'.format(map.y[2]))
    print('p_ell: {}\n'.format(map.y[3]))
    print('p_theta: {}'.format(map.y[4]))

import numpy as np
import pickle
from scipy.interpolate import interp2d
from scipy import ndimage


def load_map_data(angle_file, map_file):
    """
    Load the saved data from the integration

    Parameters
    ----------
    angle_file : str
        The path to the saved angles
    map_file : str
        The path to the saved map

    Returns
    -------
    angles : dict
        The angles at which the map was evaluated
    ray_map : array-like
        The location of the ray as t -> -infinity
    """
    with open(angle_file, 'rb') as f:
        angles = pickle.load(f)
    with open(map_file, 'rb') as f:
        ray_map = pickle.load(f)
    return angles, ray_map


def create_map_interp_funcs(angles, ray_map):
    """
    Generate the interpolating functions for theta and phi

    Parameters
    ----------
    angles : array-like
        The angles where the output map exist
    ray_map : str
        The output map

    Returns
    -------
    interp_theta, interp_phi : function
    """
    # Load the angles
    theta, phi = angles['theta'], angles['phi']

    # Reshape the arrays to specify the coordinates of
    # each point
    Ntheta, Nphi = len(theta), len(phi)
    theta = np.broadcast_to(theta, (Nphi, Ntheta))
    phi = np.broadcast_to(phi, (Ntheta, Nphi)).T
    import pdb; pdb.set_trace()

    ndimage.map_coordinates(ray_map[:, :, 1])
    # # Interpolate theta and phi individually
    # interp_theta = interp2d(theta, phi, ray_map[:, :, 1])
    # interp_phi = interp2d(theta, phi, ray_map[:, :, 2])
    return interp_theta, interp_phi


if __name__ == '__main__':
    angles, ray_map = load_map_data('data/angles.pickle',
                                    'data/ray_map.pickle')
    interp_theta, interp_phi = create_map_interp_funcs(angles, ray_map)
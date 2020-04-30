import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import RegularGridInterpolator
from skimage import io
from skimage.transform import warp
import datetime
import pdb


def find_nearest_i(array, value):
    ind = (np.abs(array - value)).argmin()
    return ind


def wrap_angle(angle, max_angle=2*np.pi, exclusive=True):
    """
    Wraps angles to the range [0, max_angle) (exclusive by
    default, but can be inclusive of max_angle)

    Parameters
    ----------
    angle : int, float, or array-like
        The angle(s) to wrap
    max_angle : int or float, optional
        The maximum angle possible (default 2*pi)
    exclusive : bool, optional
        Whether to exclude the maximum angle (default True)

    Returns
    -------
    wrapped_angle : int, float, or array_like
        The wrapped angle(s)
    """
    if exclusive:
        wrapped_angle = angle % max_angle
    else:
        max_angle_ind = np.where(angle == max_angle)[0]
        wrapped_angle = angle % max_angle
        wrapped_angle[max_angle_ind] = max_angle
    return wrapped_angle


def pixels_to_angles(xy):
    angles = xy.copy()
    angles[:, 0] = 2 * np.pi * xy[:, 0] / np.max(xy[:, 0])
    angles[:, 1] = np.pi * xy[:, 1] / np.max(xy[:, 1])
    return angles


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


def create_map_interpolator(angles, ray_map):
    """
    Generate the interpolating function for theta and phi

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

    # Interpolate the map
    interp_angles = RegularGridInterpolator((phi, theta),
                                            ray_map[:, :, 1:3])
    return interp_angles


def wormhole_warp(xy, interp_func):
    # first column is phi, second column is theta
    im_angles = pixels_to_angles(xy)
    unique_phi = np.unique(im_angles[:, 0])
    unique_theta = np.unique(im_angles[:, 1])
    xy_warped = xy.copy()
    sky_angles = interp_func(im_angles)

    # NOTE: This is highly parallelizabl
    # x corresponds to phi, y corresponds to theta
    print('Total pixels in image: {}'.format(len(xy)))
    start = datetime.datetime.now()
    for i, pixel in enumerate(xy):
        # pdb.set_trace()
        if (i + 1) % 1e4 == 0:
            now = datetime.datetime.now()
            print('{} pixels completed in {}'.format(i+1, now-start), end='\r')
        sky_angles[i][0] = wrap_angle(sky_angles[i][0], 2*np.pi)
        sky_angles[i][1] = wrap_angle(sky_angles[i][1], 2*np.pi)
        ind_phi = find_nearest_i(unique_phi, sky_angles[i][0])
        ind_theta = find_nearest_i(unique_theta, sky_angles[i][1])
        xy_warped[i] = np.array([ind_phi, ind_theta])
    end = datetime.datetime.now()
    print('Time elapsed: {}'.format(end - start))

    return xy_warped


if __name__ == '__main__':
    angles, ray_map = load_map_data('data/angles_250_500.pickle',
                                    'data/ray_map_250_500.pickle')
    interp_angles = create_map_interpolator(angles, ray_map)

    image_path = 'images/saturn.jpg'
    image = io.imread(image_path)

    map_args = {'interp_func': interp_angles}
    warped_image = warp(image, wormhole_warp, map_args=map_args)

    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ax.imshow(warped_image)
    fig.savefig('images/warped_dneg_saturn_250_500_map.png', bbox_inches='tight')
    fig.show()

    # for i in range(len(test[:, :, 0])):
    #     for j in range(len(test[:, :, 0][0])):
    #         test[i, j, 0] = find_nearest(np.unique(im_phi),
    #                                      test[i, j, 0])
    #         test[i, j, 1] = find_nearest(np.unique(im_theta),
    #                                      test[i, j, 1])

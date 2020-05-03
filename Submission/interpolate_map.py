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

def celestial_data_sorter(ray_map):
    """
    Sorts elements of the ray_map tensor based of the sign of distance ell at
    each point (that is, which celestial sphere the light ray originates from).
    Eliminates (hopefully) numerical artifacts from cross-interpolation of two
    rays from different celestial spheres which have similar phi & theta, but
    opposite sign.

    Parameters
    ----------
    ray_map : str
        The output map

    Returns
    -------
    ray_map : str
        The output map (but sorted)
    """
    Ntheta = 250 # = ray_map[1]
    Nphi   = 500 # = ray_map[2]

    sorter_helper_friend = np.zeros((Nphi))
    temp_position = np.zeros(3)
    temp_array = np.zeros((1, Ntheta, 3))

    for i in range(Nphi): # == Nphi
        for j in range(Ntheta - 1):
            # If the initial theta's sign is - and the next is +, transfer all positional data
            if ray_map[i, j, 0] < ray_map[i, j+1, 0]:
                # prototypical sorting shuffle - transfer of sign, radial vector components
                temp_position[0:3] = ray_map[i, j, :]
                ray_map[i, j, :]   = ray_map[i, j+1, :]
                ray_map[i, j+1, :] = temp_position[0:3]
            if ray_map[i, j, 0] < 0:
                sorter_helper_friend[i] += 1 # counts how many "pluses" there are - i.e. how many points are in the upper sphere

    # I'm smart enough to know this can be optimized...I'm just not smart enough to know how
    for i in range(Nphi - 1):
        if sorter_helper_friend[i] < sorter_helper_friend[i+1]:
            # prototypical sorting shuffle - transfer of sign, radial vector components
            temp_array          = ray_map[i, :, :]
            ray_map[i, :, :]    = ray_map[i+1, :, :]
            ray_map[i+1, :, :]  = temp_array

    return ray_map

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

    #print(ray_map)
    #print("ray_map shape: \t\t", np.shape(ray_map))

    Ntheta = 250 # = ray_map[1]
    Nphi   = 500 # = ray_map[2]

    ########################################
    if False: # below is a very hopeless attempt at organizing celestial data
        lower_theta = 0
        lower_phi   = 0
        lower = 0

        for i in range(Nphi): # == Nphi
            for j in range(Ntheta): # == Ntheta
                if ray_map[i, j, 0] < 0:
            #        print("ray_map ell less than zer0")
                    lower += 1
            #upper_theta += Ntheta - lower
            lower_theta += lower
            lower = 0
            if ray_map[i, j, 0] < 0:
            #    print("ray_map ell less than zer0")
                lower_phi += 1

        upper_theta = Ntheta - lower_theta
        upper_phi = Nphi - lower_phi

        print("lower_theta:\t\t", lower_theta)
        print("upper_theta:\t\t", upper_theta)
        print("Ntheta:\t\t", Ntheta)
        print("lower_phi:\t\t", lower_phi)
        print("upper_phi:\t\t", upper_phi)
        print("Nphi:\t\t", Nphi)


        ray_map_upper = np.zeros((upper_phi, upper_theta, 2))
        ray_map_lower = np.zeros((lower_phi, lower_theta, 2))

        for i in range(ray_map[1]): # == Nphi
            for j in range(ray_map[2]): # == Ntheta
                if ray_map[i, j, 0] < 0:
                    ray_map_lower[i, j, 0:2] = ray_map[1:3]
                else:
                    ray_map_upper[i, j, 0:2] = ray_map[1:3]

        interp_angles_lower = RegularGridInterpolator((phi, theta),
                                                ray_map_lower[:, :, 0:2])

        interp_angles_upper = RegularGridInterpolator((phi, theta),
                                                ray_map_upper[:, :, 0:2])

        lower = 0
        bool = True

        for i in range(Nphi%10): # == Nphi
            for j in range(Ntheta%10): # == Ntheta
                if ray_map[i, j, 0] < 0:
            #        print("ray_map ell less than zer0")
                    print("@@@ BOTTOM @@@")
                else:
                    print("---  TOP  ---")
                    lower += 1

        upper = Nphi * Ntheta - lower
        ray_map_upper = np.zeros((upper, 2))
        ray_map_lower = np.zeros((lower, 2))

        interp_angles_lower = RegularGridInterpolator((phi, theta),
                                                ray_map_lower[:, 0:2])

        interp_angles_upper = RegularGridInterpolator((phi, theta),
                                                ray_map_upper[:, 0:2])


        return interp_angles_lower, interp_angles_upper
        #######################################################

    ray_map = celestial_data_sorter(ray_map)

    interp_angles = RegularGridInterpolator((phi, theta),
                                            ray_map[:, :, 1:3])
    return interp_angles
    # Else, if sign of ell is positive, interpolate in upper celestial sphere.


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

import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage import io
from skimage import transform
from scipy import ndimage
import datetime


def find_nearest(array, value):
    ind = (np.abs(array - value)).argmin()
    return array[ind]


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


# def angles_to_pixels(pixel_array, max_angle=2*np.pi,
#                      endpoint=True):
#     NotImplemented


def pixels_to_angles(pixel_array, max_angle=2*np.pi,
                     endpoint=True):
    Npixels = int(np.max(pixel_array) + 1)
    angle_arr = np.linspace(0, max_angle, Npixels,
                            endpoint=endpoint)
    return angle_arr


def wormhole_lens(xy, ray_map):
    # Convert pixels to angles
    cols, rows = xy[:, 0], xy[:, 1]
    col_angles = pixels_to_angles(cols, max_angle=2*np.pi,
                                  endpoint=False)
    row_angles = pixels_to_angles(rows, max_angle=np.pi)
    Nrows, Ncols = len(row_angles), len(col_angles)

    # Broadcast and flatten the angle arrays to make them
    # the correct shape
    col_angles_reshape = np.broadcast_to(col_angles,
                                         (Nrows, Ncols)).T
    col_angles_reshape = col_angles_reshape.flatten()
    row_angles_reshape = np.broadcast_to(row_angles,
                                         (Ncols, Nrows)).flatten()
    row_angles_reshape = row_angles_reshape.flatten()
    coordinates = np.stack((col_angles_reshape, row_angles_reshape),
                            axis=0)

    theta_warp = ndimage.map_coordinates(ray_map[:, :, 1],
                                         coordinates, mode='wrap')
    phi_warp = ndimage.map_coordinates(ray_map[:, :, 2],
                                       coordinates, mode='wrap')

    print(theta_warp)
    print(phi_warp)

    start = datetime.datetime.now()
    for i, theta in enumerate(theta_warp):
        print(i, end='\r')
        theta_warp[i] = find_nearest(row_angles, theta)
    end = datetime.datetime.now()
    print(end - start)
    print(theta_warp)
    start = datetime.datetime.now()
    for i, phi in enumerate(phi_warp):
        print(i, end='\r')
        phi_warp[i] = find_nearest(col_angles, phi)
    end = datetime.datetime.now()
    print(end - start)
    print(phi_warp)

    return xy


if __name__ == '__main__':
    # Images are already a rectangular grid of RA/dec
    # (related to azimuth and zenith angles)
    # saturn_path = 'images/saturn.jpg'
    # saturn = io.imread(saturn_path)
    star_field_path = 'images/star_field.jpg'
    star_field = io.imread(star_field_path)

    angles, ray_map = load_map_data('data/angles.pickle',
                                    'data/ray_map.pickle')
    map_args = {'ray_map': ray_map}
    translated_star_field = transform.warp(star_field, wormhole_lens,
                                           map_args=map_args)
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ax.imshow(translated_star_field)
    fig.tight_layout()
    plt.show()

    # theta, phi = pix_to_angle(star_field)
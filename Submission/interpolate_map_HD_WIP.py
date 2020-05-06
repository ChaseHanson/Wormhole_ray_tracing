import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.transform import resize
from scipy.interpolate import griddata
from skimage import io
from skimage.transform import warp
import datetime


def find_nearest_i(array, value):
    ind = (np.abs(array - value)).argmin()
    return ind


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


def wormhole_warp(xy, phi_cs, theta_cs, ray_map):
    # First column is phi, second column is theta
    im_angles = pixels_to_angles(xy)
    unique_phi = np.unique(im_angles[:, 0])
    unique_theta = np.unique(im_angles[:, 1])
    xy_warped = xy.copy()
    angle_arr = np.stack((phi_cs, theta_cs), axis=1)
    sky_theta_arr = griddata(angle_arr, ray_map[:, 0], im_angles)
    sky_phi_arr = griddata(angle_arr, ray_map[:, 1], im_angles)

    # NOTE: This is highly parallelizable
    # x corresponds to phi, y corresponds to theta
    print('Total pixels in image: {}'.format(len(xy)))
    start = datetime.datetime.now()
    for i, pixel in enumerate(xy):
        if (i + 1) == len(xy):
            now = datetime.datetime.now()
            print('{} pixels completed in {}'.format(i+1, now-start))
        elif (i + 1) % 1e4 == 0:
            now = datetime.datetime.now()
            print('{} pixels completed in {}'.format(i+1, now-start), end='\r')
        sky_phi, sky_theta = sky_phi_arr[i], sky_theta_arr[i]
        ind_phi = find_nearest_i(unique_phi, sky_phi)
        ind_theta = find_nearest_i(unique_theta, sky_theta)
        xy_warped[i] = np.array([ind_phi, ind_theta])

    return xy_warped


if __name__ == '__main__':
    angles, ray_map = load_map_data('data/angles_150_300.pickle',
                                    'data/ray_map_150_300.pickle')
    image_path_lower = 'images/saturn.jpg'
    image_path_upper = 'images/star_field.jpg'
    image_lower = io.imread(image_path_lower)
    image_upper = io.imread(image_path_upper)
    resize_lower = resize(image_lower,(2160,3840),order=1)
    resize_upper = resize(image_upper,(2160,3840),order=1)
    #fig,ax1 = plt.subplots(1,1)
    #ax1.imshow(resize_lower,aspect='auto',interpolation='spline36')
    #plt.axis('off')
    #fig.set_size_inches(9,5)
    #fig.savefig('images/resized_saturn.png',dpi=650, bbox_inches='tight')
    ##ax2.imshow(resize_upper,aspect='auto',interpolation='spline36')
    #plt.axis('off')
    #fig.set_size_inches(9,5)
    #fig.savefig('images/resized_star_field.png',dpi=650, bbox_inches='tight')
    # Sort both the input angles on the camera sky and the output map
    # by which celestial sphere they're on
    sign = ray_map[:, :, 0]  # The sign of the final location of the ray
    upper_inds, lower_inds = np.where(sign > 0), np.where(sign < 0)
    upper_phi = angles['phi'][upper_inds[0]]
    lower_phi = angles['phi'][lower_inds[0]]
    upper_theta = angles['theta'][upper_inds[1]]
    lower_theta = angles['theta'][lower_inds[1]]
    upper_ray_map = ray_map[upper_inds][:, 1:3]
    lower_ray_map = ray_map[lower_inds][:, 1:3]

    # Warp the images on the two celestial spheres
    map_args_lower = {'phi_cs': lower_phi, 'theta_cs': lower_theta,
                      'ray_map': lower_ray_map}
    map_args_upper = {'phi_cs': upper_phi, 'theta_cs': upper_theta,
                      'ray_map': upper_ray_map}
    warped_lower = warp(resize_lower, wormhole_warp, map_args=map_args_lower)
    warped_upper = warp(resize_upper, wormhole_warp, map_args=map_args_upper)

    # Reverse the x coordinates to match the Double Negative axes
    warped_lower, warped_upper = warped_lower[:, ::-1], warped_upper[:, ::-1]
    warped_lower_resize = resize(warped_lower,(2160,3840),order=4)
    warped_upper_resize = resize(warped_upper,(2160,3840),order=4)
    # Hacky way to combine them
    alpha = 0.5
    warped_image = alpha * warped_lower_resize + (1-alpha) * warped_upper_resize
    warped_image_resized = resize(warped_image,(2160,3840),order=4)
    # Show it!
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ax.imshow(warped_image_resized, aspect = 'auto',interpolation = 'spline36',origin='upper')
    plt.axis('off')
    #ax.set_xlim([820, 920])
    #ax.set_ylim([485, 385])
    fig.set_size_inches(9,5)
    fig.savefig('images/warped_saturn_star_field.png',dpi=650, bbox_inches='tight')
    #fig.savefig('images/warped_final.png',bbox_inces='tight')
    fig.show()

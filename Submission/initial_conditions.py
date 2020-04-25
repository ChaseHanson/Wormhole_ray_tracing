# Program to set our initial conditions for building our map

import numpy as np
from wormhole import Wormhole

# parameters
a            = 1              # half the height of wormhole embedded cylinder 
rho          = 200 * a        # radius of the cylinder  
theta_camera = np.pi/2        # initial theta of the camera, in the equatorial plane
l_camera     = 6.25 * rho + a # initial radius of the camera from the origin
phi_camera   = 0              # initial phi of the camera 
W            = 0.05 * rho     # lensing width

theta_init   = theta_camera

# Our unit vectors look like so: s>0 for the upper celestial sphere, s<0 for the lower.
if False:
    el, ephi, etheta = 1, 1, 1
    if s>0:
        ex, ey, ez = el, ephi, -etheta

    if s<0:
        ex, ey, ez = -el, ephi, etheta

# Initialize a direction on the camera's local sky
theta_cs     = np.pi/8
phi_cs       = np.pi/6

# The unit vector N pointing in this direction has Cartesian components
N_x = np.sin(theta_cs)*np.cos(phi_cs)
N_y = np.sin(theta_cs)*np.sin(phi_cs)
N_z = np.cos(theta_cs)

# The direction n of the propagation of the incoming ray that arrives from -N
n_l     = -N_x
n_phi   = -N_y
n_theta =  N_z

# Initialize our wormhole class, calculate radius r
wormhole = Wormhole(a, rho, W)
r        = wormhole.calc_r(l_camera)

# The incoming light ray's canonical momenta
p_l     = n_l
p_phi   = r * n_phi
p_theta = r * np.sin(theta_init) * n_theta

# The ray's constants of motion
b     = p_phi
Bsqrt = r**2*(n_theta**2 + n_phi**2)



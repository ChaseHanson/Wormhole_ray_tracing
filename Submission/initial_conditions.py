# Program to set our initial conditions for building our map

import numpy as np
import scipy.integrate as integrate
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
drdl     = wormhole.calc_drdl(l_camera)
# The incoming light ray's canonical momenta
p_l     = n_l
p_phi   = r * n_phi
p_theta = r * np.sin(theta_init) * n_theta

# The ray's constants of motion
b     = p_phi
Bsqrd = r**2*(n_theta**2 + n_phi**2)

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

def integrate_geo_eqns():
    # initial conditions (lines 8 - 51)
    # y = np.zeros()
    def f(t, y):
        return geo_eqns(y[3], y[4], y[0], y[1], y[2], wormhole)
    # integrate with scipy.integrate.solve_ivp
    map = integrate.solve_ivp(f,(0,-100),geo_eqns(p_l,p_theta,l_camera,theta_camera,phi_camera,wormhole))
    print(map.y)
    print(np.shape(map.y))
integrate_geo_eqns()

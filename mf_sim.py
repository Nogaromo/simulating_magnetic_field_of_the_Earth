import numpy as np
from tqdm import tqdm
from scipy.special import lpmv, eval_legendre, jv, jnp_zeros, gamma
from scipy.constants import c, mu_0 as mu0, e, m_e, m_p
from scipy.optimize import curve_fit
from scipy.stats import rv_continuous
import scipy.integrate as integrate
from numba import njit, prange
import json
from sympy import assoc_legendre, cos, sin, lambdify, symbols, limit, pi
from sympy.abc import x, m, n, z
import os
import re
import multiprocessing as mp
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Configuring parameters of the magnetospehere and file managment')
parser.add_argument('--c_i', default=0.9, type=float, help='C_i coeffitient')
parser.add_argument('--c_a', default=0.01, type=float, help='C_a coeffitient')
parser.add_argument('--l', default=0.15, type=float, help='lambda coeffitient')
parser.add_argument('--start_index', default=0, type=int, help='lambda coeffitient')
parser.add_argument('--end_index', default=1_000_000, type=int, help='lambda coeffitient')
parser.add_argument('--batch_size', default=1_000, type=int, help='batch size to track progress')
parser.add_argument('--num_processes', default=6, type=int, help='number of processes to for multiprocessing')

# Setting up the constants
g_1_0 = -29404.8 * 1e-9*0
g_1_1 = -1450.9 * 1e-9*0
h_1_1 = 4652.5 * 1e-9*0
g_2_0 = -2499.6 * 1e-9
g_2_1 = 2982.0 * 1e-9*0
h_2_1 = -2991.6 * 1e-9*0
g_2_2 = 1677.0 * 1e-9*0
h_2_2 = -734.6 * 1e-9*0
Re = 6400 * 1e3
Rm = 16 * Re
R_to_moon = 384_400e3
R_moon = 1737.4e3
b = 5 * Re
# c = 3*1e8
# m_e = 1.67e-27
# e = 1.6e-19
# mu0 = 1.25663706*1e-6
m_x = 4*np.pi*Re**3/mu0*g_1_1
m_y = 4*np.pi*Re**3/mu0*h_1_1
m_z = 4*np.pi*Re**3/mu0*g_1_0
Q_xx = 4*np.pi*Re**4/mu0*(-g_2_0 + np.sqrt(3) * g_2_2)
Q_yy = 4*np.pi*Re**4/mu0*(-g_2_0 - np.sqrt(3) * g_2_2)
Q_xy = 4*np.pi*np.sqrt(3)*Re**4/mu0*h_2_2
Q_xz = 4*np.pi*np.sqrt(3)*Re**4/mu0*g_2_1
Q_yz = 4*np.pi*np.sqrt(3)*Re**4/mu0*h_2_1
phi_ecl = np.pi / 36
cos_ecl = np.cos(phi_ecl)
sin_ecl = np.sin(phi_ecl)
coef_mu0_4pi_b_3 = np.array([mu0/(4*np.pi*b**3)])
coef_mu0_4pi = -np.array([mu0/(4*np.pi)])
coef_mu0_8pi = np.array([mu0/(8*np.pi)])
B_imf = np.array([-5*1e-9, 0., 0.])
args = parser.parse_args()
Ci = args.c_i
Ca = args.c_a
l = args.l
n_start = args.start_index
n_end = args.end_index
###


def calc_a(k, x_k_i):
    global Rm, b
    f_0 = lambda x, N: 0.5*(0.5*x**2+b**2)/((b**2+x**2)**(5/2))-\
            1/(4*b**3)*np.sum([(n+1)*(n-1)*(-b/Rm)**(n+1)*(x/Rm)**n*lpmv(0, n, 0) for n in range(1, N+1)])
    f_1 = lambda x, N: -x*b/((b**2+x**2)**(5/2))-\
            1/(3*b**3)*np.sum([(n+1)*(n-1)/n*(-b/Rm)**(n+1)*(x/Rm)**n*lpmv(1, n, 0) for n in range(1, N+1)])
    f_2 = lambda x, N: -x**2/(2*(b**2+x**2)**(5/2))-\
            1/(6*b**3)*np.sum([(n+1)/n*(-b/Rm)**(n+1)*(x/Rm)**n*lpmv(2, n, 0) for n in range(1, N+1)])
    f = {0: f_0, 1: f_1, 2: f_2}
    a = 2/((1-(k/x_k_i)**2)*(Rm*jv(k, x_k_i))**2)*integrate.quad(lambda x: x*jv(k, x_k_i/Rm*x)*f[k](x, 100), 0, Rm)[0]
    return a


def set_x_and_a(N=200, load=False):
    x_0_i_list = jnp_zeros(0, N)
    x_1_i_list = jnp_zeros(1, N)
    x_2_i_list = jnp_zeros(2, N)
    if load:
        a_0_i_list = np.load("a_0_i_list.npy")
        a_1_i_list = np.load("a_1_i_list.npy")
        a_2_i_list = np.load("a_2_i_list.npy")
    else:
        a_0_i_list = []
        a_1_i_list = []
        a_2_i_list = []
        for i in tqdm(range(N)):
            a_0_i_list.append(calc_a(0, x_0_i_list[i]))
            a_1_i_list.append(calc_a(1, x_1_i_list[i]))
            a_2_i_list.append(calc_a(2, x_2_i_list[i]))
        a_0_i_list = np.array(a_0_i_list, dtype=float)
        a_1_i_list = np.array(a_1_i_list, dtype=float)
        a_2_i_list = np.array(a_2_i_list, dtype=float)

    return x_0_i_list, x_1_i_list, x_2_i_list, a_0_i_list, a_1_i_list, a_2_i_list


def set_limits_of_legendre(N=50, load=False):
    upper_bound_of_n = N
    if load:
        limits_at_zero_1 = np.load("limits_at_zero_1.npy")
        limits_at_pi_1 = np.load("limits_at_pi_1.npy")
    else:
        limits_at_zero_1 = [0.]
        limits_at_pi_1 = [0.]
        for i in tqdm(range(1, upper_bound_of_n + 1)):
            limits_at_zero_1.append(limit(assoc_legendre(i, 1, cos(x)).diff(), x, 0))
            limits_at_pi_1.append(limit(assoc_legendre(i, 1, cos(x)).diff(), x, pi))
        limits_at_zero_1 = np.array(limits_at_zero_1, dtype=float)
        limits_at_pi_1 = np.array(limits_at_pi_1, dtype=float)
    return limits_at_zero_1, limits_at_pi_1


def set_assoc_leg_1(N=50, load=False):
    upper_bound_of_n = N
    if load:
        assoc_leg_1_at_zero = np.load("assoc_leg_1_at_zero.npy")
        assoc_leg_1_at_pi = np.load("assoc_leg_1_at_pi.npy")
    else:
        assoc_leg_1_at_zero = []
        assoc_leg_1_at_pi = []
        for i in tqdm(range(1, upper_bound_of_n + 1)):
            assoc_leg_1_at_zero.append(limit(assoc_legendre(i, 1, cos(x)) / sin(x), x, 0))
            assoc_leg_1_at_pi.append(limit(assoc_legendre(i, 1, cos(x)) / sin(x), x, pi))
        assoc_leg_1_at_zero = np.array(assoc_leg_1_at_zero, dtype=float)
        assoc_leg_1_at_pi = np.array(assoc_leg_1_at_pi, dtype=float)
    return assoc_leg_1_at_zero, assoc_leg_1_at_pi


def gse_to_m(x_gse, y_gse, z_gse):
    global b
    return np.array([z_gse, y_gse, -(x_gse+b)])


def get_idx(sol):
    indexes = []
    for traj in sol:
        index = 0
        for point in traj:
            x, v_x, y, v_y, z, v_z = point
            x_ = -z-b
            y_ = y
            z_ = x
            x__ = x_*cos_ecl - z_*sin_ecl
            y__ = y_
            z__ = x_*sin_ecl + z_*cos_ecl
            if (R_to_moon + R_moon - np.sqrt(x__**2 + y__**2))**2+z__**2 < R_moon**2:
                indexes.append(index)
                break
            index += 1
    return indexes


def sample_start_points(N=10, coord_system='gse'):
    global b, Re
    phi = np.random.uniform(0, 2*np.pi, N)
    theta = np.random.uniform(0, np.pi, N)
    r = Re*np.random.uniform(2, 7, N)
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    if coord_system == 'gse':
        init = np.stack((x, y, z), axis=-1)
    if coord_system == 'm':
        init = np.stack((z, y, -(x+b)), axis=-1)
    return init


def sample_start_points_pe(N=10, coord_system='gse'):
    global b, Re
    phi = np.random.uniform(0, 2*np.pi, N)
    theta = np.random.uniform(0, np.pi, N)
    r = Re*np.random.uniform(3, 7, N)
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    if coord_system == 'gse':
        init = np.stack((x, y, z), axis=-1)
    if coord_system == 'm':
        init = np.stack((z, y, -(x+b)), axis=-1)
    return init


def ms(x, y, z, radius, resolution=100):
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*2j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)


def torus(x, y, z, resolution=100):
    global R_to_moon, R_moon, cos_ecl, sin_ecl, b
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:2*np.pi:resolution*2j]
    R = R_to_moon + R_moon
    X = (R + R_moon*(np.cos(v)))*np.cos(u)
    Y = (R + R_moon*(np.cos(v)))*np.sin(u)
    Z = R_moon*np.sin(v)
    X = X*cos_ecl - Z*sin_ecl - b
    Y = Y
    Z = X*sin_ecl + Z*cos_ecl

    return (X, Y, Z)


def initial_c(points, E):
    global c, m_p
    m = 16 * m_p
    y_ = []
    v_ = []
    v = np.sqrt(2*E/m)
    alpha_rotate = np.random.uniform(0, 2*np.pi, len(points))
    beta_rotate = np.random.uniform(0, 2*np.pi, len(points))
    gamma_rotate = np.random.uniform(0, 2*np.pi, len(points))
    for i in range(len(points)):
        alpha = alpha_rotate[i]
        beta = beta_rotate[i]
        gamma = gamma_rotate[i]
        C = np.array([[np.cos(beta)*np.cos(gamma), -np.sin(gamma)*np.cos(beta), np.sin(beta)],
                      [np.sin(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(gamma)*np.cos(alpha),
                       -np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), -np.sin(alpha)*np.cos(beta)],
                      [np.sin(alpha)*np.sin(gamma)-np.sin(beta)*np.cos(alpha)*np.cos(gamma),
                       np.sin(alpha)*np.cos(gamma)+np.sin(beta)*np.sin(gamma)*np.cos(alpha), np.cos(alpha)*np.cos(beta)]])
        new_v = v/np.sqrt(3) * np.array([1, 1, 1]) @ C
        v_.append(new_v)
    for i in range(len(points)):
        x, y, z = points[i][0], points[i][1], points[i][2]
        v_x, v_y, v_z = v_[i][0], v_[i][1], v_[i][2]
        y_.append([x, v_x, y, v_y, z, v_z])
    return y_


def initial_point(point, E):
    global c, m_p
    m = 16 * m_p
    y_ = []
    #v = np.sqrt(2*E/m)
    v = c*np.sqrt(1-1/(1+(E*e)/(m*c**2))**2)
    alpha = np.random.uniform(0, 2*np.pi, 1)
    beta = np.random.uniform(0, 2*np.pi, 1)
    gamma = np.random.uniform(0, 2*np.pi, 1)
    C = np.array([[np.cos(beta)*np.cos(gamma), -np.sin(gamma)*np.cos(beta), np.sin(beta)],
                  [np.sin(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(gamma)*np.cos(alpha),
                   -np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), -np.sin(alpha)*np.cos(beta)],
                  [np.sin(alpha)*np.sin(gamma)-np.sin(beta)*np.cos(alpha)*np.cos(gamma),
                   np.sin(alpha)*np.cos(gamma)+np.sin(beta)*np.sin(gamma)*np.cos(alpha), np.cos(alpha)*np.cos(beta)]])
    v_ = v/np.sqrt(3) * np.array([1, 1, 1]) @ C
    x, y, z = point[0], point[1], point[2]
    v_x, v_y, v_z = v_[0], v_[1], v_[2]
    y_.append([x, v_x, y, v_y, z, v_z])
    return y_


@njit
def transform(theta, phi):
    C = np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],
                  [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                  [-np.sin(phi), np.cos(phi), 0.]])
    return C


@njit
def cyl_transform(phi):
    C = np.array([[np.cos(phi), -np.sin(phi), 0.],
                  [np.sin(phi), np.cos(phi), 0.],
                  [0., 0., 1.]])
    return C


@njit
def B_qp_e(rr, tt, pp, x, y, z):
    global b, Q_xx, Q_yy, Q_xy, Q_xz, Q_yz, coef_mu0_4pi
    r = np.sqrt(x**2+y**2+(z+b)**2)
    a_q = Q_xx*(z+b)**2+Q_yy*y**2-(Q_xx+Q_yy)*x**2-2*Q_xy*(z+b)*y-2*Q_xz*(z+b)*x+2*Q_yz*x*y
    B_x = -((Q_xx+Q_yy)*x+Q_xz*(z+b)-Q_yz*y)/r**5-5*x*a_q/(2*r**7)
    B_y = (Q_yy*y-Q_xy*(z+b)+Q_yz*x)/r**5-5*y*a_q/(2*r**7)
    B_z = (Q_xx*(z+b)-Q_xy*y-Q_xz*x)/r**5-5*(z+b)*a_q/(2*r**7)
    return coef_mu0_4pi*np.array([B_x, B_y, B_z])


@njit
def B_cfi_r(n, r, theta, phi, cos_theta):
    global b, Rm, Q_xx, Q_yy, Q_xy, Q_xz, Q_yz, m_x, m_y, m_z
    B_r = (n+1)*(-b/Rm)**(n+2)*(r/Rm)**(n-1)*(
                    -n*(n-1)/(4*b)*Q_xx*lpmv(0., n, cos_theta)+
                    ((n-1)/(3*b)*Q_xz*np.cos(phi)+(n-1)/(3*b)*Q_xy*np.sin(phi))*(-lpmv(1., n, cos_theta))+
                    1/(6*b)*((0.5*Q_xx+Q_yy)*np.cos(2*phi)-Q_yz*np.sin(2*phi))*lpmv(2., n, cos_theta)
                                           )
    return B_r


@njit
def B_cfi_theta(n, r, theta, phi, sin_theta, cos_theta):
    global b, Rm, Q_xx, Q_yy, Q_xy, Q_xz, Q_yz, limits_at_zero_1, limits_at_pi_1

    if sin_theta != 0.:
        d_p_0 = -((1+n)/sin_theta*(cos_theta*lpmv(0., n, cos_theta)-lpmv(0., n+1., cos_theta)))
        d_p_1 = -cos_theta/sin_theta*lpmv(1., n, cos_theta)*(1+n)+n/sin_theta*lpmv(1., n+1., cos_theta)
        d_p_2 = -cos_theta/sin_theta*lpmv(2., n, cos_theta)*(1+n)+lpmv(2., n+1., cos_theta)/sin_theta*(n-1)
    else:
        d_p_0 = 0.
        d_p_2 = 0.
        int_n = int(n)
        if theta == 0.:
            d_p_1 = limits_at_zero_1[int_n]
        else:
            d_p_1 = limits_at_pi_1[int_n]
    B_theta = (n+1)/n*(-b/Rm)**(n+2)*(r/Rm)**(n-1)*(
        -n*(n-1)/(4*b)*Q_xx*d_p_0-d_p_1*(n-1)/(3*b)*(Q_xz*np.cos(phi)+Q_xy*np.sin(phi))+
        d_p_2/(6*b)*((0.5*Q_xx+Q_yy)*np.cos(2*phi)-Q_yz*np.sin(2*phi))
               )
    return B_theta


@njit
def B_cfi_phi(n, r, theta, phi, sin_theta, cos_theta):
    global b, Rm, Q_xx, Q_yy, Q_xy, Q_xz, Q_yz, assoc_leg_1_at_zero, assoc_leg_1_at_pi
    if sin_theta == 0.:
        int_n = int(n)
        assoc_leg_2 = 0.
        assoc_leg_1 = assoc_leg_1_at_zero[int_n] if theta == 0. else assoc_leg_1_at_pi[int_n]
    else:
        assoc_leg_1 = lpmv(1., n, cos_theta)/sin_theta
        assoc_leg_2 = lpmv(2., n, cos_theta)/sin_theta
    B_phi = (n+1)/n*(-b/Rm)**(n+2)*(r/Rm)**(n-1)*(
            -(n-1)/(3*b)*(-Q_xz*np.sin(phi)+Q_xy*np.cos(phi))*assoc_leg_1-
            1/(3*b)*((0.5*Q_xx+Q_yy)*np.sin(2*phi)+Q_yz*np.cos(2*phi))*assoc_leg_2
    )
    return B_phi


@njit
def B_tail_r(r, phi, z, a_0_i, a_2_i, x_0_i, x_2_i, x_1_i=None, a_1_i=None):
    global Rm, Q_xx
    B_tail_r_ = x_0_i/Rm*Q_xx*a_0_i*jv(1.,x_0_i/Rm*r)*np.exp(-x_0_i/Rm*z)+\
    x_1_i/Rm*1/2*a_1_i*(jv(2.,x_2_i/Rm*r)-jv(0.,x_2_i/Rm*r))*np.exp(-x_1_i/Rm*z)*(Q_xz*np.cos(phi)+Q_xy*np.sin(phi))+\
    x_2_i/Rm*1/2*a_2_i*(jv(3.,x_2_i/Rm*r)-jv(1.,x_2_i/Rm*r))*np.exp(-x_2_i/Rm*z)*((0.5*Q_xx+Q_yy)*np.cos(2*phi)+Q_yz*np.sin(2*phi))
    return B_tail_r_


@njit
def B_tail_phi(r, phi, z, a_2_i, x_2_i, x_1_i=None, a_1_i=None):
    global Rm, Q_xx
    B_tail_phi_ = 1/r*(Q_xz*np.sin(phi)-Q_xy*np.cos(phi))*jv(1.,x_1_i/Rm*r)*np.exp(-x_1_i/Rm*z)+\
    2/r*((0.5*Q_xx+Q_yy)*np.sin(2*phi)-Q_yz*np.cos(2*phi))*a_2_i*jv(2.,x_2_i/Rm*r)*np.exp(-x_2_i/Rm*z)
    return B_tail_phi_


@njit
def B_tail_z(r, phi, z, a_0_i, a_2_i, x_0_i, x_2_i, x_1_i=None, a_1_i=None):
    global Rm, Q_xx
    B_tail_z_ = x_0_i/Rm*Q_xx*a_0_i*jv(0.,x_0_i/Rm*r)*np.exp(-x_0_i/Rm*z)+\
    x_1_i/Rm*a_1_i*np.cos(phi)*jv(1.,x_2_i/Rm*r)*np.exp(-x_1_i/Rm*z)*(Q_xz*np.cos(phi)+Q_xy*np.sin(phi))+\
    x_2_i/Rm*a_2_i*jv(2.,x_2_i/Rm*r)*np.exp(-x_2_i/Rm*z)*((0.5*Q_xx+Q_yy)*np.cos(2*phi)+Q_yz*np.sin(2*phi))
    return B_tail_z_


@njit
def B_tail(r, theta, phi, N=20, lambd=0.15):
    global x_0_i_list, x_1_i_list, x_2_i_list, a_0_i_list, a_1_i_list, a_2_i_list, coef_mu0_8pi
    rho = r*np.sin(theta)
    z = r*np.cos(theta)
    B_r = 0.0
    B_phi = 0.0
    B_z = 0.0
    for i in range(N+1):
        x_0_i = x_0_i_list[i]
        x_1_i = x_1_i_list[i]
        x_2_i = x_2_i_list[i]
        a_0_i = a_0_i_list[i]
        a_1_i = a_1_i_list[i]
        a_2_i = a_2_i_list[i]
        B_r += B_tail_r(rho, phi, z, a_0_i, a_2_i, x_0_i, x_2_i, x_1_i=x_1_i, a_1_i=a_1_i)
        B_r += lambd*B_tail_r(rho, phi, lambd*z, a_0_i, a_2_i, x_0_i, x_2_i, x_1_i=x_1_i, a_1_i=a_1_i)
        B_phi += B_tail_phi(rho, phi, z, a_2_i, x_2_i, x_1_i=x_1_i, a_1_i=a_1_i)
        B_phi += lambd*B_tail_phi(rho, phi, lambd*z, a_2_i, x_2_i, x_1_i=x_1_i, a_1_i=a_1_i)
        B_z += B_tail_z(rho, phi, z, a_0_i, a_2_i, x_0_i, x_2_i, x_1_i=x_1_i, a_1_i=a_1_i)
        B_z += B_tail_z(rho, phi, lambd*z, a_0_i, a_2_i, x_0_i, x_2_i, x_1_i=x_1_i, a_1_i=a_1_i)

    return coef_mu0_8pi*np.array([B_r, B_phi, B_z])


@njit
def B_cfa(r, theta, phi, x, y, z, region='dayside'):
    global Rm, B_imf
    B_imf_x, B_imf_y, B_imf_z = B_imf
    if region == 'dayside':
        p_1_0 = lpmv(0., 1., np.cos(theta))
        p_1_1 = lpmv(1., 1., np.cos(theta))
        d_p_1_0 = -np.sin(theta)
        d_p_1_1 = -np.cos(theta)*np.sign(np.sin(theta))
        k_1 = -np.sign(np.sin(theta))
        B_r = -(Rm/r)**3*(B_imf_z*p_1_0-B_imf_x*p_1_1*np.cos(phi)-B_imf_y*p_1_1*np.sin(phi))
        B_theta = 0.5*(Rm/r)**3*(B_imf_z*d_p_1_0-B_imf_x*d_p_1_1*np.cos(phi)-B_imf_y*d_p_1_1*np.sin(phi))
        B_phi = -0.5*k_1*(Rm/r)**3*(-B_imf_x*np.sin(phi)+B_imf_y*np.cos(phi))
        return np.array([B_r, B_theta, -B_phi])
    else:
        B_r = -(Rm/(r*np.sin(theta)))**2*(B_imf_x*np.cos(phi)+B_imf_y*np.sin(phi))
        B_phi = (Rm/(r*np.sin(theta)))**2*(-B_imf_x*np.sin(phi)+B_imf_y*np.cos(phi))
        return np.array([B_r, -B_phi, 0.])


@njit
def B_cfi_2004_plus_earth_qp_symm(r, theta, phi, x, y, z, N=20, C_i=0.9, C_a=0.01, lambd=0.15):
    '''r: радиус в смещенной системе координат
       theta: угол относительно оси z в смещенной системе координат
       phi: полярный угол в смещенной системе координат
       N: число членов ряда'''
    global b, B_imf, coef_mu0_4pi_b_3
    if z > 0.:
        if (r*np.sin(theta)) < Rm:
            B_tail_qp_symm = B_tail(r, theta, phi, lambd=lambd)
            C = cyl_transform(phi)
            B = C @ B_tail_qp_symm + C_a*B_imf
        else:
            C = cyl_transform(phi)
            B = (1-C_a)*(C @ B_cfa(r, theta, phi, x, y, z, region='nightside')) + B_imf
    else:
        if r < Rm:
            B_r = 0.0
            B_theta = 0.0
            B_phi = 0.0
            for n in range(1, N+1):
                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)
                n = float(n)
                B_r += B_cfi_r(n, r, theta, phi, cos_theta)
                B_theta += B_cfi_theta(n, r, theta, phi, sin_theta, cos_theta)
                B_phi += B_cfi_phi(n, r, theta, phi, sin_theta, cos_theta)
            B_cfi = coef_mu0_4pi_b_3*np.array([B_r, B_theta, B_phi])
            B_e_qp_symm = B_qp_e(r, theta, phi, x, y, z)
            C = transform(theta, phi)
            B = (1-C_i)*(B_cfi @ C) + B_e_qp_symm + C_a*B_imf
        else:
            C = transform(theta, phi)
            B = (1-C_a)*(B_cfa(r, theta, phi, x, y, z, region='dayside') @ C) + B_imf

    return B


@njit
def eqn(y, t):
    global b, e, m_p, m_e, Re, Ci, Ca, l
    m = 16*m_p
    gamma = e / m
    y_1, y_2, y_3, y_4, y_5, y_6 = y
    r = np.sqrt(y_1**2+y_3**2+y_5**2)
    theta = np.arccos(y_5 / r)
    phi = np.arctan(y_3 / y_1) if y_1 > 0 else np.arctan(y_3 / y_1) + np.pi
    B_x, B_y, B_z = B_cfi_2004_plus_earth_qp_symm(r, theta, phi, y_1, y_3, y_5, C_i=Ci, C_a=Ca, lambd=l)
    dydt = [y_2, gamma*(y_4 * B_z - y_6 * B_y), y_4, -gamma*(y_2 * B_z - y_6 * B_x), y_6, gamma*(y_2 * B_y - y_4 * B_x)]
    return dydt


@njit
def check_in_torus(x, y, z):
    global cos_ecl, sin_ecl, R_to_moon, R_moon
    x_ = -z-b
    y_ = y
    z_ = x
    x__ = x_*cos_ecl - z_*sin_ecl
    y__ = y_
    z__ = x_*sin_ecl + z_*cos_ecl
    check_arr = (R_to_moon + R_moon - np.sqrt(x__**2 + y__**2))**2+z__**2
    check_arr_ = check_arr <= R_moon ** 2

    if np.unique(check_arr <= R_moon**2).shape[0] > 1:
        dropped = True
        itemindex = np.argwhere(check_arr_ == True)
        return dropped, z[check_arr_], y[check_arr_], x[check_arr_], itemindex[0][0]
    else:
        dropped = False
        return dropped, z[check_arr_], y[check_arr_], x[check_arr_], 0


def plot_traj(initial_conditions, t_0=7200, color='red'):
    global b
    moon_solutions = []
    moon_index = []
    check_res = []
    moon_statistic = 0
    all_particles = 0
    stuck_on_earth = 0
    stuck_on_earth_p = 0
    back = 0
    front = 0
    end_points = []
    t = np.linspace(0, t_0, t_0*10)

    all_particles += 1
    sol = integrate.odeint(eqn, initial_conditions, t)
    x_end = sol[:, 0][-1]
    y_end = sol[:, 0][-1]
    z_end = sol[:, 0][-1]
    if (x_end**2+y_end**2+z_end**2) < 100*Re**2:
        stuck_on_earth += 1

    res = check_in_torus(sol[:, 0], sol[:, 2], sol[:, 4])

    if res[0]:

        moon_statistic += 1
        curr_exp_path = fr"experiments\\{Ci}_{Ca}_{l}_{B_imf[0]}_{B_imf[1]}_{B_imf[2]}"
        curr_dir_path = curr_exp_path + "\\" + f"{n_start}_{n_end}" + "\\" + "moon_solutions"
        files_list = os.listdir(curr_dir_path)

        if len(files_list) != 0:
            # number_in_files = [int(re.findall(r'\d+', file)[0]) for file in files_list]
            last_name_number = len([int(re.findall(r'\d+', file)[0]) for file in files_list])
        else:
            last_name_number = 0

        save_path = curr_dir_path + fr"\moon_solution_{int(last_name_number) + 1}.txt"

        if last_name_number % 5 == 0:
            np.savetxt(save_path, sol[:res[4]])
        else:
            start_ = sol[0]
            end_ = sol[res[4]]
            first_and_last = np.stack((start_, end_))
            np.savetxt(save_path, first_and_last)

        if np.mean(res[1]) > 0.0:
            back += 1
        else:
            front += 1

    return moon_statistic, stuck_on_earth, back, front


x_0_i_list, x_1_i_list, x_2_i_list, a_0_i_list, a_1_i_list, a_2_i_list = set_x_and_a(load=True)
limits_at_zero_1, limits_at_pi_1 = set_limits_of_legendre(load=True)
assoc_leg_1_at_zero, assoc_leg_1_at_pi = set_assoc_leg_1(load=True)


if __name__ == '__main__':

    main_dir_path = fr"experiments\\{Ci}_{Ca}_{l}_{B_imf[0]}_{B_imf[1]}_{B_imf[2]}"

    try:
        os.mkdir(main_dir_path)
        os.mkdir(main_dir_path + "\\" + f"{n_start}_{n_end}")
        os.mkdir(main_dir_path + "\\" + f"{n_start}_{n_end}" + "\\" + "moon_solutions")
    except OSError:
        pass

    ics = np.load("start_cond_1M.npy")[n_start:n_end]

    batch_size = args.batch_size
    num_processes = args.num_processes
    N = len(ics) // batch_size
    full_stats = np.zeros((N, 4))
    k = 0
    for i in tqdm(range(0, len(ics), batch_size)):
        start_cond = ics[i:i+batch_size]
        p = mp.Pool(num_processes)
        mp_solutions = p.map(plot_traj, start_cond)
        full_stats[k] = np.sum(np.array(mp_solutions), axis=0)
        k += 1

    statistic = np.sum(full_stats, axis=0)
    statistic_dict = {
                    "moon": statistic[0],
                    "stuck_on_earth": statistic[1],
                    "back": statistic[2],
                    "front": statistic[3]
                     }

    with open(main_dir_path + r"\\" + f"{n_start}_{n_end}" + "\\" + "stats.json", 'w', encoding='utf-8') as file:
        json.dump(statistic_dict, file, ensure_ascii=False, indent=4)

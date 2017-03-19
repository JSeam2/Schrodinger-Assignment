import scipy.constants as c
from math import factorial as ff
import cmath as math
import numpy as np


# Question 1
def energy_n(n):
    """
    Create a function to calculate the energy level of a given
    principal quantum number. This function should take 1 int
    argument and return the energy level in eV. Round to 5
    decimal places

    :param: n(int) : nodes
    :output: float (rounded to 5 decimal places)
    """

    # Do we need to calculate the energy levels properly???
    # This feels rather hacky
    assert type(n) == int
    return round(-13.60569 / n ** 2, 5)

# Question 2
def degToRad(deg):
    """
    Convert deg to rad. 5 decimal places output

    :param: deg(float): degrees
    :output: rad(float): radians
    """
    # Convert to float if int
    if type(deg) == int:
        deg = float(deg)

    assert type(deg) == float
    return round(deg * 3.14159265359 / 180, 5)

def radToDeg(rad):
    """
    Convert rad to deg. 5 decimal places output

    :param: rad(float): radians
    :output: deg(float): degrees
    """
    # Convert to float if int
    if type(rad) == int:
        rad = float(rad)

    assert type(rad) == float
    return round(rad * 180 / 3.14159265359, 5)

def sphericalToCartesian(r, theta, phi):
    '''
    convert spherical coord to cartesian coord
    :param r: radius
    :param theta: theta angle
    :param phi: phi angle
    :return: (x, y, z) tuple of floats
    '''
    x = float(r) * np.sin(theta) * np.cos(phi)
    y = float(r) * np.sin(theta) * np.sin(phi)
    z = float(r) * np.cos(theta)

    return round(x, 5), round(y, 5), round(z, 5)

def cartesianToSpherical(x, y, z):
    x = float(x)
    y = float(y)
    z = float(z)
    r = round(np.sqrt(x ** 2 + y ** 2 + z ** 2), 5)
    theta = round(np.arctan2(y, x), 5)
    phi = round(np.arctan2(np.sqrt(x ** 2 + y ** 2), z), 5)

    return r, phi, theta

def p00(theta):
    return 1

def p01(theta):
    return np.cos(theta)

def p02(theta):
    return 0.5*(3*np.cos(theta)**2-1)

def p03(theta):
    return 0.5*(5*np.cos(theta)**3-3*np.cos(theta))

def p11(theta):
    return np.sin(theta)

def p12(theta):
    return 3*np.sin(theta)*np.cos(theta)

def p13(theta):
    return -1.5*np.sin(theta)*(5*np.cos(theta)**2-1)

def p22(theta):
    return 3*np.sin(theta)**2

def p23(theta):
    return 15*np.sin(theta)**2*np.cos(theta)

def p33(theta):
    return -15*np.sin(theta)**3

def assoc_legendre(m,l):
    if m==0 and l==0:
        return p00
    elif m==0 and l==2:
        return p02
    elif m==1 and l==1:
        return p11
    elif m==3 and l==3:
        return p33
    elif m==0 and l==1:
        return p01
    elif m==2 and l==3:
        return p23
    elif m==2 and l==2:
        return p22
    elif m==1 and l==3:
        return p13
    elif m==1 and l==2:
        return p12
    elif m==0 and l==3:
        return p03
    else:
        return None

def l00(x):
    return 1

def l01(x):
    return -x+1

def l02(x):
    return x*x-4*x+2

def l10(x):
    return 1

def l11(x):
    return -2*x+4

def l12(x):
    return 3*x*x-18*x+18

def l13(x):
    return -4*x*x*x+48*x*x-144*x+96

def l20(x):
    return 2

def l21(x):
    return -6*x+18

def l23(x):
    return -20*x*x*x+300*x*x-1200*x+1200

def l22(x):
    return 12*x*x-96*x+144

def l03(x):
    return (-x)**3+9*x**2-18*x+6

def l30(x):
    return 6

def l31(x):
    return -24*x+96

def l32(x):
    return 60*x*x-600*x+1200

def l33(x):
    return -120*(x**3 - 18*x**2 + 90*x - 120)

def assoc_laguerre(p,qmp):
    if p==0 and qmp==0:
        return l00
    elif p==0 and qmp==1:
        return l01
    elif p==0 and qmp==2:
        return l02
    elif p==0 and qmp==3:
        return l03
    elif p==1 and qmp==0:
        return l10
    elif p==1 and qmp==1:
        return l11
    elif p==1 and qmp==2:
        return l12
    elif p==1 and qmp==3:
        return l13
    elif p==2 and qmp==0:
        return l20
    elif p==2 and qmp==1:
        return l21
    elif p==2 and qmp==2:
        return l22
    elif p==2 and qmp==3:
        return l23
    elif p==3 and qmp==0:
        return l30
    elif p==3 and qmp==1:
        return l31
    elif p==3 and qmp==2:
        return l32
    elif p==3 and qmp==3:
        return l33
    else:
        return None

def angular_wave_func(m,l,theta,phi):
    if m >0:
        eps = (-1)**(m)
    else:
        eps = 1

    a = (2*l + 1)/(4*np.pi)
    cca = np.math.factorial(l-abs(m))
    ccb = np.math.factorial(l+abs(m))
    b = float(cca)/float(ccb)
    sqr = np.sqrt(a*b)
    #print sqr

    expon = np.exp(1j*m*float(phi))
    #print expon

    pfunc = assoc_legendre(m,l)
    y = pfunc(float(theta))

    #print(y)
    final = eps*sqr*expon*y
    return complex(round(final.real,5), round(final.imag,5))


def radial_wave_func(n,l,r):
    a = c.physical_constants['Bohr radius'][0]
    AA = (2.0 / (n*a))**3
    B = np.math.factorial(n-l-1)
    Ca = 2.0*n*((np.math.factorial(n+l))**3)
    BB= float(B)/Ca
    #print BB
    sqre = np.sqrt(AA*BB)
    #print sqre

    expp = np.exp(-float(r)/(n*a))

    Lexp = ((2*float(r))/(n*a))**l

    pfunc = assoc_laguerre(2*l+1, n-l-1)
    y = pfunc(((2*r)/(n*a)))

    return round(sqre*expp*Lexp*y / a**(-3.0/2),5)

'''
a = c.physical_constants["Bohr radius"][0]
print radial_wave_func(1,0,a) == 0.73576
print radial_wave_func(2,1,a) == 0.12381
print radial_wave_func(2,1,2*a) == 0.15019
print radial_wave_func(3,1,2*a) == 0.08281
print radial_wave_func(1,0,3*a) == 0.09957
'''

def hydrogen_wave_func(n, m, l, roa, Nx, Ny, Nz):
    # To find overall wavefunction
    # psi = Y(angle?,angle?) * R(y)

    #if m! = 0 real wave function is a linear combinaton of two stationary states

    a = c.physical_constants['Bohr radius'][0]

    # need to make the grid first
    X = np.linspace(-roa,roa,Nx)
    Y = np.linspace(-roa,roa,Ny)
    Z =  np.linspace(-roa,roa,Nz)

    xx,yy,zz = np.meshgrid(X,Y,Z)

    # convert the grid into spherical coordinate 
    CTOS = np.vectorize(cartesianToSpherical)
    r, t, p = CTOS(xx,yy,zz)

    a = c.physical_constants['Bohr radius'][0]
    rr = r * a

    Rad = np.vectorize(radial_wave_func)
    Ang = np.vectorize(angular_wave_func)

    mag = Rad(n,l,rr)* Ang(m,l,t,p)
    absmag = np.absolute(mag)**2

    # return the x,y,z values and magnitude

    return np.round(xx,5), np.round(yy,5), np.round(zz,5), np.round(absmag,5)

'''
x,y ,z, mag = hydrogen_wave_func(2,0,0,3,5,4,3)
#x,y,z,mag= hydrogen_wave_func(2,1,1,5,3,4,2)
print "x,y,z"
print x,y,z
print
print "mag"
print mag
#print hydrogen_wave_func(2,0,0,3,5,4,3) 
'''

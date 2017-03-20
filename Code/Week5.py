import scipy.constants as c
from math import factorial as ff
import cmath as math
import numpy as np



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
'''
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
    if m < 0:
        m = -m
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

def l03(x):
    return (-x)**3+9*x**2-18*x+6

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
'''

def p00(theta):
    return 1

def p01(theta):
    return np.cos(theta)

def p02(theta):
    return 0.5*(3*np.cos(theta)**2-1)

def p03(theta):
    return 0.5*(5*np.cos(theta)**3-3*np.cos(theta))

def p04(theta):
    return 0.125 * (35 * np.cos(theta) ** 4 - 30 * np.cos(theta) ** 2 + 3)

def p11(theta):
    return np.sin(theta)

def p12(theta):
    return 3*np.sin(theta)*np.cos(theta)

def p13(theta):
    return 1.5*np.sin(theta)*(5*np.cos(theta)**2-1)

def p14(theta):
    return 2.5 * (7 * np.cos(theta) ** 3 - 3 * np.cos(theta)) * np.sin(theta)

def p20(theta):
    return 0.5 * (3 * (np.cos(theta) ** 2) - 1)

def p22(theta):
    return 3*np.sin(theta)**2

def p23(theta):
    return 15*np.sin(theta)**2*np.cos(theta)

def p24(theta):
    return 7.5 * (7 * np.cos(theta) ** 2 - 1) * np.sin(theta) ** 2

def p33(theta):
    return 15*np.sin(theta)*(1-np.cos(theta)**2)

def p34(theta):
    return 105 * np.cos(theta) * np.sin(theta) ** 3

def p43(theta):
    return 105 * np.cos(theta) * np.sin(theta) ** 3

def p44(theta):
    return 105 * np.sin(theta) ** 4

def assoc_legendre(m,l):
    m=abs(m)
    if m==0 and l==0:
        return p00
    elif m==0 and l==1:
        return p01
    elif m==0 and l==2:
        return p02
    elif m==0 and l==3:
        return p03
    elif m==0 and l==4:
        return p04
    elif m==1 and l==1:
        return p11
    elif m==1 and l==2:
        return p12
    elif m==1 and l==3:
        return p13
    elif m==1 and l==4:
        return p14
    elif m==2 and l==0:
        return p20
    elif m==2 and l==2:
        return p22
    elif m==2 and l==3:
        return p23
    elif m==2 and l==4:
        return p24
    elif m==3 and l==3:
        return p33
    elif m==3 and l==4:
        return p34
    elif m==4 and l==3:
        return p43
    elif m==4 and l==4:
        return p44
    else:
        return None


def l00(x):
    return 1

def l01(x):
    return -x+1

def l02(x):
    return x*x-4*x+2

def l03(x):
    return -x*x*x+9*x*x-18*x+6

def l04(x):
    return x ** 4 - 16 * x ** 3 + 72 * x ** 2 - 96 * x +24

def l10(x):
    return 1

def l11(x):
    return -2*x+4

def l12(x):
    return 3*x*x-18*x+18

def l13(x):
    return -4*x*x*x+48*x*x-144*x+96

def l14(x):
    return 5 * x ** 4 - 100 * x ** 3 + 600 * x ** 2 - 1200 * x + 600

def l15(x):
    return -720 * (x - 6)

def l20(x):
    return 2

def l21(x):
    return -6*x+18

def l22(x):
    return 12*x*x-96*x+144

def l23(x):
    return -20*x*x*x+300*x*x-1200*x+1200

def l24(x):
    return 30 * x ** 4 - 720 * x ** 3 + 5400 * x ** 2 - 14400 * x + 10800

def l30(x):
    return 6

def l31(x):
    return -24*x+96

def l32(x):
    return 60*x*x-600*x+1200

def l33(x):
    return -120 * x ** 3 + 2160 * x ** 2 - 10800 * x + 14400

def l34(x):
    return -210 * x ** 4 + 5880 * x ** 3 - 52920 * x ** 2 + 176400 * x - 176400

def l44(x):
    return 1680 * x ** 4 - 53760 * x ** 3 + 564480 * x ** 2 - 2257920 * x + 2822400

def l50(x):
    return 120

def l51(x):
    return -720 * (x - 6)

def l70(x):
    return 5040

def assoc_laguerre(p,qmp):
    if p==0 and qmp==0:
        return l00
    elif p==0 and qmp==1:
        return l01
    elif p==0 and qmp==2:
        return l02
    elif p==0 and qmp==3:
        return l03
    elif p==0 and qmp==4:
        return l04
    elif p==1 and qmp==0:
        return l10
    elif p==1 and qmp==1:
        return l11
    elif p==1 and qmp==2:
        return l12
    elif p==1 and qmp==3:
        return l13
    elif p==1 and qmp==4:
        return l14
    elif p==1 and qmp==5:
        return l15
    elif p==2 and qmp==0:
        return l20
    elif p==2 and qmp==1:
        return l21
    elif p==2 and qmp==2:
        return l22
    elif p==2 and qmp==3:
        return l23
    elif p==2 and qmp==4:
        return l24
    elif p==3 and qmp==0:
        return l30
    elif p==3 and qmp==1:
        return l31
    elif p==3 and qmp==2:
        return l32
    elif p==3 and qmp==3:
        return l33
    elif p==3 and qmp==4:
        return l34
    elif p==4 and qmp==4:
        return l44
    elif p==5 and qmp==0:
        return l50
    elif p==5 and qmp==1:
        return l51
    elif p==7 and qmp==0:
        return l70
    else:
        return None

def angular_wave_func(m,l,theta,phi):
    if m >0:
        eps = (-1)**(m)
    else:
        eps = 1

    a = (2*l + 1)/(4*np.pi)
    cca = np.math.factorial(l-np.absolute(m))
    ccb = np.math.factorial(l+np.absolute(m))
    b = float(cca)/float(ccb)
    sqr = np.sqrt(a*b)
    #print sqr

    expon = np.exp(1j*m*float(phi))
    #print expon

    pfunc = assoc_legendre(m,l)
    y = pfunc(float(theta))

    #print(y)
    final = sqr*expon*y
    final = eps * final.real
    #return complex(round(final.real,5), round(final.imag,5))
    return np.round(final,5)


def complex_angular_wave_func(m,l,theta,phi):
    if m >0:
        eps = (-1)**(m)
    else:
        eps = 1

    a = (2*l + 1)/(4*np.pi)
    cca = np.math.factorial(l-np.absolute(m))
    ccb = np.math.factorial(l+np.absolute(m))
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

def radial_wave_func2(n,l,r):
    a = c.physical_constants['Bohr radius'][0]
    L = assoc_laguerre(2*l+1,n-l-1)
    C1 = (2.0/(n*a))**3
    C2 = np.math.factorial(n-l-1)/float(2*n*np.math.factorial(n+l)**3)
    root = np.sqrt(C1*C2)
    exp = np.exp(-r/(n*a))
    r2na = r*2/(n*a)
    R = ((root*exp)*(r2na**l)*(L(r2na))) / (a**(-1.5))
    return round(R,5)

def fact(n):
    result=1
    while n>1:
        result*=n
        n-=1
    return result


def radial_wave_func(n,l,r):
    a = c.physical_constants['Bohr radius'][0]
    AA = (2.0 / (n*a))**3
    B = fact(n-l-1)
    Ca = 2.0*n*((fact(n+l))**3)
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
print radial_wave_func(1,0,0) 
print radial_wave_func(2,1,a) == 0.12381
print radial_wave_func(2,1,2*a) == 0.15019
print radial_wave_func(3,1,2*a) == 0.08281
print radial_wave_func(1,0,3*a) == 0.09957
'''

def hydrogen_wave_func(n, l, m, roa, Nx, Ny, Nz):
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

    rr = r * a

    Rad = np.vectorize(radial_wave_func)
    Ang = np.vectorize(angular_wave_func)
    cAng = np.vectorize(complex_angular_wave_func)


    if m == 0:
        mag = Rad(n,l,rr)* Ang(m,l,t,p)
        absmag = np.absolute(mag)**2

    # return the x,y,z values and magnitude

    elif m > 0:
        ang = cAng(m,l,t,p) - cAng(-m,l,t,p)
        comb = (1/np.sqrt(2))*(Rad(n,l,rr)*ang)
        if m == 2:
            absmag = np.absolute(comb)**2
        else:
            absmag = np.absolute(comb.real)**2

    elif m < 0:
        ang = cAng(m,l,t,p) + cAng(-m,l,t,p)
        comb = (1/(1j*np.sqrt(2)))*(Rad(n,l,rr)*ang)
        absmag = np.absolute(comb.real)**2
        if m == -2:
            absmag = np.absolute(comb)**2
        else:
            absmag = np.absolute(comb.real)**2

    return np.round(xx,5), np.round(yy,5), np.round(zz,5), np.round(absmag,5)



if __name__ == "__main__":
    print angular_wave_func(1,1,np.pi-0.2,0.12411)

    '''
    x,y ,z, mag = hydrogen_wave_func(4,3,3,3,5,4,3)
    print "x,y,z"
    print x,y,z
    print
    print "mag"
    print mag
    '''

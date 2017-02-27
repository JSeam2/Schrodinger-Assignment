import scipy.constants as c
from math import factorial as ff
import cmath as math
import numpy as np

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
	return 1.5*np.sin(theta)*(5*np.cos(theta)**2-1)

def p22(theta):
	return 3*np.sin(theta)**2

def p23(theta):
	return 15*np.sin(theta)**2*np.cos(theta)

def p33(theta):
	return 15*np.sin(theta)*(1-np.cos(theta)**2)

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
    return -x*x*x+9*x*x-18*x+6

def l30(x):
	return 6

def l31(x):
	return -24*x+96

def l32(x):
	return 60*x*x-600*x+1200

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

	a = (2*l + 1)/(4*math.pi)
	cca = ff(l-abs(m))
	ccb = ff(l+abs(m))
	b = float(cca)/float(ccb)
	sqr = math.sqrt(a*b)
	#print sqr

	expon = math.exp(1j*m*float(phi))
	#print expon

	pfunc = assoc_legendre(m,l)
	y = pfunc(float(theta))

	#print(y)
	final = eps*sqr*expon*y
	return complex(round(final.real,5), round(final.imag,5))

#print('angular_wave_func(0,0,0,0)')
#print(angular_wave_func(0,0,0,0))

#print('angular_wave_func(0,1,c.pi,0)')
#print(angular_wave_func(0,1,math.pi,0))

#print('angular_wave_func(1,1,c.pi/2,c.pi)')
#print(angular_wave_func(1,1,math.pi/2,math.pi))

#print('angular_wave_func(0,2,c.pi,0)')
#print(angular_wave_func(0,2,math.pi,0))

def radial_wave_func(n,l,r):
	import math
	from math import factorial as ff
	a = c.physical_constants['Bohr radius'][0]
	#a = float(r)**(-3/2.0)
	AA = 2.0 / (n*a)
	AA = AA**3
	#print AA

	B = ff(n-l-1)
	Ca = 2*n*((ff(n+l))**3)
	BB= float(B)/Ca
	#print BB

	sqre = math.sqrt(AA*BB)
	#print sqre

	expp = math.exp(float(-r)/(n*a))

	Lexp = ((2*float(r))/(n*a))**l

	pfunc = assoc_laguerre(2*l+1, n-l-1)
	y = pfunc(((2*r)/(n*r)))

	return round(sqre*expp*Lexp*y / a**(-3.0/2),5)


'''
a = c.physical_constants['Bohr radius'][0]
print radial_wave_func(1,0,a)

print radial_wave_func(2,1,a)

print radial_wave_func(2,1,2*a)

print radial_wave_func(3,1,2*a)

print radial_wave_func(1,0,3*a)
'''


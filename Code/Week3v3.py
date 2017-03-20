from sympy import *
import math
def assoc_legendre(m,l):
	def f00(a):
		x = symbols('x')
		if m == 0 and l == 0:
			return 1
		else:
			diff_l =  diff((x**2 - 1)**l,x,l)
			fact_l = factorial(int(l))
			fact_l = 1.0/((fact_l) * (2**l))
			lp = fact_l * diff_l
			asslegen = ((1-x**2)**((abs(m))/2)) * (diff(lp,x,abs(m)))
			print asslegen
			return asslegen.subs({x:math.cos(a)})

	return f00

#print assoc_legendre(0,0)(1)
print assoc_legendre(0,1)(1)

def assoc_laguerre(p,qmp):
	def f(a):
		x = symbols('x')
		if p == 0 and qmp == 0:
			return 1
		else:
			y = (E**(-x))* (x**(qmp+p))
			eel = (E**(x))
			diff_l = eel * y.diff(x, qmp+p)
			ass_l = ((-1)**p)*diff_l.diff(x,p)
			return int(round(ass_l.evalf(subs={x:a}),0))
	return f

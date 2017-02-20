from __future__ import print_function
import numpy as np
import unittest
import scipy.constants as c

class codeTester(unittest.TestCase):
    def test_qn4(self):
        self.assertEqual(fact(3), 6)
        self.assertEqual(fact(5), 120)
        self.assertEqual(fact(4), 24)
        self.assertEqual(fact(1), 1)

    def test_qn5(self):
        self.assertAlmostEqual(assoc_legendre(0,0)(1),1)
        self.assertAlmostEqual(assoc_legendre(1,1)(1),0.841470984808)
        self.assertAlmostEqual(assoc_legendre(2,3)(1),5.73860550926)
        self.assertAlmostEqual(assoc_legendre(2,3)(0),0.0)




def fact(n):
    """
    factorial of n
    :param n: int
    :return: int
    """
    if type(n) != int:
        n = int(n)

    if n == 1:
        return 1

    else:
        acc = 1
        for x in range(1,n+1):
            acc = acc * x

    return int(acc)

from sympy import *
from math import factorial
import math
def assoc_legendre(m,l):
    def f00(a):
        x = symbols('x')
        #x, l, m = symbols('x l m')
        if m == 0 and l == 0:
            return 1
        else:
            diff_l =  diff((x**2 - 1)**l,x,l)
            #print(type(l))
            fact_l = factorial(int(l))
            fact_l = 1.0/((fact_l) * (2**l))
            lp = fact_l * diff_l

            asslegen = ((1-x**2)**((abs(m))/2)) * (diff(lp,x,abs(m)))
            return asslegen.evalf(subs={x:math.cos(a)})
            #return diff_l
    return f00


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(codeTester)
    unittest.TextTestRunner(verbosity=2).run(suite)
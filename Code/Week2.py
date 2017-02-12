from __future__ import print_function
import numpy as np
import unittest
import scipy.constants as c

"""
Testing Framework with unittest
"""
class codeTester(unittest.TestCase):
    def test_qn1(self):
        self.assertEqual(energy_n(1), -13.60569)
        self.assertEqual(energy_n(2), -3.40142)
        self.assertEqual(energy_n(3), -1.51174)

    def test_qn2(self):
        self.assertEqual(degToRad(90), 1.5708)
        self.assertEqual(degToRad(180), 3.14159)
        self.assertEqual(degToRad(270), 4.71239)
        self.assertEqual(radToDeg(3.14), 179.90875)
        self.assertEqual(radToDeg(3.14/2.0), 89.95437)
        self.assertEqual(radToDeg(3.14*3/4), 134.93156)

    def test_qn3(self):
        self.assertEqual(sphericalToCartesian(3, 0, np.pi), (-0.0, 0.0, 3.0))
        self.assertEqual(sphericalToCartesian(3, np.pi/2.0, np.pi/2.0), (0.0, 3.0, 0.0))
        self.assertEqual(sphericalToCartesian(3, np.pi, 0), (0.0, 0.0, -3.0))
        self.assertEqual(cartesianToSpherical(3, 0, 0), (3.0, 1.5708, 0.0))
        self.assertEqual(cartesianToSpherical(0, 3, 0), (3.0, 1.5708, 1.5708))
        self.assertEqual(cartesianToSpherical(0, 0, 3), (3.0, 0.0, 0.0))
        self.assertEqual(cartesianToSpherical(0, -3, 0), (3.0, 1.5708, -1.5708))


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
    return round(-13.60569/n**2, 5)

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
    return round(deg*3.14159265359/180, 5)

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

# Question 3
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

    return (round(x,5), round(y,5), round(z,5))

def cartesianToSpherical(x, y, z):
    r = round(np.sqrt(x**2 + y**2 + z**2),5)
    theta = round(np.arctan2(y,x),5)
    phi = round(np.arctan2(np.sqrt(x**2+y**2),z), 5)

    return (r, phi, theta)




# Run the testing framework to verify the code
if __name__ == "__main__":
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(codeTester)
    unittest.TextTestRunner(verbosity=3).run(suite)

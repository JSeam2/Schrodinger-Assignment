from __future__ import print_function
#import numpy
import unittest
#import scipy.constants as c

"""
Testing Framework with unittest
"""
class codeTester(unittest.TestCase):
    def qn1(self):
    	self.assertEqual(energy_n(1), -13.60569)
	self.assertEqual(energy_n(2), -3.40142)
	self.assertEqual(energy_n(3), -1.51174)

    def qn2(self):
    	self.assertEqual(deg_to_rad(90), 1.5708)
	self.assertEqual(deg_to_rad(180), 3.14159)
	self.assertEqual(deg_to_rad(270), 4,71239)
	self.assertEqual(rad_to_deg(3.14), 179.90875)
	self.assertEqual(rad_to_deg(3.14/2.0), 89.95437)
	self.assertEqual(rad_to_deg(3.14*3/4), 134.93156)

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
def deg_to_rad(deg):
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

def rad_to_deg(rad):
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



# Run the testing framework to verify the code
if __name__ == "__main__":
    unittest.main()    

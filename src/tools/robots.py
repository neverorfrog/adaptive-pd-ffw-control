from roboticstoolbox import *
import spatialmath.base.symbolic as sym
import numpy as np
import sympy


class OneLink(DHRobot):
    """
    Class that models a 1-link robot (for now planar in the xy plane) with fictituous dynamic parameters
    """

    def __init__(self):

        from math import pi
        zero = 0.0
        a1 = 0.5
        a2 = 0.5
        deg = pi / 180
            
        # links
        link1 = RevoluteDH(
            alpha = 0, #link twist
            a = 1, #link length
            d = 0, #offset along the z axis
            m = 1, #mass of the link
            r = [0.612,0,0], #position of COM with respect to link frame
            I=[0, 0, 7.5, 0, 0, 0], #inertia tensor,
            B = 1, #viscous friction
            qlim=[-135 * deg, 135 * deg]
        )

        links = [link1]

        super().__init__(links, name="Planar 1R", keywords=("planar",), symbolic = False)
        

class TwoLink(DHRobot):
    """
    Class that models a 2-link robot (for now planar in the xy plane) with fictituous dynamic parameters
    """

    def __init__(self):

        from math import pi
        zero = 0.0
        a1 = 0.45
        a2 = 0.3
        deg = pi / 180
            
        # links
        link1 = RevoluteDH(
            alpha = 0, #link twist
            a = a1, #link length
            d = 0, #offset along the z axis
            m = 23.902, #mass of the link
            r = [-0.35,0,0], #position of COM with respect to link frame
            I=[0, 0, 1.26, 0, 0, 0], #inertia tensor,
            B = 0, #viscous friction
            qlim=[-135 * deg, 135 * deg]
        )
        link2 = RevoluteDH(
            alpha = 0,
            a = a2,
            d = 0,
            m = 3.88,
            r = [-0.25,0,0],
            I=[0, 0, 0.093, 0, 0, 0],
            B = 0,
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )

        links = [link1, link2]

        super().__init__(links, name="Planar 2R", keywords=("planar",), symbolic = False)

        self.qr = np.array([0, pi / 2])
        self.qg = np.array([pi / 2, -pi/2])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qg", self.qg)
    
    def jacob0(self, q=None, T=None, half=None, start=None, end=None):
        J = DHRobot.jacob0(self,q)
        return J[0:2,:] 

class SymbolicPlanarRobot(DHRobot):
    """
    Class that models a 2-link robot (for now planar in the xy plane) with fictituous dynamic parameters
    """

    def __init__(self, n):

        pi = sym.pi()
        a = sym.symbol(f"a(1:{n+1})")
        zero = sym.zero()
        deg = pi / 180
            
        # links
        links = []
        for i in range(n):
            links.append(RevoluteDH(
                alpha = zero, #link twist
                a = a[i], #link length
                d = zero, #offset along the z axis
                qlim=[-135 * deg, 135 * deg] #TODO: is it correct to leave it as a number?
            ))

        super().__init__(links, name="Planar NR", keywords=("planar",), symbolic = True)
    

class ThreeLink(DHRobot):
    """
    Class that models a 3-link robot with fictituous dynamic parameters
    """

    def __init__(self, symbolic = False):

        if symbolic:
            pi = sym.pi()
            a1, a2, a3 = sympy.symbols("a1 a2 a3")
            zero = sym.zero()
        else:
            from math import pi
            zero = 0.0
            a1 = 0.5
            a2 = 0.5
        
        deg = pi / 180
            
        # links
        link1 = RevoluteDH(
            alpha = 0, #link twist
            a = a1, #link length
            d = 0, #offset along the z axis
            m = 10, #mass of the link
            r = [0.25,0,0], #position of COM with respect to link frame
            I=[0, 0, 5/24, 0, 0, 0], #inertia tensor,
            B = 0, #viscous friction
            qlim=[-135 * deg, 135 * deg]
        )
        link2 = RevoluteDH(
            alpha = 0,
            a = a2,
            d = 0,
            m = 5,
            r = [0.25,0,0],
            I=[0, 0, 5/48, 0, 0, 0],
            B = 0,
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )
        link3 = RevoluteDH(
            alpha = 0,
            a = a2,
            d = 0,
            m = 5,
            r = [0.25,0,0],
            I=[0, 0, 5/48, 0, 0, 0],
            B = 0,
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )

        links = [link1, link2, link3]

        super().__init__(links, name="3R", keywords=("planar",), symbolic = symbolic)
    
    def jacob0(self, q=None, T=None, half=None, start=None, end=None):
        J = DHRobot.jacob0(self,q)
        return J[0:3,:]       
    

class UncertainTwoLink(DHRobot):
    """
    Class that models a 2-link robot (for now planar in the xy plane) with fictituous dynamic parameters
    """

    def __init__(self, symbolic = False):

        if symbolic:
            pi = sym.pi()
            a1, a2 = sympy.symbols("a1 a2")
            zero = sym.zero()
        else:
            from math import pi
            zero = 0.0
            a1 = 1
            a2 = 1
        
        deg = pi / 180
            
        # links
        link1 = RevoluteDH(
            alpha = 0, #link twist
            a = a1, #link length
            d = 0, #offset along the z axis
            m = 1.4, #mass of the link
            r = [0.5,1.0,0], #position of COM with respect to link frame
            I=[0, 0, 1, 0, 1, 0], #inertia tensor,
            B = 1, #viscous friction
            qlim=[-135 * deg, 135 * deg]
        )
        link2 = RevoluteDH(
            alpha = 0,
            a = a2,
            d = 0,
            m = 0.3,
            r = [0.5,0,0.3],
            I=[0, 0.5, 0.2, 0, 0, 0],
            B = 0.7,
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )

        links = [link1, link2]

        super().__init__(links, name="Planar 2R uncertant", keywords=("planar",), symbolic = symbolic)
        
        self.q = [0,0]
        self.qd = [0,0]
        self.qdd = [0,0]

        self.qr = np.array([0, pi / 2])
        self.qg = np.array([pi / 2, -pi/2])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qg", self.qg)
    
    def jacob0(self, q=None, T=None, half=None, start=None, end=None):
        J = DHRobot.jacob0(self,q)
        return J[0:2,:] 
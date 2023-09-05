from roboticstoolbox import *
import spatialmath.base.symbolic as sym
import numpy as np
import sympy
from tools.utils import skew


class ParametrizedRobot(DHRobot):
    '''Takes a real robot and parametrizes its dynamic data (adding some noise if needed for adaptive control tests)'''
    def __init__(self, realrobot, dev_factor = 0.2, seed = 42):
        np.random.seed(seed)
        robot = realrobot
        n = len(robot.links)
        self.realrobot = realrobot
        super().__init__(robot.links, name=robot.name, gravity = robot.gravity)
        
        self.realpi = np.zeros(10*n) #real parameters
        self.pi = np.zeros(10*n) #the ones we believe are true
        
        for i in range(n):
            shift = 10*i

            real_I_link = robot.links[i].I + robot.links[i].m*np.matmul(skew(robot.links[i].r).T, skew(robot.links[i].r))
            real_mr = robot.links[i].m*robot.links[i].r 
            
            #Computation of actual dynamic parameters of the real robot (ground truth)
            self.realpi[shift+0] = robot.links[i].m 
            self.realpi[shift+1:shift+4] = real_mr
            self.realpi[shift+4:shift+10] = real_I_link[np.triu_indices(3)]

            #Computation of actual dynamic parameters of the belief robot
            self.pi[shift+0] = max(0, np.random.normal(robot.links[i].m, robot.links[i].m * dev_factor, 1))
            self.pi[shift+1:shift+4] = np.random.normal(real_mr, np.abs(real_mr * dev_factor), (3,))
            self.pi[shift+4:shift+10] = np.random.normal(real_I_link[np.triu_indices(3)], real_I_link[np.triu_indices(3)] * dev_factor, (6,))
        
        

class TwoLink(DHRobot):
    """
    Class that models a 2-link robot (for now planar in the xy plane) with fictituous dynamic parameters
    """

    def __init__(self):

        from math import pi
        zero = 0.0
        deg = pi / 180
            
        # links
        link1 = RevoluteDH(
            alpha = 0, #link twist
            a = 0.5, #link length
            d = 0, #offset along the z axis
            m = 10, #mass of the link
            r = [-0.35,0,0], #position of COM with respect to link frame
            I=[0.0, 1.1, 2.2, 0.1, 1.2, 0.2], #inertia tensor I = [I_xx, I_yy, I_zz,I_xy, I_yz, I_xz]
            qlim=[-135 * deg, 135 * deg]
        )
        link2 = RevoluteDH(
            alpha = 0,
            a = 0.5,
            d = 0,
            m = 10,
            r = [-0.25,0,0],
            I=[0, 0, 0.2, 0, 0, 0],
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )

        links = [link1, link2]

        super().__init__(links, name="Planar 2R", keywords=("planar",), symbolic = False)

        self.qr = np.array([0, pi / 2])
        self.qg = np.array([pi / 2, -pi/2])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qg", self.qg)
    

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
            I=[0, 0, 5/24, 0, 0, 0], #inertia tensor
            qlim=[-135 * deg, 135 * deg]
        )
        link2 = RevoluteDH(
            alpha = 0,
            a = a2,
            d = 0,
            m = 5,
            r = [0.25,0,0],
            I=[0, 0, 5/48, 0, 0, 0],
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )
        link3 = RevoluteDH(
            alpha = 0,
            a = a2,
            d = 0,
            m = 5,
            r = [0.25,0,0],
            I=[0, 0, 5/48, 0, 0, 0],
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )

        links = [link1, link2, link3]

        super().__init__(links, name="Planar 3R", keywords=("planar",), symbolic = symbolic)
        
    
class Polar2R(DHRobot):
    """
    Class that models a 2-link polar robot with fictituous dynamic parameters
    """
    def __init__(self):

        from math import pi
        deg = pi / 180
            
        # links
        link1 = RevoluteDH(
            alpha = pi/2, #link twist
            a = 0, #link length
            d = 0.5, #offset along the z axis
            m = 20, #mass of the link
            r = [0,-0.35,0], #position of COM with respect to link frame
            I=[0, 0.2, 0, 0, 0, 0], #inertia tensor
            qlim=[-135 * deg, 135 * deg]
        )
        link2 = RevoluteDH(
            alpha = 0,
            a = 0.5,
            d = 0,
            m = 10,
            r = [-0.25,0,0],
            I=[0, 0.1, 0.1, 0, 0, 0],
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )

        links = [link1, link2]

        super().__init__(links, name="Polar 2R", keywords=("polar",), symbolic = False)

        self.qr = np.array([0, pi / 2])
        self.qg = np.array([pi / 2, -pi/2])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qg", self.qg)
        

class Spatial3R(DHRobot):
    """
    Class that models a 3-link antropomorphic robot with fictituous dynamic parameters
    """
    def __init__(self):

        from math import pi
        deg = pi / 180
            
        # links
        link1 = RevoluteDH(
            alpha = pi/2, #link twist
            a = 0, #link length
            d = 0.5, #offset along the z axis
            m = 10, #mass of the link
            r = [0,-0.35,0], #position of COM with respect to link frame
            I=[0, 0.2, 0, 0, 0, 0], #inertia tensor,
            qlim=[-135 * deg, 135 * deg]
        )
        link2 = RevoluteDH(
            alpha = 0,
            a = 0.4,
            d = 0,
            m = 3,
            r = [-0.25,0,0],
            I=[0, 0.1, 0.1, 0, 0, 0],
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )
        link3 = RevoluteDH(
            alpha = 0,
            a = 0.4,
            d = 0,
            m = 3,
            r = [-0.25,0,0],
            I=[0, 0.05, 0.05, 0, 0, 0],
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )

        links = [link1, link2, link3]

        super().__init__(links, name="Spatial 3R", keywords=("polar",), symbolic = False)
    
    

class SCARA(DHRobot):
    """
    Class that models a SCARA type robot
    Source: https://espace.rmc.ca/jspui/bitstream/11264/942/1/HYBRID%20FORCE-POSITION%20CONTROL%20OF%20A%204-DOF%20SCARA%20MANIPULATOR%20%2827%20October%202022%29.pdf
    """
    def __init__(self):

        from math import pi
        deg = pi / 180
            
        # links
        link1 = RevoluteDH(
            alpha = 0, #link twist
            a = 0.4, #link length
            d = 0.5, #offset along the z axis
            m = 6.01, #mass of the link
            r = [-0.185,0,0], #position of COM with respect to link frame
            I=[0.0132, 0.1810, 0.1807, 0, 0, 0], #inertia tensor,
            qlim=[-135 * deg, 135 * deg]
        )
        link2 = RevoluteDH(
            alpha = 0,
            a = 0.4,
            d = 0,
            m = 5.37,
            r = [-0.224,0,0],
            I=[0.0234, 0.1261, 0.1558, 0, 0, 0],
            qlim=[-135 * deg, 135 * deg]  # minimum and maximum joint angle
        )
        link3 = PrismaticDH(
            theta = 0,
            a = 0,
            alpha = 0,
            m = 4.03,
            r = [0,0,-0.201], #position of COM with respect to link frame
            I=[0.0802, 0.0802, 0.0064, 0, 0, 0], #inertia tensor,
            qlim=[-0.3, 0.1] 
        )
        link4 = RevoluteDH(
            alpha = pi, #link twist
            a = 0, #link length
            d = 0, #offset along the z axis
            m = 0.91, #mass of the link
            r = [0,0,-0.122], #position of COM with respect to link frame
            I=[0.0016, 0.0016, 0.0025, 0, 0, 0], #inertia tensor,
            qlim=[-135 * deg, 135 * deg]
        )

        links = [link1, link2, link3, link4]

        super().__init__(links, name="SCARA", keywords=("scara",), symbolic = False)

        self.qr = np.array([0, pi / 2])
        self.qg = np.array([pi / 2, -pi/2])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qg", self.qg)
 
class UR3(DHRobot):
    """
    Class that models a Universal Robotics UR3 manipulator
    """ 

    def __init__(self):


        from math import pi
        zero = 0.0
        deg = pi / 180

        # robot length values (metres)
        a = [0, -0.24365, -0.21325, 0, 0, 0]
        d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]
        
        #link twists
        alpha = [pi / 2, zero, zero, pi / 2, -pi / 2, zero]

        # mass data, no inertia available
        mass = [2, 3.42, 1.26, 0.8, 0.8, 0.35]
        coms = [
            [0, -0.02, 0],
            [0.13, 0, 0.1157],
            [0.05, 0, 0.0238],
            [0, 0, 0.01],
            [0, 0, 0.01],
            [0, 0, -0.02],
        ]
        I_link = [0,0,0,0,0,0]
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], m=mass[j], r=coms[j], I = I_link
            )
            links.append(link)

        super().__init__(
            links,
            name="UR3",
            manufacturer="Universal Robotics",
            keywords=("dynamics", "symbolic")
        )

        self.qr = np.array([180, 90, 0, 45, 90, 0]) * deg
        self.qz = np.zeros(6)
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)
    
        
class Panda(DHRobot):
    
    def __init__(self):
        from math import pi
        
        #Kinematic Data
        d1 = 0.3330 #base-spalla
        d3 = 0.3160 #spalla-gomito
        d5 = 0.3840 #gomito-polso
        d7 = 0.1070 #polso-tcp
        a3 = 0.0825 #offset1 gomito
        a4 = -0.0825 #offset2 gomit
        a6 = 0.0880 #offset polso
        alpha = [pi/2, -pi/2, pi/2, -pi/2, pi/2, pi/2, 0]
        a =     [0,     0,    a3,    a4,   0,    a6,   0]
        d =     [d1,    0,    d3,    0,    d5,   0,   d7]
        
        #Dynamic Data
        mass = [4.9706, 0.6469, 3.2286, 3.5878, 1.2259, 1.6665, 7.35e-1]
        cx   = [3.875e-03, -3.141e-03, 2.7518e-02,  -5.317e-02,   -1.1953e-02,  6.0149e-02,  1.0517e-02]
        cy   = [2.081e-03, -2.872e-02, 3.9252e-02,  1.04419e-01,   4.1065e-02,  -1.4117e-02, -4.252e-03]
        cz   = [2.081e-03,  3.495e-03,-6.6502e-02,  2.7454e-02,   -3.8437e-02 , -1.0517e-02, 6.1597e-02]
        inertia = []      #Ixx         Iyy         Izz         Ixy          Iyz           Ixz
        inertia.append([7.0337e-01, 7.0661e-01, 9.1170e-03, -1.3900e-04,  1.9169e-02,  6.7720e-03])
        inertia.append([7.9620e-03, 2.8110e-02, 2.5995e-02, -3.9250e-03,  7.0400e-04,  1.0254e-02])
        inertia.append([3.7242e-02, 3.6155e-02, 1.0830e-02, -4.7610e-03, -1.2805e-02, -1.1396e-02])
        inertia.append([2.5853e-02, 1.9552e-02, 2.8323e-02,  7.7960e-03,  8.6410e-03, -1.3320e-03])
        inertia.append([3.5549e-02, 2.9474e-02, 8.6270e-03, -2.1170e-03,  2.2900e-04, -4.0370e-03])
        inertia.append([1.9640e-03, 4.3540e-03, 5.4330e-03,  1.0900e-04,  3.4100e-04, -1.1580e-03])
        inertia.append([1.2516e-02, 1.0027e-02, 4.8150e-03, -4.2800e-04, -7.4100e-04, -1.1960e-03])
    
        links = []
        for j in range(7):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], 
                m=mass[j], r=[cx[j],cy[j],cz[j]], I=inertia[j]
            )
            links.append(link)
            
        super().__init__(links,name="Panda")     
        
                
if __name__ == "__main__":
    robot = Panda()
    print(robot.links[2].r)

                          
class Puma560(DHRobot):
    """Class that models a Puma 560 manipulator"""
    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        base = 26.45 * inch  # from mounting surface to shoulder axis

        L = [
            RevoluteDH(
                d=base,  # link length (Dennavit-Hartenberg notation)
                a=0,  # link offset (Dennavit-Hartenberg notation)
                alpha=pi / 2,  # link twist (Dennavit-Hartenberg notation)
                I=[0, 0.35, 0, 0, 0, 0],
                # inertia tensor of link with respect to
                # center of mass I = [L_xx, L_yy, L_zz,
                # L_xy, L_yz, L_xz]
                r=[0, 0, 0],
                # distance of ith origin to center of mass [x,y,z]
                # in link reference frame
                m=0,  # mass of link
                qlim=[-160 * deg, 160 * deg],  # minimum and maximum joint angle
            ),
            RevoluteDH(
                d=0,
                a=0.4318,
                alpha=zero,
                I=[0.13, 0.524, 0.539, 0, 0, 0],
                r=[-0.3638, 0.006, 0.2275],
                m=17.4,
                qlim=[-110 * deg, 110 * deg],  # qlim=[-45*deg, 225*deg]
            ),
            RevoluteDH(
                d=0.15005,
                a=0.0203,
                alpha=-pi / 2,
                I=[0.066, 0.086, 0.0125, 0, 0, 0],
                r=[-0.0203, -0.0141, 0.070],
                m=4.8,
                qlim=[-135 * deg, 135 * deg],  # qlim=[-225*deg, 45*deg]
            ),
            RevoluteDH(
                d=0.4318,
                a=0,
                alpha=pi / 2,
                I=[1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
                r=[0, 0.019, 0],
                m=0.82,
                qlim=[-266 * deg, 266 * deg],  # qlim=[-110*deg, 170*deg]
            ),
            RevoluteDH(
                d=0,
                a=0,
                alpha=-pi / 2,
                I=[0.3e-3, 0.4e-3, 0.3e-3, 0, 0, 0],
                r=[0, 0, 0],
                m=0.34,
                qlim=[-100 * deg, 100 * deg],
            ),
            RevoluteDH(
                d=0,
                a=0,
                alpha=zero,
                I=[0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0],
                r=[0, 0, 0.032],
                m=0.09,
                qlim=[-266 * deg, 266 * deg],
            ),
        ]

        super().__init__(
            L,
            name="Puma560",
            manufacturer="Unimation",
            keywords=("dynamics", "symbolic", "mesh"),
            symbolic=symbolic,
            meshdir="meshes/UNIMATE/puma560",
        )

        self.qr = np.array([0, pi / 2, -pi / 2, 0, 0, 0])
        self.qz = np.zeros(6)

        # nominal table top picking pose
        self.qn = np.array([0, pi / 4, pi, 0, pi / 4, 0])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qn", self.qn)

        # straight and horizontal
        self.addconfiguration_attr("qs", np.array([0, 0, -pi / 2, 0, 0, 0]))




class HeadlessPuma560(DHRobot):
    """
    Class that models a Puma 560 manipulator
    """

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi

            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        base = 26.45 * inch  # from mounting surface to shoulder axis

        L = [
            RevoluteDH(
                d=base,  # link length (Dennavit-Hartenberg notation)
                a=0,  # link offset (Dennavit-Hartenberg notation)
                alpha=pi / 2,  # link twist (Dennavit-Hartenberg notation)
                I=[0, 0.35, 0, 0, 0, 0],
                # inertia tensor of link with respect to
                # center of mass I = [L_xx, L_yy, L_zz,
                # L_xy, L_yz, L_xz]
                r=[0, 0, 0],
                # distance of ith origin to center of mass [x,y,z]
                # in link reference frame
                m=0,  # mass of link
                qlim=[-160 * deg, 160 * deg],  # minimum and maximum joint angle
            ),
            RevoluteDH(
                d=0,
                a=0.4318,
                alpha=zero,
                I=[0.13, 0.524, 0.539, 0, 0, 0],
                r=[-0.3638, 0.006, 0.2275],
                m=17.4,
                qlim=[-110 * deg, 110 * deg],  # qlim=[-45*deg, 225*deg]
            ),
            RevoluteDH(
                d=0.15005,
                a=0.0203,
                alpha=-pi / 2,
                I=[0.066, 0.086, 0.0125, 0, 0, 0],
                r=[-0.0203, -0.0141, 0.070],
                m=4.8,
                qlim=[-135 * deg, 135 * deg],  # qlim=[-225*deg, 45*deg]
            ),
            RevoluteDH(
                d=0.4318,
                a=0,
                alpha=pi / 2,
                I=[1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
                r=[0, 0.019, 0],
                m=0.82,
                qlim=[-266 * deg, 266 * deg],  # qlim=[-110*deg, 170*deg]
            )
        ]

        super().__init__(
            L,
            name="Headless Puma 560",
            manufacturer="Unimation",
            keywords=("dynamics", "symbolic", "mesh"),
            symbolic=symbolic,
            meshdir="meshes/UNIMATE/puma560"
        )

        self.qr = np.array([0, pi / 2, -pi / 2, 0,])
        self.qz = np.zeros(6)

        # nominal table top picking pose
        self.qn = np.array([0, pi / 4, pi, 0])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qn", self.qn)

        # straight and horizontal
        self.addconfiguration_attr("qs", np.array([0, 0, -pi / 2, 0]))
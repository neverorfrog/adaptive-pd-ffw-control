from roboticstoolbox import *
import numpy as np
import spatialmath.base.symbolic as sym
import sympy
from spatialmath import SE3, base
from tools.Utils import index2var, coeff_dict

class EulerLagrange():

    def __init__(self, n, robot):

        self.robot = robot
        self.n = n
        q = sym.symbol(f"q(1:{n+1})")  # link variables
        q_d = sym.symbol(f"q_dot_(1:{n+1})")
        a = sym.symbol(f"a(1:{n+1})")  # link lenghts

        # dynamic parameters
        g0 = sym.symbol("g")
        m = sym.symbol(f"m(1:{n+1})")  # link masses
        dc = sym.symbol(f"dc(1:{n+1})")
        Ixx = sym.symbol(f"Ixx(1:{n+1})")
        Iyy = sym.symbol(f"Iyy(1:{n+1})")
        Izz = sym.symbol(f"Izz(1:{n+1})")
        # Ixy = sym.symbol(f"Ixy(1:{n+1})")
        # Ixz = sym.symbol(f"Ixz(1:{n+1})")
        # Iyz = sym.symbol(f"Iyz(1:{n+1})")

        I = np.full((3,3), sym.zero(), dtype = object) #ith matrix representing inertia matrix of ith COM
        ri = np.full((3,), sym.zero(), dtype = object) #vector from RF i-1 to i wrt RF i-1
        rc = np.full((3,), sym.zero(), dtype = object) #position vector of COM i seen from RF i
        rim1c = np.full((3,), sym.zero(), dtype = object) #vector from RF i-1 to COM as seen from RF i
        riim1 = np.full((3,), sym.zero(), dtype = object) #vector from RF i to RF i-1 as seen from RF i
        Rinv = np.full((3,3), sym.zero(), dtype = object) #ith matrix representing rotation from Rf i to Rf i-1
        w = np.full((3,), sym.zero(), dtype = object) #angular velocity of link i wrt RF i
        v = np.full((3,), sym.zero(), dtype = object) #linear velocity of link i wrt RF i
        T = 0 #total kinetic energy of the robot
        U = 0 #total potential energy of the robot
        gv = np.array([0, -g0, 0])
        

        for i in range(n):
            #Preprocessing
            sigma = int(robot.links[i].isprismatic) #check for prismatic joints
            I = np.diag([Ixx[i],Iyy[i],Izz[i]]) #diagonal inertias
            A = robot[i].A(q[i]) #homogeneus transformation from frame i to i+1
            ri = A.t
            Ainv = A.inv()
            Rinv = Ainv.R #rotation from frame i+1 to i
            riim1 = Ainv.t

            #COM Position
            if sigma == 0:
                rim1c = Rinv @ [elem.subs(a[i],dc[i]) for elem in ri]
                rc = riim1 + rim1c
            else:
                rc = [elem.subs(q[i],dc[i]) for elem in riim1]
            rc = sym.simplify(rc)
                
            #Kinetic Energy
            w_im1 = w + (1-sigma) * q_d[i] * np.array([0,0,1]) #omega of link i wrt RF i-1 (3 x 1) 
            w = sym.simplify(Rinv @ w_im1)
            v_im1 = v + sigma * q_d[i] * np.array([0,0,1]) + np.cross(w_im1, ri) #linear v of link i wrt RF i-1
            v = sym.simplify(Rinv @ v_im1); 
            vc = sym.simplify(v + np.cross(w,rc))
            Ti = 0.5*(m[i]*np.matmul(vc, vc) + np.matmul(np.matmul(w, I), w))
            Ti = sympy.simplify(sympy.collect(sympy.expand(Ti),[*q_d]))
            T = sym.simplify(T + Ti)
            
            #Potential Energy
            hom = robot.A(i,q).A #transformation from RF 0 to RF i+1
            rc = np.array([*rc , 1])
            r0ci = sym.simplify(hom@rc)[:-1]
            Ui = sym.simplify(-m[i]* np.matmul(gv,r0ci))
            U = U + Ui
            
        
        #Inertia Matrix
        coeffs = coeff_dict(T, *q_d)
        self.M = np.full((n,n), sym.zero(), dtype = object)
        for row in range(n):
            self.M[row,:] = [coeffs[index2var(row,column,q_d)] for column in range(n)]
                    
        #Coriolis and centrifugal terms and gravity terms
        self.c = np.full((n,), sym.zero(), dtype = object)
        self.g = np.full((n,), sym.zero(), dtype = object)
        M = sympy.Matrix(self.M)
        for i in range(n):
            C_temp = M[:,i].jacobian(q)
            C = sym.simplify(0.5 * (C_temp + C_temp.T - M.diff(q[i])))
            self.c[i] = sym.simplify(np.matmul(np.matmul(q_d, C),q_d))
            self.g[i] = U.diff(q[i])
            

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
    

class UncertantTwoLink(DHRobot):
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

class Puma560(DHRobot):
    """
    Class that models a Puma 560 manipulator
    
    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. note::
        - SI units are used.
        - The model includes armature inertia and gear ratios.
        - The value of m1 is given as 0 here.  Armstrong found no value for it
          and it does not appear in the equation for tau1 after the
          substituion is made to inertia about link frame rather than COG
          frame.
        - Gravity load torque is the motor torque necessary to keep the joint
          static, and is thus -ve of the gravity caused torque.

    .. codeauthor:: Peter Corke
    """  # noqa

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
                Jm=200e-6,  # actuator inertia
                G=-62.6111,  # gear ratio
                B=1.48e-3,  # actuator viscous friction coefficient (measured
                # at the motor)
                Tc=[0.395, -0.435],
                # actuator Coulomb friction coefficient for
                # direction [-,+] (measured at the motor)
                qlim=[-160 * deg, 160 * deg],  # minimum and maximum joint angle
            ),
            RevoluteDH(
                d=0,
                a=0.4318,
                alpha=zero,
                I=[0.13, 0.524, 0.539, 0, 0, 0],
                r=[-0.3638, 0.006, 0.2275],
                m=17.4,
                Jm=200e-6,
                G=107.815,
                B=0.817e-3,
                Tc=[0.126, -0.071],
                qlim=[-110 * deg, 110 * deg],  # qlim=[-45*deg, 225*deg]
            ),
            RevoluteDH(
                d=0.15005,
                a=0.0203,
                alpha=-pi / 2,
                I=[0.066, 0.086, 0.0125, 0, 0, 0],
                r=[-0.0203, -0.0141, 0.070],
                m=4.8,
                Jm=200e-6,
                G=-53.7063,
                B=1.38e-3,
                Tc=[0.132, -0.105],
                qlim=[-135 * deg, 135 * deg],  # qlim=[-225*deg, 45*deg]
            ),
            RevoluteDH(
                d=0.4318,
                a=0,
                alpha=pi / 2,
                I=[1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
                r=[0, 0.019, 0],
                m=0.82,
                Jm=33e-6,
                G=76.0364,
                B=71.2e-6,
                Tc=[11.2e-3, -16.9e-3],
                qlim=[-266 * deg, 266 * deg],  # qlim=[-110*deg, 170*deg]
            ),
            RevoluteDH(
                d=0,
                a=0,
                alpha=-pi / 2,
                I=[0.3e-3, 0.4e-3, 0.3e-3, 0, 0, 0],
                r=[0, 0, 0],
                m=0.34,
                Jm=33e-6,
                G=71.923,
                B=82.6e-6,
                Tc=[9.26e-3, -14.5e-3],
                qlim=[-100 * deg, 100 * deg],
            ),
            RevoluteDH(
                d=0,
                a=0,
                alpha=zero,
                I=[0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0],
                r=[0, 0, 0.032],
                m=0.09,
                Jm=33e-6,
                G=76.686,
                B=36.7e-6,
                Tc=[3.96e-3, -10.5e-3],
                qlim=[-266 * deg, 266 * deg],
            ),
        ]

        super().__init__(
            L,
            name="Puma 560",
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
from control.trajectory_control import TrajectoryControl
import numpy as np
import sympy as sp

import spatialmath.base.symbolic as sym

from tools.dynamics import EulerLagrange
from tools.utils import *


class TestAdaptive_FFW(TrajectoryControl):

    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = 700):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = u_bound 

        self.ffw_test = True
        self.m_test = True
        self.s_test = True
        self.g_test = True
        self.convergency = False


    def feedback(self):

        q = self.robot.q
        qd = self.robot.qd
        qdd = self.robot.qdd

        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        e = q_d - q
        ed = qd_d - qd

        realY = self.dynamicModel.evaluateMatrix(self.dynamicModel.Y,q,qd,qd,qdd)

        gtFFW = self.robot.inertia(q) @ qdd + self.robot.coriolis(q, qd) @ qd + self.robot.gravload(q, gravity = -self.gravity)
        predFFW = np.matmul(realY, self.robot.realpi)

        gtM = self.robot.inertia(q)
        predM = self.dynamicModel.inertia(q,True)

        gtS = self.robot.coriolis(q,qd)
        predS = self.dynamicModel.coriolis(q,qd,True)

        gtG = self.robot.gravload(q, gravity = -self.gravity)
        predG = np.squeeze(self.dynamicModel.gravity(q,True))


        ffwError = np.linalg.norm(np.array(gtFFW - predFFW, dtype=np.float64))
        mError = np.linalg.norm(np.array(gtM - predM, dtype=np.float64))
        sError = np.linalg.norm(np.array(gtS - predS, dtype=np.float64))
        gError = np.linalg.norm(np.array(gtG - predG, dtype=np.float64))

        self.m_test &= mError < 1e-10
        self.s_test &= sError < 1e-10
        self.g_test &= gError < 1e-10

        self.ffw_test &= ffwError < 1e-10

        arrived = self.check_termination(e,ed)

        Y = self.dynamicModel.evaluateY(q_d, qd_d, qd_d, qdd_d)
        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(Y, self.robot.pi).astype(np.float64)
        torque = np.clip(torque, -self.u_bound, self.u_bound)
        self.append(q_d,qd_d,qdd_d,torque)


        gainMatrix = np.eye(len(self.robot.pi)) * 0.1
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaPi = gainMatrix @ (Y.T @ (sat_e + ed))
        self.robot.pi = self.robot.pi + deltaPi

        self.append(q_d,qd_d,qdd_d,[1,1])

        self.convergency |= arrived


        myY = self.dynamicModel.Y

        q_sym = sym.symbol(f"q(1:{self.dynamicModel.n+1})") 
        qd_sym = sym.symbol(f"q_dot_(1:{self.dynamicModel.n+1})")
        qdd_sym = sym.symbol(f"q_dot_dot_(1:{self.dynamicModel.n+1})")
        qd_S_sym = sym.symbol(f"q_dot_S_(1:{self.dynamicModel.n+1})")

        #print(myY[0][0])
        #vars = q_sym + qd_sym + qdd_sym + qd_S_sym
        
        pol = sympy.Poly(myY[0][0])
        print(myY[0][0])
        print()
        print(pol)
        print(pol.coeffs())

        exit(0)

        return torque, arrived or self.t[-1]/2 > self.reference.T + 1
    

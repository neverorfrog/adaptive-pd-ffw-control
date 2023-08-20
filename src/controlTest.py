from control.adaptive_control import AdaptiveControl
import numpy as np
import sympy as sp

class TestAdaptive_FFW(AdaptiveControl):

    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0], plotting = True):
        super().__init__(robot, env, dynamicModel, gravity, plotting)
        self.ffw_test = True
        self.m_test = True
        self.s_test = True
        self.g_test = True


    def feedback(self):

        q = self.robot.q
        qd = self.robot.qd
        qdd = self.robot.qdd

        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        e = q_d - q
        ed = qd_d - qd

        realY = self.dynamicModel.evaluateMatrix(self.dynamicModel.Y,q,qd,qd,qdd)

        gtFFW = self.robot.inertia(q) @ qdd + self.robot.coriolis(q, qd) @ qd + self.robot.gravload(q, gravity = self.gravity)
        predFFW = np.matmul(realY, self.pi)

        gtM = self.robot.inertia(q)
        predM = self.dynamicModel.evaluateMatrix(self.dynamicModel.getM(self.robot, self.pi), q, qd, qd, qdd)

        gtS = self.robot.coriolis(q,qd)
        predS = self.dynamicModel.evaluateMatrix(sp.Matrix(self.dynamicModel.getS(self.robot, self.pi)), q, qd, qd, qdd)

        gtG = self.robot.gravload(q, gravity = self.gravity)
        predG = np.array(self.dynamicModel.evaluateMatrix(self.dynamicModel.getG(self.robot, self.pi), q, qd, qd, qdd)).astype(np.float64)

        ffwError = np.linalg.norm(np.array(gtFFW - predFFW, dtype=np.float32))
        mError = np.linalg.norm(np.array(gtM - predM, dtype=np.float32))
        sError = np.linalg.norm(np.array(gtS - predS, dtype=np.float32))
        gError = np.linalg.norm(np.array(gtG - predG, dtype=np.float32))

        self.m_test &= mError < 1e-10
        self.s_test &= sError < 1e-10
        self.g_test &= gError < 1e-10

        self.ffw_test &= ffwError < 1e-10

        self.append(q_d,qd_d,qdd_d,[1,1])

        return [1,1], self.t[-1] > 1

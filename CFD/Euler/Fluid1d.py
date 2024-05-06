import numpy as np
import math
'''
理想气体的一维数据
'''


class Fluid_1d:
    DIM = 1
    U: np.array
    gamma: float
    p: float
    rho: float
    u: float

    def __init__(self, U, gamma=1.4):
        self.U = U
        self.gamma = gamma
        self.update_value()

    def update_value(self):
        self.rho = self.U[0]
        self.u = self.U[1] / self.rho
        self.p = (self.gamma - 1) * (self.U[2] - 0.5 * self.U[1] * self.u)

    def set_U(self, U):
        self.U = U
        self.update_value()

    def set_gamma(self, gamma):
        self.gamma = gamma
        self.update_value()

    def is_fisable(self)->bool:
        if (self.p >= 0 and self.rho >= 0):
            return True
        else:
            return False

    def sound_speed(self)->float:
        return math.sqrt(self.gamma * self.p / self.rho)

    def max_speed(self)->float:
        return abs(self.u) + self.sound_speed()

    # python 中没有 Switch
    def eig_value(self, i)->float:
        self.update_value()
        if i == 1:
            return self.u - self.sound_speed()
        elif i == 2:
            return self.u
        elif i == 3:
            return self.u + self.sound_speed()

    def flux(self)->np.array:
        self.update_value()
        Fu = np.zeros(self.DIM + 2)
        Fu[0] = self.U[1]
        Fu[1] = self.U[1] * self.U[1] / self.U[0] + self.p
        Fu[2] = (self.U[2] + self.p) * self.u
        return Fu

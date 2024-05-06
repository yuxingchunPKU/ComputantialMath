import numpy as np

import Fluid1d as Fd

'''
返回数值通量
'''


class NumericalFlux:
    UL: Fd.Fluid_1d
    UR: Fd.Fluid_1d
    DIM = 1

    def __init__(self, UL, UR):
        self.UL = UL
        self.UR = UR
        self.DIM = self.UL.DIM

    def LLF(self):
        max_speedL=max(self.UL.u-self.UL.sound_speed(),self.UR.u-self.UR.sound_speed())
        max_speedR = max(self.UL.u + self.UL.sound_speed(), self.UR.u + self.UR.sound_speed())
        max_speed = max(max_speedL,max_speedR)
        # max_speed = max([self.UL.max_speed(), self.UR.max_speed()])
        return 0.5 * (self.UL.flux() + self.UR.flux()) - 0.5 * max_speed * (self.UR.U - self.UL.U)

    def guess_speed(self):
        pL = self.UL.p
        pR = self.UR.p
        aL = self.UL.sound_speed()
        aR = self.UR.sound_speed()
        gammaL = self.UL.gamma
        gammaR = self.UR.gamma
        uL = self.UL.u
        uR = self.UR.u
        p1 = 0.5 * (pL + pR) + 0.125 * (uL - uR) * (self.UL.rho + self.UR.rho) * (aL + aR)
        p1 = max(0, p1)
        qL = 1
        qR = 1
        if p1 > pL:
            qL = (1 + (gammaL + 1) * (p1 / pL - 1) / (2 * gammaL)) ** 0.5

        if p1 > pR:
            qR = (1 + (gammaR + 1) * (p1 / pR - 1) / (2 * gammaR)) ** 0.5

        return uL - qL * aL, uR + qR * aR

    def HLL(self):
        '''
        首先估计激波速度
        然后判断返回通量
        :return:
        '''
        [SL,SR]=self.guess_speed()
        if SL>=0:
            return self.UL.flux()
        elif SL <0 and SR >=0:
            return (SR*self.UL.flux()-SL*self.UR.flux()+SL*SR*(self.UR.U-self.UL.U))/(SR-SL)
        else:
            return self.UR.flux()

    def HLLC(self):
        '''
        先估计激波速度 再估计中间状态的速度和压强 最后分类装配通量
        :return:
        '''
        [SL, SR] = self.guess_speed()
        pL = self.UL.p
        pR = self.UR.p
        # aL = self.UL.sound_speed()
        # aR = self.UR.sound_speed()
        uL = self.UL.u
        uR = self.UR.u
        rhoL = self.UL.rho
        rhoR = self.UR.rho
        alphaL = rhoL * (SL - uL)
        alphaR = rhoR * (SR - uR)
        S = (alphaR*uR-alphaL*uL+pL-pR)/(alphaR-alphaL)
        if SL >=0:
            return self.UL.flux()
        elif SL<0 and S>=0:
            #P =(alphaR*pL-alphaL*pR-alphaL*alphaR*(uL-uR))/(alphaR-alphaL)
            '''
            HLLC V1
            '''
            # rhoL1 = alphaL/(SL-S)
            # EL1 = rhoL1*(self.UL.U[2]/rhoL+(S-uL)*(S+pL/alphaL))
            # U1 = np.zeros(3,dtype=float)
            # U1[0] = rhoL1
            # U1[1] = rhoL1*S
            # U1[2] = EL1
            # return self.UL.flux()+SL*(U1-self.UL.U)
            '''
            HLLC V2
            '''
            # D = np.zeros(3,dtype=float)
            # D[1] = 1
            # D[2] = S
            # return (S*(SL*self.UL.U-self.UL.flux())+SL*(pL+alphaL*(S-uL))*D)/(SL-S)
            '''
            HLLC V3
            '''
            D = np.zeros(3,dtype=float)
            D[1] = 1
            D[2] = S
            PLR = 0.5*(pL+pR+alphaL*(S-uL)+alphaR*(S-uR))
            return (S*(SL*self.UL.U-self.UL.flux())+SL*PLR*D)/(SL-S)

        elif S<0 and SR >=0:
            # P =(alphaR*pL-alphaL*pR-alphaL*alphaR*(uL-uR))/(alphaR-alphaL)
            # rhoR1 = alphaR / (SR - S)
            # ER1 = rhoR1*(self.UR.U[2]/rhoR+(S-uR)*(S+pR/alphaR))
            # F = np.zeros(3,dtype=float)
            # F[0] = rhoR1*S
            # F[1] = rhoR1*S*S+P
            # F[2] = (ER1+P)*S
            # D = np.zeros(3,dtype=float)
            # D[1] = 1
            # D[2] = S
            # F = (S*(SR*self.UR.U-self.UR.flux())+S*P*D)/(SR-S)
            # rhoR1 = alphaR/(SR-S)
            # ER1 = rhoR1*(self.UR.U[2]/rhoR+(S-uR)*(S+pR/alphaR))
            # U1 = np.zeros(3,dtype=float)
            # U1[0] = rhoR1
            # U1[1] = rhoR1*S
            # U1[2] = ER1
            # return self.UR.flux()+SR*(U1-self.UR.U)
            '''
            HLLC V2
            '''
            # D = np.zeros(3,dtype=float)
            # D[1] = 1
            # D[2] = S
            # return (S*(SR*self.UR.U-self.UR.flux())+SR*(pR+alphaR*(S-uR))*D)/(SR-S)
            '''
            HLLC V3
            '''
            D = np.zeros(3,dtype=float)
            D[1] = 1
            D[2] = S
            PLR = 0.5*(pL+pR+alphaL*(S-uL)+alphaR*(S-uR))
            return (S*(SR*self.UR.U-self.UR.flux())+SR*PLR*D)/(SR-S)
        else:
            return self.UR.flux()

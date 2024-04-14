'''
激波管问题的完全解
'''
import numpy as np
import Fluid as Fd
class Riemann_Complete_solve:
    p_star : float
    u_star : float
    WL: Fd.Fluid
    WR: Fd.Fluid
    x_L : float
    x_R : float
    N : int
    t_final : float
    def __init__(self,p_star,u_star,WL,WR,t_final,x_L=-0.5,x_R=0.5,N=200):
        self.p_star = p_star
        self.u_star = u_star
        self.WL = WL
        self.WR = WR
        self.t_final =t_final
        self.x_L = x_L
        self.x_R = x_R
        self.N = N

    def solve(self):
        x = np.arange(self.x_L, self.x_R, (self.x_R-self.x_L)/self.N)
        rho = np.zeros(len(x),dtype='float')
        u = np.zeros(len(x), dtype='float')
        p = np.zeros(len(x), dtype='float')
        S = self.u_star #接触间断的速度
        '''
        分别考虑左激波，左稀疏波的情况
        '''
        if self.p_star >= self.WL.p:
            #计算接触间断左侧的密度
            part1 = 2 * self.WL.gamma * self.WL.p_inf + (self.WL.gamma + 1) * self.p_star + (self.WL.gamma - 1) * self.WL.p
            part2 = 2 * (self.WL.p + self.WL.gamma * self.WL.p_inf) + (self.WL.gamma - 1) * self.p_star + (self.WL.gamma - 1) * self.WL.p
            rho_star_L = self.WL.rho * part1 / part2
            A_L = 2 / ((self.WL.gamma + 1) * self.WL.rho)
            #计算激波的速度
            B_L = (self.WL.gamma - 1) * (self.WL.p + self.WL.p_inf) / (self.WL.gamma + 1)
            S_L = self.WL.u - (1 / self.WL.rho) * ((B_L + self.WL.p_inf + self.p_star) / A_L) ** 0.5
            for i in range(len(x)):
                xi = x[i]
                si = xi / self.t_final
                if si <= S_L:
                    rho[i] = self.WL.rho
                    u[i] = self.WL.u
                    p[i] = self.WL.p
                elif si > S_L and si <= S:
                    rho[i] = rho_star_L
                    u[i] = self.u_star
                    p[i] = self.p_star
        else:
            # 计算接触间断左侧密度
            part1 = self.p_star + self.WL.p_inf
            part2 = self.WL.p + self.WL.p_inf
            rho_star_L = self.WL.rho * (part1 / part2) ** (1 / self.WL.gamma)
            # 计算左稀疏波波头和波尾的速度
            aL_star = self.WL.a() * ((self.p_star + self.WL.p_inf) / (self.WL.p + self.WL.p_inf)) ** ((self.WL.gamma - 1) / (2 * self.WL.gamma))
            S_HL = self.WL.u - self.WL.a()
            S_TL = self.u_star - aL_star
            for i in range(len(x)):
                xi = x[i]
                si = xi / self.t_final
                if si <= S_HL:
                    rho[i] = self.WL.rho
                    u[i] = self.WL.u
                    p[i] = self.WL.p
                elif si <= S_TL and si > S_HL:
                    rho[i] = self.WL.rho * (
                                2 / (self.WL.gamma + 1) + (self.WL.gamma - 1) * (self.WL.u - si) / ((self.WL.gamma + 1) * self.WL.a())) ** (
                                         2 / (self.WL.gamma - 1))
                    u[i] = 2 * (self.WL.a() + (self.WL.gamma - 1) * self.WL.u / 2 + si) / (self.WL.gamma + 1)
                    p[i] = (self.WL.p + self.WL.p_inf) * (
                                2 / (self.WL.gamma + 1) + (self.WL.gamma - 1) * (self.WL.u - si) / ((self.WL.gamma + 1) * self.WL.a())) ** (
                                       2 * self.WL.gamma / (self.WL.gamma - 1)) - self.WL.p_inf
                elif si <= S and si > S_TL:
                    rho[i] = rho_star_L
                    u[i] = self.u_star
                    p[i] = self.p_star

        '''
        分别考虑右稀疏波和右激波的情况
        '''
        if self.p_star>=self.WR.p:
            # 计算接触间断右侧密度
            part1 = 2 * self.WR.gamma * self.WR.p_inf + (self.WR.gamma + 1) * self.p_star + (self.WR.gamma - 1) * self.WR.p
            part2 = 2 * (self.WR.p + self.WR.gamma * self.WR.p_inf) + (self.WR.gamma - 1) * self.p_star + (self.WR.gamma - 1) * self.WR.p
            rho_star_R = self.WR.rho * part1 / part2
            # 右激波速度
            A_R = 2 / ((self.WR.gamma + 1) * self.WR.rho)
            B_R = (self.WR.gamma - 1) * (self.WR.p + self.WR.p_inf) / (self.WR.gamma + 1)
            S_R = self.WR.u + (1 / self.WR.rho) * ((B_R + self.WR.p_inf + self.p_star) / A_R) ** 0.5
            for i in range(len(x)):
                xi = x[i]
                si = xi / self.t_final
                if si > S and si <= S_R:
                    rho[i] = rho_star_R
                    u[i] = self.u_star
                    p[i] = self.p_star
                elif si > S_R:
                    rho[i] = self.WR.rho
                    u[i] = self.WR.u
                    p[i] = self.WR.p

        else:
            # 计算接触间断右侧密度
            part1 = self.p_star + self.WR.p_inf
            part2 = self.WR.p + self.WR.p_inf
            rho_star_R = self.WR.rho * (part1 / part2) ** (1 / self.WR.gamma)
            # 右波头和波尾
            aR_star = self.WR.a() * ((self.p_star + self.WR.p_inf) / (self.WR.p + self.WR.p_inf)) ** ((self.WR.gamma - 1) / (2 * self.WR.gamma))
            S_HR = self.WR.u + self.WR.a()
            S_TR = self.u_star + aR_star
            for i in range(len(x)):
                xi = x[i]
                si = xi / self.t_final
                if si<=S_TR and si>S:
                    rho[i] = rho_star_R
                    u[i] = self.u_star
                    p[i] = self.p_star
                elif si<=S_HR and si>S_TR:
                    rho[i] = self.WR.rho*(2/(self.WR.gamma+1)-(self.WR.gamma-1)*(self.WR.u-si)/((self.WR.gamma+1)*self.WR.a()))**(2/(self.WR.gamma-1))
                    u[i] =2*(-self.WR.a()+(self.WR.gamma-1)*self.WR.u/2+si)/(self.WR.gamma+1)
                    p[i] = (self.WR.p+self.WR.p_inf)*(2/(self.WR.gamma+1)-(self.WR.gamma-1)*(self.WR.u-si)/((self.WR.gamma+1)*self.WR.a()))**(2*self.WR.gamma/(self.WR.gamma-1))-self.WR.p_inf
                elif si>S_HR:
                    rho[i] = self.WR.rho
                    u[i] = self.WR.u
                    p[i] = self.WR.p
        return x,rho,u,p
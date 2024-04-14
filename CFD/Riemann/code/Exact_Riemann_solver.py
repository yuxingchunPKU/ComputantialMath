import math
import Fluid as Fd
from typing import *
'''
黎曼问题的精确求解
求解的数值方法为牛顿方法
输入参数为左右两种状态的流体
最大迭代次数 以及精度
'''
class ExactRiemannSolver:
    WL : Fd.Fluid
    WR : Fd.Fluid
    tol : float
    maxit : int
    p_star :float
    u_star :float

    def __init__(self, WL, WR, tol=1e-12,maxit=10000):
        '''
        初始化函数
        :param WL: 激波管左侧的流体 原始变量
        :param WR: 激波管右侧的流体
        :param tol: 计算精度
        :param maxit: 最大迭代次数
        '''
        self.WL = WL
        self.WR = WR
        self.tol =tol
        self.maxit = maxit

    def f_K_p(self,p, W) -> float:
        '''
        计算f_K(p) K=L/R
        :param p: 变量 求解过程中的压强
        :param W: 固定量 要么是WL 要么是WR
        :return: f_K(p) 函数值
        '''
        if p >= W.p:
            A_K = 2 / ((W.gamma + 1) * W.rho)
            B_K = (W.gamma - 1) * (W.p + W.p_inf) / (W.gamma + 1)
            return (p - W.p) * (A_K / (B_K + p + W.p_inf)) ** 0.5
        else:
            return (2 * W.a() / (W.gamma - 1)) * (
                        ((p + W.p_inf) / (W.p + W.p_inf)) ** ((W.gamma - 1) / (2 * W.gamma)) - 1)

    def f_prime_K_p(self, p: float,W: Fd.Fluid)->float:
        '''
        计算f'_K(p) K=L/R 它是 f_K(p) 关于 p 的导数
        :param p: 变量 求解过程中的压强
        :param W:  固定量 要么是WL 要么是WR
        :return:f'_K(p) 函数值
        '''
        if p >= W.p:
            A_K = 2 / ((W.gamma + 1) * W.rho)
            B_K = (W.gamma - 1) * (W.p + W.p_inf) / (W.gamma + 1)
            part1 = ((A_K) / (B_K + p + W.p_inf)) ** 0.5
            part2 = 1 - (p - W.p) / (2 * (B_K + p + W.p_inf))
            return part1 * part2
        else:
            part1 = ((p + W.p_inf) / (W.p + W.p_inf)) ** (-(W.gamma + 1) / (2 * W.gamma))
            part2 = 1 / (W.rho * W.a())
            return part1 * part2

    def p_init(self)->float:
        '''
        计算出用于迭代的压强初始值
        :return:
        '''
        return max(self.tol, 0.5 * (self.WL.p + self.WR.p) - 0.125 * (self.WR.u - self.WL.u) * (self.WR.rho + self.WL.rho) * (self.WR.a() + self.WL.a()))

    def solver(self)->Tuple[float,float]:
        p = self.p_init()
        p_new = 0.0
        for i in range(self.maxit):
            if p < 0.0:
                print("negative pressure")
                break
            f_p = self.f_K_p(p, self.WL) + self.f_K_p(p, self.WR) + self.WR.u - self.WL.u
            df_p = self.f_prime_K_p(p, self.WL) + self.f_prime_K_p(p, self.WR)
            p_new = p - f_p / df_p

            if p_new < 0.0:
                print("negative pressure")
                break
            if 2 * abs(p_new - p) / (p + p_new) < self.tol:
                p = p_new
                break
            else:
                p = p_new

        p_star = p
        u_star = 0.5 * (self.WL.u + self.WR.u) + 0.5 * (self.f_K_p(p_star, self.WR) - self.f_K_p(p_star, self.WL))
        return  p_star,u_star

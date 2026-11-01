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

        采用 PVRS 估计；当刚性气体参数 p_inf 很大导致估计值非正时，
        退回到左右压强的算术平均。
        :return:
        '''
        p_pv = 0.5 * (self.WL.p + self.WR.p) - 0.125 * (self.WR.u - self.WL.u) * \
               (self.WR.rho + self.WL.rho) * (self.WR.a() + self.WL.a())
        if p_pv <= self.tol:
            p_pv = 0.5 * (self.WL.p + self.WR.p)
        return max(self.tol, p_pv)

    def f_p(self, p: float) -> float:
        '''
        黎曼问题的匹配方程
        :param p: 压强
        :return: f(p)
        '''
        return self.f_K_p(p, self.WL) + self.f_K_p(p, self.WR) + self.WR.u - self.WL.u

    def u_star_from_p(self, p_star: float) -> float:
        '''
        由中间压强计算接触间断速度
        :param p_star: 中间压强
        :return: 接触间断速度
        '''
        return 0.5 * (self.WL.u + self.WR.u) + 0.5 * (
                    self.f_K_p(p_star, self.WR) - self.f_K_p(p_star, self.WL))

    def solver(self)->Tuple[float,float]:
        '''
        牛顿迭代求解
        :return: (p_star, u_star)
        '''
        p = self.p_init()
        converged = False
        for _ in range(self.maxit):
            if p <= 0.0:
                break
            f_p = self.f_p(p)
            df_p = self.f_prime_K_p(p, self.WL) + self.f_prime_K_p(p, self.WR)
            p_new = p - f_p / df_p

            if p_new <= 0.0:
                # 阻尼：避免压强变为负值
                p_new = 0.5 * p
            if 2 * abs(p_new - p) / (p + p_new) < self.tol:
                p = p_new
                converged = True
                break
            p = p_new

        if not converged or p <= 0.0:
            # 牛顿迭代失败时退回 Brent 方法
            return self.solver_brent()

        p_star = p
        u_star = self.u_star_from_p(p_star)
        return  p_star,u_star

    def find_interval(self, p0: float):
        '''
        寻找根的存在区间
        :param p0: 初始猜测
        :return: (a,b)，满足 f(a)*f(b)<=0
        '''
        p1 = p0
        if self.f_p(p0) <= 0.0:
            for _ in range(200):
                if self.f_p(p1) > 0.0:
                    return p0, p1
                p1 = 2.0 * p1
        else:
            for _ in range(200):
                if self.f_p(p1) <= 0.0:
                    return p1, p0
                p1 = 0.5 * p1
        raise RuntimeError("无法找到根的存在区间")

    def solver_brent(self) -> Tuple[float, float]:
        '''
        使用 Brent 方法求解匹配方程
        :return: (p_star, u_star)
        '''
        p0 = self.p_init()
        a, b = self.find_interval(p0)
        fa = self.f_p(a)
        fb = self.f_p(b)
        if fa * fb > 0.0:
            raise RuntimeError("Brent 方法失败：区间端点函数值同号")
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c, fc = a, fa
        d = 0.0
        mflag = True
        for _ in range(self.maxit):
            if abs(b - a) / abs(a) < self.tol or fb == 0.0:
                break
            if fa != fc and fb != fc:
                # 逆二次插值
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                    (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                # 割线法
                s = b - fb * (b - a) / (fb - fa)

            conditions = (
                (s < 0.25 * (3 * a + b)) or (s > b),
                mflag and (abs(s - b) >= abs(b - c) / 2),
                (not mflag) and (abs(s - b) >= abs(c - d) / 2),
                mflag and (abs(b - c) < self.tol),
                (not mflag) and (abs(c - d) < self.tol),
            )
            if any(conditions):
                s = 0.5 * (a + b)
                mflag = True
            else:
                mflag = False

            fs = self.f_p(s)
            d, c, fc = c, b, fb
            if fa * fs < 0.0:
                b, fb = s, fs
            else:
                a, fa = s, fs
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa

        p_star = b
        return p_star, self.u_star_from_p(p_star)

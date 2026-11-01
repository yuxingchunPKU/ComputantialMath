'''
考虑含有燃烧波的一维 Euler 方程组的精确 Riemann 求解器
可能涉及到
爆轰波
爆燃波
爆燃-爆轰波的转换
'''
import numpy as np
import math
from typing import *
import FluidReact as Fd

class ExactRiemannSolverDetonation:
    WL : Fd.FluidReact
    WR : Fd.FluidReact
    tol : float
    maxit : int
    p_star :float
    u_star :float
    p_cj : float
    u_cj : float
    rho_cj : float
    quiet : bool
    dq : float

    def __init__(self, WL, WR, tol=1e-08,maxit=10000,quiet=True):
        '''
        初始化函数
        :param WL:
        :param WR:
        :param tol:
        :param maxit:
        :param quiet:
        '''
        self.WL = WL
        self.WR = WR
        self.tol =tol
        self.maxit = maxit
        self.quiet = quiet
        self.dq = WL.q - WR.q
        self.CJ_stauts()
    def f_L_p(self,p:float)-> float:
        '''
        计算f_L(p)
        :param p:
        :return:
        '''
        W = self.WL
        p0 = W.p
        rho0 = W.rho
        tau0 = 1/W.rho
        gamma = W.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        if p >= p0:
            '''
            激波分支
            '''
            A = (1-mu2)*tau0
            B = mu2*p0
            return (p-p0)*(A/(p+B))**0.5
        else:
            '''
            稀疏波分支
            '''
            a0 = W.a()
            idx = (gamma-1)/(2*gamma)
            return 2*a0/(gamma-1)*((p/p0)**idx-1)

    def f_prime_L_p(self,p:float)->float:
        '''
        计算f'_L(p)
        :param p:
        :return:
        '''
        W = self.WL
        p0 = W.p
        rho0 = W.rho
        tau0 = 1/rho0
        gamma = W.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        if p >= p0:
            '''
            激波分支
            '''
            A = (1-mu2)*tau0
            B = (mu2*p0)
            part1 = (A/(p+B))**0.5
            part2 = 1-0.5*(p-p0)/(p+B)
            return part1*part2
        else:
            '''
            稀疏波分支
            '''
            idx = - (gamma+1)/(2*gamma)
            a0 = W.a()
            return (tau0/a0)*((p/p0)**idx)
    def CJ_stauts(self):
        '''
        计算 CJ状态
        :return:
        '''
        WR = self.WR
        dq = self.dq
        p0 = WR.p
        rho0 = WR.rho
        u0 = WR.u
        gamma = WR.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        # 计算 CJ状态
        t1 = 2*gamma*p0/(((gamma**2)-1)*rho0*dq)
        t2 = 1 + (1-t1)**(1/2)
        pcj = p0 - (gamma-1)*rho0*dq*t2
        rhocj = rho0*(pcj*(1+gamma)-p0)/(gamma*pcj)
        tau = 1/rho0
        t3 = ((1-mu2)*tau-2*mu2*dq/(p0-pcj))/(mu2*p0+pcj)
        ucj = u0 + (pcj-p0)*(t3**(1/2))
        self.p_cj = pcj
        self.rho_cj = rhocj
        self.u_cj = ucj

    def f_R_p(self,p:float)->float:
        '''
        计算燃烧波分支 其分支取决于p_cj
        计算f_R(p)
        :param p:
        :return:
        '''
        W = self.WR
        p0 = W.p
        rho0 = W.rho
        tau0 = 1/rho0
        gamma = W.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        tsu_cj = 1/self.rho_cj
        if p >= self.p_cj:
            '''
            爆轰波分支 激波
            '''
            return (p-p0)*(((1-mu2)*tau0-2*mu2*self.dq/(p0-p))/(mu2*p0+p))**(1/2)
        else:
            '''
            CJ爆轰波 由激波和稀疏波组成的复合波
            '''
            part1 = (self.p_cj-p0)*(((1-mu2)*tau0-2*mu2*self.dq/(p0-self.p_cj))/(mu2*p0+self.p_cj))**(1/2)
            part2 = 2*((gamma**0.5)/(gamma-1))*math.sqrt(self.p_cj*tsu_cj)*((p/self.p_cj)**((gamma-1)/(2*gamma))-1)
            return part1+part2

    def f_prime_R_p(self,p:float)->float:
        '''
        求燃烧波分支的导数 f'_R(p)
        :param p:
        :return:
        '''
        W = self.WR
        p0 = W.p
        rho0 = W.rho
        tau0 = 1/rho0
        gamma = W.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        if p>= self.p_cj:
            tau_p = (tau0*(p0+mu2*p)-2*mu2*self.dq)/(p+mu2*p0)
            tau_dp = ((mu2**2-1)*tau0*p0+2*mu2*self.dq)/(p+mu2*p0)**2
            return 0.5*(tau_dp*(p0-p)-(tau_p-tau0))/((tau_p-tau0)*(p0-p))**0.5
        else:
            # 关于稀疏波上的求导
            tau_cj = 1/self.rho_cj
            a_cj = (W.gamma*self.p_cj*tau_cj)**0.5
            return tau_cj*(p/self.p_cj)**(-(gamma+1)/(2*gamma))/a_cj

    def f_p(self,p):
        return self.f_L_p(p) + self.f_R_p(p) + self.WR.u - self.WL.u
    def p_init(self):
        return self.p_cj


    def solver(self):
        p = self.p_init()
        p_new = 0
        for _ in range(self.maxit):
            fp = self.f_p(p)
            dfp = self.f_prime_L_p(p) + self.f_prime_R_p(p)
            p_new = max(p - fp / dfp, self.tol)
            if 2*abs(p_new-p)/(p+p_new)<self.tol:
                p = p_new
                break
            else:
                p = p_new

        self.p_star = p
        self.u_star = 0.5*(self.WL.u+self.WR.u) + 0.5*(self.f_R_p(p)-self.f_L_p(p))
        '''
        输出数据 包括接触间断和燃烧波 波速 和 压强
        '''
        if self.quiet == False:
            print('u_star {:.16f}'.format(self.u_star))
            print('p_star {:.16f}'.format(self.p_star))
            print('rho_cj {:.16f}'.format(self.rho_cj))
            print('u_cj {:.16f}'.format(self.u_cj))
            print('p_cj {:.16f}'.format(self.p_cj))

        return self.p_star,self.u_star
    def find_interval(self,p0:float):
        # 寻找根的存在区间
        p1 = p0
        if self.f_p(p0) <= 0:
            for _ in range(200):
                if self.f_p(p1) > 0:
                    return p0, p1
                p1 = 2*p1
        else:
            for _ in range(200):
                if self.f_p(p1) <= 0:
                    return p1, p0
                p1 = 0.5*p1
        raise RuntimeError("无法找到根的存在区间")
    def solver_brent(self):
        # Brent's method to find a root of the function f_p
        p0 = self.p_init()
        p0,p1 = self.find_interval(p0)
        a = p0
        b = p1
        fa = self.f_p(a)
        fb = self.f_p(b)
        if fa*fb>0:
            raise RuntimeError("Brent 方法失败：区间端点函数值同号")
        if abs(fa)<abs(fb):
            a,b = b,a
            fa,fb = fb,fa
        c = a
        fc = fa
        s = 0
        d = 0
        mflag = True
        iter = 0
        for _ in range(self.maxit):
            if abs(b-a)/abs(a)<self.tol:
                break
            if fb==0:
                break
            if (fa != fc) and (fb != fc):
                # Inverse quadratic interpolation
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                    (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                # Secant method
                s = b - fb * (b - a) / (fb - fa)
            conditions = (
                ((s < 0.25*(3 * a + b)) or (s > b)),
                (mflag and (abs(s - b) >= abs(b - c) / 2)),
                (not mflag and (abs(s - b) >= abs(c - d) / 2)),
                (mflag and (abs(b - c) < self.tol)),
                (not mflag and (abs(c - d) < self.tol))
            )
            if any(conditions):
                s = 0.5*(a+b)
                mflag = True
            else:
                mflag = False

            fs = self.f_p(s)
            d = c
            c = b
            fc = fb
            if (fa*fs)<0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs
            if abs(fa)<abs(fb):
                a,b = b,a
                fa,fb = fb,fa

            iter += 1
        self.p_star = b
        self.u_star = 0.5*(self.WL.u+self.WR.u) + 0.5*(self.f_R_p(b)-self.f_L_p(b))
        '''
        输出数据 包括接触间断和燃烧波 波速 和 压强
        '''
        if self.quiet == False:
            print('u_star {:.16f}'.format(self.u_star))
            print('p_star {:.16f}'.format(self.p_star))
            print('rho_cj {:.16f}'.format(self.rho_cj))
            print('u_cj {:.16f}'.format(self.u_cj))
            print('p_cj {:.16f}'.format(self.p_cj))

        return self.p_star,self.u_star

    def CJ_data(self):
        return self.rho_cj,self.u_cj,self.p_cj

class RiemannCompleteSolDet:
    p_star : float
    u_star : float
    WL : Fd.FluidReact
    WR : Fd.FluidReact
    x_L : float
    x_R : float
    x_c : float
    N : int
    t_final : float
    quite : bool
    def __init__(self,p_star,u_star,WL,WR,t_final,x_L=-0.5,x_R=0.5,x_c =0.0,N=200,quite=True):
        self.p_star = p_star
        self.u_star = u_star
        self.WL = WL
        self.WR = WR
        self.t_final =t_final
        self.x_L = x_L
        self.x_R = x_R
        self.x_c = x_c
        self.N = N
        self.quite = quite

    def complete_solution(self):
        x = np.linspace(self.x_L, self.x_R,self.N+1)
        rho = np.zeros(len(x),dtype='float')
        u = np.zeros(len(x), dtype='float')
        p = np.zeros(len(x), dtype='float')
        S = self.u_star #接触间断的速度
        ERSC = ExactRiemannSolverDetonation(self.WL,self.WR)
        [rho_cj,u_cj,p_cj] = ERSC.CJ_data()
        if self.p_star >= self.WL.p:
            #计算接触间断 左侧的非退化基本波
            gamma = self.WL.gamma
            mu2 = (gamma - 1) / (gamma + 1)
            p0 = self.WL.p
            p1 = self.p_star
            rho0 = self.WL.rho
            u0 = self.WL.u
            rho_star_L = rho0*(p1+mu2*p0)/(p0+mu2*p1)
            S_L = (rho_star_L*self.u_star-rho0*u0)/(rho_star_L-rho0)
            '''
            输出数据
            '''
            if self.quite == False:
                print("Left Shock Wave")
                print('rho_star_L {:.16f}'.format(rho_star_L))
                print('S_L {:.16f}'.format(S_L))

            for i in range(len(x)):
                xi = x[i]
                si = (xi-self.x_c) / self.t_final
                if si <= S_L:
                    rho[i] = rho0
                    u[i] = u0
                    p[i] = p0
                elif si > S_L and si <= S:
                    rho[i] = rho_star_L
                    u[i] = self.u_star
                    p[i] = self.p_star
        else:
            #稀疏波
            gamma = self.WL.gamma
            mu2 = (gamma - 1) / (gamma + 1)
            rho0 = self.WL.rho
            u0 = self.WL.u
            p0 = self.WL.p
            rho_star_L = rho0*(self.p_star/p0)**(1/gamma)
            # 两种声速
            a0 = self.WL.a()
            a_star_L = a0*(self.p_star/p0)**((gamma-1)/(2*gamma))
            S_HL = u0 - a0
            S_TL = self.u_star - a_star_L
            if self.quite == False:
                print("Left Rarefaction Wave")
                print('rho_star_L {:.16f}'.format(rho_star_L))
                print('S_HL {:.16f}'.format(S_HL))
                print('S_TL {:.16f}'.format(S_TL))

            for i in range(len(x)):
                xi = x[i]
                si = (xi-self.x_c) / self.t_final
                if si <= S_HL:
                    rho[i] = self.WL.rho
                    u[i] = self.WL.u
                    p[i] = self.WL.p
                elif si <= S_TL and si > S_HL:
                    rho[i] = rho0*(2/(gamma+1)+(gamma-1)*(u0-si)/((gamma+1)*a0))**(2/(gamma-1))
                    u[i] = 2*(a0+(gamma-1)*u0/2+si)/(gamma+1)
                    p[i] = p0*(2/(gamma+1)+(gamma-1)*(u0-si)/((gamma+1)*a0))**(2*gamma/(gamma-1))
                elif si <= S and si > S_TL:
                    rho[i] = rho_star_L
                    u[i] = self.u_star
                    p[i] = self.p_star

        # 右燃烧波
        if self.p_star >= p_cj:
            #强燃烧波分支
            gamma = self.WR.gamma
            mu2 = (gamma - 1) / (gamma + 1)
            p0 = self.WR.p
            p1 = self.p_star
            rho0 = self.WR.rho
            u0 = self.WR.u
            M = - (p1-p0)/(self.u_star-u0)
            S_R = u0 - M/rho0
            rho_star_R = M/(self.u_star-S_R)
            if self.quite == False:
                print("Right Strong Detonation Wave")
                print('rho_star_R {:.16f}'.format(rho_star_R))
                print('S_Detonation {:.16f}'.format(S_R))
            for i in range(len(x)):
                xi = x[i]
                si = (xi-self.x_c) / self.t_final
                if si >= S_R:
                    rho[i] = self.WR.rho
                    u[i] = self.WR.u
                    p[i] = self.WR.p
                elif si < S_R and si >= S:
                    rho[i] = rho_star_R
                    u[i] = self.u_star
                    p[i] = self.p_star

        else:
            #CJ 燃烧波分支
            gamma = self.WR.gamma
            rho0 = self.WR.rho
            u0 = self.WR.u
            p0 = self.WR.p
            p1 = self.p_star
            rho_star_R = rho_cj*(p1/p_cj)**(1/gamma)
            a_cj = (gamma*p_cj/rho_cj)**0.5
            a_star_R = a_cj*(p1/p_cj)**((gamma-1)/(2*gamma))
            S_TR = self.u_star + a_star_R
            S_HR = u_cj + a_cj
            if self.quite == False:
                print("Right CJ Detonation Wave")
                print('rho_star_R {:.16f}'.format(rho_star_R))
                print('S_TR {:.16f}'.format(S_TR))
                print('S_HR {:.16f}'.format(S_HR))

            for i in range(len(x)):
                xi = x[i]
                si = (xi-self.x_c) / self.t_final
                if si <= S_TR and si > S:
                    rho[i] = rho_star_R
                    u[i] = self.u_star
                    p[i] = self.p_star
                elif si <= S_HR and si > S_TR:
                    rho[i] = rho_cj*(2/(gamma+1)-(gamma-1)*(u_cj-si)/((gamma+1)*a_cj))**(2/(gamma-1))
                    u[i] =2*(-a_cj+(gamma-1)*u_cj/2+si)/(gamma+1)
                    p[i] = p_cj*(2/(gamma+1)-(gamma-1)*(u_cj-si)/((gamma+1)*a_cj))**(2*gamma/(gamma-1))
                elif si>S_HR:
                    rho[i] = rho0
                    u[i] = u0
                    p[i] = p0

        return x,rho,u,p


class ExactRiemannSolverDeflagration:
    WL : Fd.FluidReact
    WR : Fd.FluidReact
    tol : float
    maxit : int
    p_star :float
    u_star :float
    p0 : float
    rho0 : float
    p_cj : float
    u_cj : float
    rho_cj : float
    quiet : bool
    dq : float
    V : float # 燃烧波的速度
    K : float # 燃烧波的参数1
    Q : float # 燃烧波的参数2 指数

    def __init__(self, WL, WR, K,Q=2, tol=1e-08,maxit=10000,quiet=True):
        '''
        初始化函数
        :param WL:
        :param WR:
        :param tol:
        :param maxit:
        :param quiet:
        '''
        self.WL = WL
        self.WR = WR
        self.K = K
        self.Q = Q
        self.tol =tol
        self.maxit = maxit
        self.quiet = quiet
        self.dq = WL.q - WR.q

    def CJ_stauts(self,W0):
        # 计算CJ状态
        dq = self.dq
        p0 = W0.p
        rho0 = W0.rho
        u0 = W0.u
        gamma = W0.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        # 计算 CJ状态
        t1 = 2*gamma*p0/(((gamma**2)-1)*rho0*dq)
        t2 = 1 - (1-t1)**(1/2)
        pcj = p0 - (gamma-1)*rho0*dq*t2
        rhocj = rho0*(pcj*(1+gamma)-p0)/(gamma*pcj)
        tau = 1/rho0
        t3 = ((1-mu2)*tau-2*mu2*dq/(p0-pcj))/(mu2*p0+pcj)
        ucj = u0 + (pcj-p0)*(t3**(1/2))
        self.p_cj = pcj
        self.rho_cj = rhocj
        self.u_cj = ucj
    def f_L_p(self,p:float)-> float:
        '''
        计算f_L(p)
        :param p:
        :return:
        '''
        W = self.WL
        p0 = W.p
        rho0 = W.rho
        tau0 = 1/rho0
        gamma = W.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        if p >= p0:
            '''
            激波分支
            '''
            A = (1-mu2)*tau0
            B = (mu2*p0)
            return (p-p0)*(A/(p+B))**0.5
        else:
            '''
            稀疏波分支
            '''
            a0 = W.a()
            idx = (gamma-1)/(2*gamma)
            return 2*a0/(gamma-1)*((p/p0)**idx-1)
    def deflagration_speed(self,rho0,u0,p0)->float:
        '''
        计算爆燃波的速度
        :param rho0:
        :param u0:
        :param p0:
        :return:
        '''
        K = self.K
        Q = self.Q
        return u0 + K*(p0/rho0)**Q
    def state0_from_p0(self,p0:float):
        '''
        由波前（未燃气体）压强 p0 计算火焰前方状态 0 的密度和速度

        状态 0 与右端未燃状态 WR 之间是一道非反应波（激波或稀疏波）。
        :param p0: 火焰前方压强（求解变量）
        :return: (rho0, u0)
        '''
        WR = self.WR
        rhoR = WR.rho
        uR = WR.u
        pR = WR.p
        gamma = WR.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        if p0 >= pR:
            # 激波分支
            rho0 = rhoR*(p0+mu2*pR)/(mu2*p0+pR)
            phi = (p0-pR)*((1-mu2)/(rhoR*(p0+mu2*pR)))**0.5
        else:
            # 稀疏波分支
            rho0 = rhoR*(p0/pR)**(1/gamma)
            phi = 2*WR.a()/(gamma-1)*((p0/pR)**((gamma-1)/(2*gamma))-1)
        return rho0, uR + phi

    def state1_from_p0(self,p0:float):
        '''
        由火焰前方压强 p0 计算火焰波后的状态 1

        依次使用：
            1. 右端非反应波：WR -> 状态 0
            2. 爆燃波速度关系：V = u0 + K (p0/rho0)^Q
            3. Rayleigh-Hugoniot 关系：状态 0 -> 状态 1
        :param p0: 火焰前方压强
        :return: (rho0, u0, V, p1, u1)
        '''
        rho0, u0 = self.state0_from_p0(p0)
        V = self.deflagration_speed(rho0, u0, p0)
        p1 = self.compute_p1(rho0, u0, p0)
        tau0 = 1/rho0
        mu2 = (self.WR.gamma - 1) / (self.WR.gamma + 1)
        phi_def = (p1-p0)*(((1-mu2)*tau0-2*mu2*self.dq/(p0-p1))/(mu2*p0+p1))**0.5
        return rho0, u0, V, p1, u0 + phi_def

    def f_R_p(self,p:float)->float:
        '''
        计算右波（非反应波 + 爆燃波）两侧的速度差 u1 - uR
        :param p: 火焰前方压强 p0
        :return: u1 - uR
        '''
        _, _, _, _, u1 = self.state1_from_p0(p)
        return u1 - self.WR.u

    def compute_p1(self,rho0,u0,p0):
        gamma = self.WR.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        V = self.deflagration_speed(rho0, u0, p0)
        tau0 = 1 / rho0
        c0 = (gamma * p0 * tau0) ** 0.5
        v0 = u0 - V
        M2 = (v0/c0)**2
        p_ratio1 = (1-mu2)*(1+gamma*M2)
        p_ratio2 = math.sqrt(((1+mu2)**2)*((1-M2)**2)+8*mu2*(gamma**2)*M2*self.dq/(c0**2))
        p1 = p0*(p_ratio1+p_ratio2)/2
        return p1

    def f_p(self,p):
        '''
        匹配方程

        求解变量 p 是火焰前方压强 p0。左波匹配的是火焰波后压强 p1，
        因此左波特征函数必须在 p1 处取值。
        '''
        _, _, _, p1, _ = self.state1_from_p0(p)
        return self.f_L_p(p1) + self.f_R_p(p) + self.WR.u - self.WL.u

    def p_init(self):
        return self.WR.p

    def find_interval(self,p0:float):
        # 寻找根的存在区间
        p1 = p0
        if self.f_p(p0) <= 0:
            for _ in range(200):
                if self.f_p(p1) > 0:
                    return p0, p1
                p1 = 2*p1
        else:
            for _ in range(200):
                if self.f_p(p1) <= 0:
                    return p1, p0
                p1 = 0.5*p1
        raise RuntimeError("无法找到根的存在区间")

    def solver_brent(self):
        p0 = self.p_init()
        p0,p1 = self.find_interval(p0)
        a = p0
        b = p1
        fa = self.f_p(a)
        fb = self.f_p(b)
        if fa*fb>0:
            raise RuntimeError("Brent 方法失败：区间端点函数值同号")
        if abs(fa)<abs(fb):
            a,b = b,a
            fa,fb = fb,fa
        c = a
        fc = fa
        s = 0
        d = 0
        mflag = True
        iter = 0
        for _ in range(self.maxit):
            if abs(b-a)/abs(a)<self.tol:
                break
            if fb==0:
                break
            if (fa != fc) and (fb != fc):
                # Inverse quadratic interpolation
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                    (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                # Secant method
                s = b - fb * (b - a) / (fb - fa)
            conditions = (
                ((s < 0.25*(3 * a + b)) or (s > b)),
                (mflag and (abs(s - b) >= abs(b - c) / 2)),
                (not mflag and (abs(s - b) >= abs(c - d) / 2)),
                (mflag and (abs(b - c) < self.tol)),
                (not mflag and (abs(c - d) < self.tol))
            )
            if any(conditions):
                s = 0.5*(a+b)
                mflag = True
            else:
                mflag = False

            fs = self.f_p(s)
            d = c
            c = b
            fc = fb
            if (fa*fs)<0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs
            if abs(fa)<abs(fb):
                a,b = b,a
                fa,fb = fb,fa

            iter += 1
        self.p0 = b
        rho0, u0, V, p1, u1 = self.state1_from_p0(b)
        self.rho0 = rho0
        self.u0 = u0
        self.V = V
        self.p_star = p1
        self.u_star = u1
        # 由质量守恒得到火焰波后（中间状态）的密度
        self.rho1 = rho0*(V-u0)/(V-u1)
        if self.quiet == False:
            print('p_star {:.16f}'.format(self.p_star))
            print('u_star {:.16f}'.format(self.u_star))
            print('p0 {:.16f}'.format(self.p0))
            print('rho0 {:.16f}'.format(self.rho0))
            print('rho1 {:.16f}'.format(self.rho1))
            print('V {:.16f}'.format(self.V))
        # 输出数据
        return self.p_star,self.u_star,self.p0,self.rho0,self.V

    def solver_data(self):
        '''
        :return: 求解得到的波和状态数据字典
        '''
        return {
            'p_star': self.p_star,
            'u_star': self.u_star,
            'p0': self.p0,
            'rho0': self.rho0,
            'u0': self.u0,
            'rho1': self.rho1,
            'V': self.V,
        }

# 爆燃问题的完整解
class RiemannCompleteSolDef:
    '''
    爆燃波黎曼问题的完整解

    从左到右的波系为
        左端状态 WL -> 非反应波 -> 接触间断 -> 爆燃波 -> 状态 0 -> 非反应波 -> 右端状态 WR
    其中求解器给出的是状态 1（火焰波后、接触间断一侧）与状态 0（火焰波前）。
    '''
    p_star : float
    u_star : float
    p0 : float
    rho0 : float
    WL : Fd.FluidReact
    WR : Fd.FluidReact
    x_L : float
    x_R : float
    x_c : float
    N : int
    t_final : float
    quite : bool
    V : float
    rho1 : float
    u0 : float

    def __init__(self,p_star,u_star,p0,rho0,WL,WR,t_final,x_L=-0.5,x_R=0.5,x_c=0.0,N=200,quite=True,V=None,rho1=None,u0=None):
        '''
        :param p_star: 接触间断/火焰波后压强 p1
        :param u_star: 接触间断速度 u1
        :param p0: 火焰波前压强
        :param rho0: 火焰波前密度
        :param WL: 左端流体
        :param WR: 右端流体
        :param t_final: 终止时刻
        :param x_L: 左边界
        :param x_R: 右边界
        :param x_c: 初始间断位置
        :param N: 网格数
        :param quite: 是否静默
        :param V: 火焰波速度
        :param rho1: 火焰波后密度
        :param u0: 火焰波前速度
        '''
        self.p_star = p_star
        self.u_star = u_star
        self.p0 = p0
        self.rho0 = rho0
        self.WL = WL
        self.WR = WR
        self.t_final = t_final
        self.x_L = x_L
        self.x_R = x_R
        self.x_c = x_c
        self.N = N
        self.quite = quite
        self.V = V
        self.rho1 = rho1
        self.u0 = u0

    def _state0_velocity(self):
        '''由火焰波前压强 p0 和右端状态 WR 计算 u0'''
        WR = self.WR
        mu2 = (WR.gamma - 1) / (WR.gamma + 1)
        if self.p0 >= WR.p:
            phi = (self.p0 - WR.p) * ((1 - mu2) / (WR.rho * (self.p0 + mu2 * WR.p))) ** 0.5
        else:
            phi = 2 * WR.a() / (WR.gamma - 1) * (
                        (self.p0 / WR.p) ** ((WR.gamma - 1) / (2 * WR.gamma)) - 1)
        return WR.u + phi

    def complete_solution(self):
        if self.V is None:
            raise ValueError("RiemannCompleteSolDef 需要爆燃波速度 V")
        x = np.linspace(self.x_L, self.x_R, self.N + 1)
        rho = np.full(len(x), self.WL.rho, dtype='float')
        u = np.full(len(x), self.WL.u, dtype='float')
        p = np.full(len(x), self.WL.p, dtype='float')
        si = (x - self.x_c) / self.t_final
        S = self.u_star  # 接触间断速度
        p1 = self.p_star
        u0 = self._state0_velocity() if self.u0 is None else self.u0
        rho1 = self.rho0 * (self.V - u0) / (self.V - self.u_star) if self.rho1 is None else self.rho1
        if not (self.V > S):
            raise ValueError("爆燃波速度 V 必须大于接触间断速度 u*，否则波系顺序不成立")

        WL = self.WL
        gammaL = WL.gamma
        mu2L = (gammaL - 1) / (gammaL + 1)

        # ---------------- 左侧非反应波 ----------------
        if p1 >= WL.p:
            # 激波
            rho_star_L = WL.rho * (p1 + mu2L * WL.p) / (WL.p + mu2L * p1)
            if abs(rho_star_L - WL.rho) > 1.0e-14 * WL.rho:
                S_L = (rho_star_L * self.u_star - WL.rho * WL.u) / (rho_star_L - WL.rho)
            else:
                # 零强度波
                S_L = WL.u - WL.a()
            mask = (si > S_L) & (si <= S)
            rho[mask] = rho_star_L
            u[mask] = self.u_star
            p[mask] = p1
            if self.quite == False:
                print("Left Shock Wave")
                print('rho_star_L {:.16f}'.format(rho_star_L))
                print('S_L {:.16f}'.format(S_L))
        else:
            # 稀疏波
            aL = WL.a()
            rho_star_L = WL.rho * (p1 / WL.p) ** (1 / gammaL)
            a_star_L = aL * (p1 / WL.p) ** ((gammaL - 1) / (2 * gammaL))
            S_HL = WL.u - aL
            S_TL = self.u_star - a_star_L
            fan = (si > S_HL) & (si <= S_TL)
            rho[fan] = WL.rho * (2 / (gammaL + 1) + (gammaL - 1) * (WL.u - si[fan]) /
                                 ((gammaL + 1) * aL)) ** (2 / (gammaL - 1))
            u[fan] = 2 * (aL + (gammaL - 1) * WL.u / 2 + si[fan]) / (gammaL + 1)
            p[fan] = WL.p * (2 / (gammaL + 1) + (gammaL - 1) * (WL.u - si[fan]) /
                             ((gammaL + 1) * aL)) ** (2 * gammaL / (gammaL - 1))
            mask = (si > S_TL) & (si <= S)
            rho[mask] = rho_star_L
            u[mask] = self.u_star
            p[mask] = p1
            if self.quite == False:
                print("Left Rarefaction Wave")
                print('rho_star_L {:.16f}'.format(rho_star_L))
                print('S_HL {:.16f}'.format(S_HL))
                print('S_TL {:.16f}'.format(S_TL))

        # ---------------- 接触间断与火焰之间的状态 1 ----------------
        mask = (si > S) & (si < self.V)
        rho[mask] = rho1
        u[mask] = self.u_star
        p[mask] = p1

        # ---------------- 火焰波后的状态 0 与右侧非反应波 ----------------
        WR = self.WR
        gammaR = WR.gamma
        aR = WR.a()
        if self.p0 >= WR.p:
            # 激波
            tau_R = 1 / WR.rho
            tau_0 = 1 / self.rho0
            if abs(tau_R - tau_0) > 1.0e-14 * tau_R:
                S_R0 = WR.u + ((self.p0 - WR.p) / (tau_R - tau_0)) ** 0.5 / WR.rho
            else:
                S_R0 = WR.u + WR.a()
            if not (S_R0 > self.V):
                raise ValueError("右侧非反应波速度必须大于爆燃波速度 V，否则波系顺序不成立")
            mask = (si >= self.V) & (si < S_R0)
            rho[mask] = self.rho0
            u[mask] = u0
            p[mask] = self.p0
            rho[si >= S_R0] = WR.rho
            u[si >= S_R0] = WR.u
            p[si >= S_R0] = WR.p
            if self.quite == False:
                print("Right Non-reactive Shock Wave")
                print('rho0 {:.16f}'.format(self.rho0))
                print('S_R0 {:.16f}'.format(S_R0))
        else:
            # 稀疏波
            a0 = (gammaR * self.p0 / self.rho0) ** 0.5
            S_TR0 = u0 + a0
            S_HR = WR.u + aR
            if not (S_TR0 > self.V):
                raise ValueError("右侧非反应波速度必须大于爆燃波速度 V，否则波系顺序不成立")
            mask = (si >= self.V) & (si < S_TR0)
            rho[mask] = self.rho0
            u[mask] = u0
            p[mask] = self.p0
            fan = (si >= S_TR0) & (si < S_HR)
            rho[fan] = WR.rho * (2 / (gammaR + 1) - (gammaR - 1) * (WR.u - si[fan]) /
                                 ((gammaR + 1) * aR)) ** (2 / (gammaR - 1))
            u[fan] = 2 * (-aR + (gammaR - 1) * WR.u / 2 + si[fan]) / (gammaR + 1)
            p[fan] = WR.p * (2 / (gammaR + 1) - (gammaR - 1) * (WR.u - si[fan]) /
                             ((gammaR + 1) * aR)) ** (2 * gammaR / (gammaR - 1))
            rho[si >= S_HR] = WR.rho
            u[si >= S_HR] = WR.u
            p[si >= S_HR] = WR.p
            if self.quite == False:
                print("Right Non-reactive Rarefaction Wave")
                print('rho0 {:.16f}'.format(self.rho0))
                print('S_TR0 {:.16f}'.format(S_TR0))
                print('S_HR {:.16f}'.format(S_HR))

        if self.quite == False:
            print("Deflagration Wave")
            print('rho1 {:.16f}'.format(rho1))
            print('V {:.16f}'.format(self.V))
        return x, rho, u, p




# 向后兼容的别名
RiemannCompleteSolution = RiemannCompleteSolDet

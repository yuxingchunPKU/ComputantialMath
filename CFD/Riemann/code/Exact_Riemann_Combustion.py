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
        for i in range(self.maxit):
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
        iter = 0
        if self.f_p(p0)<=0:
            while self.f_p(p1)<=0:
                p1 = 2*p1
                iter += 1
            return p0,p1
        else:
            while self.f_p(p1)>0:
                p1 = 0.5*p1
                iter += 1
            return p1,p0
    def solver_brent(self):
        # Brent's method to find a root of the function f_p
        p0 = self.p_init()
        p0,p1 = self.find_interval(p0)
        a = p0
        b = p1
        fa = self.f_p(a)
        fb = self.f_p(b)
        if fa*fb>0:
            print("f(a) and f(b) have the same sign")
            exit()
        if abs(fa)<abs(fb):
            a,b = b,a
            fa,fb = fb,fa
        c = a
        fc = fa
        s = 0
        d = 0
        mflag = True
        iter = 0
        for i in range(self.maxit):
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
    def f_R_p(self,p:float)->float:
        '''
        推导爆燃波到右状态的速度差值
        :return:
        '''
        p0 = p
        rho0 = 0.0
        phi = 0.0
        WR = self.WR
        rhoR = WR.rho
        uR = WR.u
        pR = WR.p
        gamma = WR.gamma
        mu2 = (gamma - 1) / (gamma + 1)
        if p0 >= WR.p:
            # 激波分支
            rho0 += rhoR*(p0+mu2*pR)/(mu2*p0+pR)
            phi += ((p0-pR)*((1-mu2)/(rhoR*(p0+mu2*pR)))**0.5)
        else:
            # 稀疏波分支
            idx = 1/gamma
            rho0 += rhoR*(p0/pR)**idx
            phi += (2*WR.a()/(gamma-1)*((p0/pR)**((gamma-1)/(2*gamma))-1))

        u0 = phi + uR
        tau0 = 1/rho0
        p1 = self.compute_p1(rho0,u0,p0)
        phi_def = (p1-p0)*(((1-mu2)*tau0-2*mu2*self.dq/(p0-p1))/(mu2*p0+p1))**0.5
        # 返回速度差
        return phi+phi_def

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
        return self.f_L_p(p) + self.f_R_p(p) + self.WR.u - self.WL.u

    def p_init(self):
        return self.WR.p

    def find_interval(self,p0:float):
        # 寻找根的存在区间
        p1 = p0
        iter = 0
        if self.f_p(p0)<=0:
            while self.f_p(p1)<=0:
                p1 = 2*p1
                iter += 1
            return p0,p1
        else:
            while self.f_p(p1)>0:
                p1 = 0.5*p1
                iter += 1
            return p1,p0

    def solver_brent(self):
        p0 = self.p_init()
        p0,p1 = self.find_interval(p0)
        a = p0
        b = p1
        fa = self.f_p(a)
        fb = self.f_p(b)
        if fa*fb>0:
            print("f(a) and f(b) have the same sign")
            exit()
        if abs(fa)<abs(fb):
            a,b = b,a
            fa,fb = fb,fa
        c = a
        fc = fa
        s = 0
        d = 0
        mflag = True
        iter = 0
        for i in range(self.maxit):
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
        self.u_star = 0.5*(self.WL.u+self.WR.u) + 0.5*(self.f_R_p(b)-self.f_L_p(b))
        rho0 = 0.0
        u0 = 0.0
        mu2 = (self.WR.gamma - 1) / (self.WR.gamma + 1)
        if p0 >= self.WR.p:
            rho0 += self.WR.rho*(p0+mu2*self.WR.p)/(mu2*p0+self.WR.p)
            u0 += self.WR.u + ((p0-self.WR.p)*((1-mu2)/(self.WR.rho*(p0+mu2*self.WR.p)))**0.5)
        else:
            idx = 1/self.WR.gamma
            rho0 += self.WR.rho*(p0/self.WR.p)**idx
            u0 += 2*self.WR.a()/(self.WR.gamma-1)*((p0/self.WR.p)**((self.WR.gamma-1)/(2*self.WR.gamma))-1)
        self.rho0 = rho0
        self.V = self.deflagration_speed(rho0,u0,p0)
        self.p_star = self.compute_p1(rho0,u0,p0)
        # 输出四组数据
        return self.p_star,self.u_star,self.p0,self.rho0,self.V

# 爆燃问题的完整解
class RiemannCompleteSolDef:
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
    def __init__(self,p_star,u_star,p0,rho0,WL,WR,t_final,x_L=-0.5,x_R=0.5,x_c =0.0,N=200,quite=True):
        self.p_star = p_star
        self.u_star = u_star
        self.p0 = p0
        self.rho0 = rho0
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
        S = self.u_star



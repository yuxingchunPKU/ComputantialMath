import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import FluidReact as FR
from typing import *

rhoL = 0.142168
uL = -181.018
pL = 9.45695e4
qL = 0.0
gammaL = 1.4
rhoR = 1.0
uR = 0.0
pR = 1.0e5
qR = 2.0e6
gammaR = 1.4

# rhoL = 0.974514
# uL = 1586.05
# pL = 4.04266e6
# qL = 0.0
# gammaL = 1.27
#
# rhoR = 0.601
# uR = 0.0
# pR = 1.0e5
# qR = 242000/0.018
# gammaR = 1.27

WL = FR.FluidReact(rhoL,uL,pL,gammaL,qL)
WR = FR.FluidReact(rhoR,uR,pR,gammaR,qR)
t_end = 0.001
N = 1000
x_c = 0.8
x_L = 0.0
x_R = 1.6
V = 0.0
p1 = 0.0
# 燃烧速度函数
def deflagration_speed(rho,u,p):
    # u 是投影速度
    return u + (3e-9)*(p/rho)**2
# 特征函数
def f_L_p(p,WL):
    A_L = 2 / ((WL.gamma + 1) * WL.rho)
    B_L = (WL.gamma - 1) * (WL.p) / (WL.gamma + 1)
    if p >= WL.p:
        return (p - WL.p) * (A_L / (B_L + p)) ** 0.5
    else:
        return (2 * WL.a() / (WL.gamma - 1)) * ((p/ WL.p ) ** ((WL.gamma - 1) / (2 * WL.gamma)) - 1)


def f_R_p_def(p,WR,dq):
    '''
    推导爆燃波前和波后的速度差值
    :param p: 压强
    :param WR: 右侧流体
    :return:
    '''
    p0 = p
    rho0 = 0.0
    phi = 0.0
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
    V = deflagration_speed(rho0,u0,p0)
    tau0 = 1/rho0
    c0 = (gamma*p0*tau0)**0.5
    v0 = u0 - V
    M2 = (v0/c0)**2
    p_ratio1 = (1-mu2)*(1+gamma*M2)
    p_ratio2 = math.sqrt(((1+mu2)**2)*((1-M2)**2)+8*mu2*(gamma**2)*M2*dq/(c0**2))
    p1 = p0*(p_ratio1+p_ratio2)/2
    # 返回速度差
    phi_de = (p1-p0)*(((1-mu2)*tau0-2*mu2*dq/(p0-p1))/(mu2*p0+p1))**0.5
    return phi+phi_de


def Fp(p,WL,WR):
    delta = WL.q - WR.q
    return f_L_p(p,WL) + f_R_p_def(p,WR,delta)+WR.u-WL.u

def find_interval(p0,WL,WR):
    # 寻找根的存在区间
    p1 = p0
    iter = 0
    if Fp(p0,WL,WR)<=0:
        while Fp(p1,WL,WR)<=0:
            p1 = 2*p1
            iter += 1
        print("find the interval "+ str(iter))
        return p0,p1
    else:
        while Fp(p1,WL,WR)>0:
            p1 = 0.5*p1
            iter += 1
        print("find the interval "+ str(iter))
        return p1,p0


# 定义一个求解器 brent's method
def solver_brent(p0,WL,WR):
    # 确定区间
    max_iter = 10000
    tol = 1e-12
    p0,p1 = find_interval(p0,WL,WR)
    a = p0
    b = p1
    fa = Fp(a,WL,WR)
    fb = Fp(b,WL,WR)
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
    for i in range(max_iter):
        if abs(b-a)/abs(a)<tol:
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
            (mflag and (abs(b - c) < tol)),
            (not mflag and (abs(c - d) < tol))
        )
        if any(conditions):
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False
        fs = Fp(s,WL,WR)
        d = c
        c = b
        fc = fb
        if (fa * fs) < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        iter += 1
    print(iter)
    return b

dq = WL.q - WR.q
p0 = pR
p = solver_brent(p0,WL,WR)
print(p)
print(Fp(p,WL,WR))
rho0 = 0.0
phi = 0.0
if p >= pR:
    # 激波
    mu2 = (gammaR - 1) / (gammaR + 1)
    rho0 += rhoR * (p + mu2 * pR) / (mu2 * p + pR)
    phi += ((p - pR) * ((1 - mu2) / (rhoR * (p + mu2 * pR))) ** 0.5)
else:
    idx = 1 / gammaR
    rho0 += rhoR * (p / pR) ** idx
    phi += (2 * WR.a() / (gammaR - 1) * ((p / pR) ** ((gammaR - 1) / (2 * gammaR)) - 1))

u0 = phi + uR
p0 = p
print("---")
print(deflagration_speed(rho0,u0,p0))
print(rho0,u0,p0)



# p_min = 0.5e5
# p_max = 5e5
# p_mid = 0.5*(p_min+p_max)
#
# Res = Fp(p_mid,WL,WR)
# while abs(Res)>1e-8:
#     Res=Fp(p_mid,WL,WR)
#     if Res>0:
#         p_max = p_mid
#         p_mid = 0.5*(p_min+p_max)
#     else:
#         p_min = p_mid
#         p_mid = 0.5*(p_min+p_max)
# #
# u_mid = WL.u - f_L_p(p_mid,WL)
#
# # x = np.linspace(x_L,x_R,N)
# p_star = p_mid
# u_star = u_mid
#
#
# V = deflagration_speed(WR)
# p1 = p_star
# u1 = u_star
# v0 = WR.u - V
# c0 = WR.a()
# M2 = (v0 / c0) ** 2
# gamma = WR.gamma
# mu2 = (gamma - 1) / (gamma + 1)
# p_ratio1 = (1 - mu2) * (1 + gamma * M2)
# p_ratio2 = math.sqrt(((1 + mu2) ** 2) * ((1 - M2) ** 2) + 8 * mu2 * (gamma ** 2) * M2 * dq / (c0 ** 2))
# p0 = 2*p1/(p_ratio1+p_ratio2)
# # 压强 p
# print("p")
# print(pL)
# print(p_star)
# print(p1)
# print(p0)
# print(pR)
# # 左侧的激波 或者 稀疏波
# S = u_star
# rho = np.zeros(N)
# u = np.zeros(N)
# p = np.zeros(N)
#
#
# if p_star > pL:
#     # 激波
#     mu2 = (gammaL - 1) / (gammaL + 1)
#     p_r = p_star/pL
#     rhoL_star = WL.rho * (p_r + mu2) / (mu2 * p_r + 1)
#     SL = WL.u - WL.a() * ((gammaL+1)*p_star/(2*gammaL*pL) + (gammaL-1)/(2*gammaL))**0.5
#     for i in range(N):
#         if x[i]-x_c < SL*t_end:
#             rho[i] = WL.rho
#             u[i] = WL.u
#             p[i] = WL.p
#         elif x[i]-x_c < S*t_end:
#             rho[i] = rhoL_star
#             u[i] = u_star
#             p[i] = p_star
# # 右侧是一个爆轰激波
# M = - (p_star-pR)/(u_star-uR)
# SR = uR - M/rhoR
# rhoR_star = M/(u_star-SR)
# for i in range(N):
#     if (x[i]-x_c) > SR*t_end:
#         rho[i] = rhoR
#         u[i] = uR
#         p[i] = pR
#     elif ((x[i]-x_c) >= S*t_end) and ((x[i]-x_c) <= SR*t_end):
#         rho[i] = rhoR_star
#         u[i] = u_star
#         p[i] = p_star
# 右侧的解 这是一个激波

import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import FluidReact as FR
from typing import *

rhoL = 0.142168
uL = -181.018
pL = 9.45695e4
qL = 0.0
gammaL = 1.4
rhoR = 1.0
uR = 0.0
pR = 1.0e5
qR = 2.0e6
gammaR = 1.4

# rhoL = 0.974514
# uL = 1586.05
# pL = 4.04266e6
# qL = 0.0
# gammaL = 1.27
#
# rhoR = 0.601
# uR = 0.0
# pR = 1.0e5
# qR = 242000/0.018
# gammaR = 1.27

WL = FR.FluidReact(rhoL,uL,pL,gammaL,qL)
WR = FR.FluidReact(rhoR,uR,pR,gammaR,qR)
t_end = 0.001
N = 1000
x_c = 0.8
x_L = 0.0
x_R = 1.6
V = 0.0
p1 = 0.0
# 燃烧速度函数
def deflagration_speed(rho,u,p):
    # u 是投影速度
    return u + (3e-9)*(p/rho)**2
# 特征函数
def f_L_p(p,WL):
    A_L = 2 / ((WL.gamma + 1) * WL.rho)
    B_L = (WL.gamma - 1) * (WL.p) / (WL.gamma + 1)
    if p >= WL.p:
        return (p - WL.p) * (A_L / (B_L + p)) ** 0.5
    else:
        return (2 * WL.a() / (WL.gamma - 1)) * ((p/ WL.p ) ** ((WL.gamma - 1) / (2 * WL.gamma)) - 1)


def f_R_p_def(p,WR,dq):
    '''
    推导爆燃波前和波后的速度差值
    :param p: 压强
    :param WR: 右侧流体
    :return:
    '''
    p0 = p
    rho0 = 0.0
    phi = 0.0
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
    V = deflagration_speed(rho0,u0,p0)
    tau0 = 1/rho0
    c0 = (gamma*p0*tau0)**0.5
    v0 = u0 - V
    M2 = (v0/c0)**2
    p_ratio1 = (1-mu2)*(1+gamma*M2)
    p_ratio2 = math.sqrt(((1+mu2)**2)*((1-M2)**2)+8*mu2*(gamma**2)*M2*dq/(c0**2))
    p1 = p0*(p_ratio1+p_ratio2)/2
    # 返回速度差
    phi_de = (p1-p0)*(((1-mu2)*tau0-2*mu2*dq/(p0-p1))/(mu2*p0+p1))**0.5
    return phi+phi_de


def Fp(p,WL,WR):
    delta = WL.q - WR.q
    return f_L_p(p,WL) + f_R_p_def(p,WR,delta)+WR.u-WL.u

def find_interval(p0,WL,WR):
    # 寻找根的存在区间
    p1 = p0
    iter = 0
    if Fp(p0,WL,WR)<=0:
        while Fp(p1,WL,WR)<=0:
            p1 = 2*p1
            iter += 1
        print("find the interval "+ str(iter))
        return p0,p1
    else:
        while Fp(p1,WL,WR)>0:
            p1 = 0.5*p1
            iter += 1
        print("find the interval "+ str(iter))
        return p1,p0


# 定义一个求解器 brent's method
def solver_brent(p0,WL,WR):
    # 确定区间
    max_iter = 10000
    tol = 1e-12
    p0,p1 = find_interval(p0,WL,WR)
    a = p0
    b = p1
    fa = Fp(a,WL,WR)
    fb = Fp(b,WL,WR)
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
    for i in range(max_iter):
        if abs(b-a)/abs(a)<tol:
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
            (mflag and (abs(b - c) < tol)),
            (not mflag and (abs(c - d) < tol))
        )
        if any(conditions):
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False
        fs = Fp(s,WL,WR)
        d = c
        c = b
        fc = fb
        if (fa * fs) < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        iter += 1
    print(iter)
    return b

dq = WL.q - WR.q
p0 = pR
p = solver_brent(p0,WL,WR)
print(p)
print(Fp(p,WL,WR))
rho0 = 0.0
phi = 0.0
if p >= pR:
    # 激波
    mu2 = (gammaR - 1) / (gammaR + 1)
    rho0 += rhoR * (p + mu2 * pR) / (mu2 * p + pR)
    phi += ((p - pR) * ((1 - mu2) / (rhoR * (p + mu2 * pR))) ** 0.5)
else:
    idx = 1 / gammaR
    rho0 += rhoR * (p / pR) ** idx
    phi += (2 * WR.a() / (gammaR - 1) * ((p / pR) ** ((gammaR - 1) / (2 * gammaR)) - 1))

u0 = phi + uR
p0 = p
print("---")
print(deflagration_speed(rho0,u0,p0))
print(rho0,u0,p0)



# p_min = 0.5e5
# p_max = 5e5
# p_mid = 0.5*(p_min+p_max)
#
# Res = Fp(p_mid,WL,WR)
# while abs(Res)>1e-8:
#     Res=Fp(p_mid,WL,WR)
#     if Res>0:
#         p_max = p_mid
#         p_mid = 0.5*(p_min+p_max)
#     else:
#         p_min = p_mid
#         p_mid = 0.5*(p_min+p_max)
# #
# u_mid = WL.u - f_L_p(p_mid,WL)
#
# # x = np.linspace(x_L,x_R,N)
# p_star = p_mid
# u_star = u_mid
#
#
# V = deflagration_speed(WR)
# p1 = p_star
# u1 = u_star
# v0 = WR.u - V
# c0 = WR.a()
# M2 = (v0 / c0) ** 2
# gamma = WR.gamma
# mu2 = (gamma - 1) / (gamma + 1)
# p_ratio1 = (1 - mu2) * (1 + gamma * M2)
# p_ratio2 = math.sqrt(((1 + mu2) ** 2) * ((1 - M2) ** 2) + 8 * mu2 * (gamma ** 2) * M2 * dq / (c0 ** 2))
# p0 = 2*p1/(p_ratio1+p_ratio2)
# # 压强 p
# print("p")
# print(pL)
# print(p_star)
# print(p1)
# print(p0)
# print(pR)
# # 左侧的激波 或者 稀疏波
# S = u_star
# rho = np.zeros(N)
# u = np.zeros(N)
# p = np.zeros(N)
#
#
# if p_star > pL:
#     # 激波
#     mu2 = (gammaL - 1) / (gammaL + 1)
#     p_r = p_star/pL
#     rhoL_star = WL.rho * (p_r + mu2) / (mu2 * p_r + 1)
#     SL = WL.u - WL.a() * ((gammaL+1)*p_star/(2*gammaL*pL) + (gammaL-1)/(2*gammaL))**0.5
#     for i in range(N):
#         if x[i]-x_c < SL*t_end:
#             rho[i] = WL.rho
#             u[i] = WL.u
#             p[i] = WL.p
#         elif x[i]-x_c < S*t_end:
#             rho[i] = rhoL_star
#             u[i] = u_star
#             p[i] = p_star
# # 右侧是一个爆轰激波
# M = - (p_star-pR)/(u_star-uR)
# SR = uR - M/rhoR
# rhoR_star = M/(u_star-SR)
# for i in range(N):
#     if (x[i]-x_c) > SR*t_end:
#         rho[i] = rhoR
#         u[i] = uR
#         p[i] = pR
#     elif ((x[i]-x_c) >= S*t_end) and ((x[i]-x_c) <= SR*t_end):
#         rho[i] = rhoR_star
#         u[i] = u_star
#         p[i] = p_star
# 右侧的解 这是一个激波


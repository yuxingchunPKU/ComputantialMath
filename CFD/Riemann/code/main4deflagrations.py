import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import FluidReact as FR
from typing import *

rhoL = 0.142168
uL = 0
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
t_end = 0.0005
N = 1000
x_c = 0.8
x_L = 0.0
x_R = 1.6

# 燃烧速度函数
def deflagration_speed(WR):
    return WR.u + (3e-9)*(WR.p/WR.rho)**2
# 特征函数
def f_L_p(p,WL):
    A_L = 2 / ((WL.gamma + 1) * WL.rho)
    B_L = (WL.gamma - 1) * (WL.p) / (WL.gamma + 1)
    if p >= WL.p:
        return (p - WL.p) * (A_L / (B_L + p)) ** 0.5
    else:
        return (2 * WL.a() / (WL.gamma - 1)) * ((p/ WL.p ) ** ((WL.gamma - 1) / (2 * WL.gamma)) - 1)

def f_de_R_p(p,WR,V,dq):
    '''
    推导爆燃波前和波后的速度差值
    :param p: 压强
    :param WR: 右侧流体
    :param V: 燃烧波的速度
    :return:
    '''
    v0 = WR.u - V
    c0 = WR.a()
    M2 = (v0/c0)**2
    gamma = WR.gamma
    mu2 = (gamma - 1) / (gamma + 1)
    p_ratio1 = (1-mu2)*(1+gamma*M2)
    p_ratio2 = math.sqrt((1+mu2)*(1-M2)+8*mu2*(gamma**2)*M2*dq/(c0**2))
    p1 = p
    p0 = p1/(0.5*(p_ratio1+p_ratio2))
    tauR = 1/WR.rho
    phi_R = 0.0
    rho0 = 0.0
    if p0 >= WR.p:
        phi_R += (p0-WR.p)*((1-mu2)*tauR/(p0+mu2*WR.p))**0.5
        rho0 += WR.rho*((mu2+(p0/WR.p))/(mu2*(p0/WR.p)+1))
    else:
        idx = (gamma-1)/(2*gamma)
        phi_R += (((1-mu2)*tauR*WR.p)**0.5)*((p0**idx)-(WR.p**idx))/mu2
        rho0 += WR.rho*((p0/WR.p)**(1/gamma))
    tau0 = 1/rho0
    # 由tau0 计算 phi_de
    phi_de = (p1-p0)*(((1-mu2)*tau0-2*mu2*dq/(p0-p1))/(mu2*p0+p1))**0.5
    return phi_R+phi_de
# 计算 C_J状态

def Fp(p,WL,WR):
    delta = WL.q - WR.q
    V = deflagration_speed(WR)
    return f_L_p(p,WL) + f_de_R_p(p,WR,V,delta)+WR.u-WL.u


dq = WL.q - WR.q

p_min = 0.5e5
p_max = 2.5e5
p_mid = 0.5*(p_min+p_max)

Res = Fp(p_mid,WL,WR)
while abs(Res)>1e-8:
    Res=Fp(p_mid,WL,WR)
    if Res>0:
        p_max = p_mid
        p_mid = 0.5*(p_min+p_max)
    else:
        p_min = p_mid
        p_mid = 0.5*(p_min+p_max)
#
u_mid = WL.u - f_L_p(p_mid,WL)

# x = np.linspace(x_L,x_R,N)
p_star = p_mid
u_star = u_mid
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







# 对完整解进行绘图

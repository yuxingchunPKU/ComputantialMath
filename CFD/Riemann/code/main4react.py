import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import FluidReact as FR
from typing import *

rhoL = 1.57861
uL = 2799.82
pL = 7.70752e6
qL = 0.0
gammaL = 1.27
rhoR = 0.601
uR = 0.0
pR = 1.0e5
qR = 242000/0.018
gammaR = 1.27

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
x_c = 4.0
x_L = 0.0
x_R = 8.0

# 特征函数
def f_L_p(p,WL):
    A_L = 2 / ((WL.gamma + 1) * WL.rho)
    B_L = (WL.gamma - 1) * (WL.p) / (WL.gamma + 1)
    if p >= WL.p:
        return (p - WL.p) * (A_L / (B_L + p)) ** 0.5
    else:
        return (2 * WL.a() / (WL.gamma - 1)) * ((p/ WL.p ) ** ((WL.gamma - 1) / (2 * WL.gamma)) - 1)

# 计算 C_J状态
def rho_u_p_CJ(WR,delta):
    p = WR.p
    rho = WR.rho
    u = WR.u
    gamma = WR.gamma
    mu2 = (gamma - 1) / (gamma + 1)
    t1 = 2*gamma*p/(((gamma**2)-1)*rho*delta)
    t2 = 1 + (1-t1)**(1/2)
    pcj = p - (gamma-1)*rho*delta*t2
    rhocj = rho*(pcj*(1+gamma)-p)/(gamma*pcj)
    tau = 1/rho
    t3 = ((1-mu2)*tau-2*mu2*delta/(p-pcj))/(mu2*p+pcj)
    ucj = u + (pcj-p)*(t3**(1/2))
    return rhocj,ucj,pcj

def f_R_p(p,WR,rhocj,ucj,pcj,delta):
    mu2 = (WR.gamma - 1) / (WR.gamma + 1)
    tau0 = 1/WR.rho
    p0 = WR.p
    u0 = WR.u
    if p >= pcj:
        return (p-p0)*(((1-mu2)*tau0-2*mu2*delta/(p0-p))/(mu2*p0+p))**(1/2)
    else:
        return ucj+(p-pcj)*(((1-mu2)/rhocj-2*mu2*delta/(pcj-p))/(mu2*pcj+p))**(1/2)

def Fp(p,WL,WR,rhocj,ucj,pcj):
    delta = WL.q - WR.q
    return f_L_p(p,WL) + f_R_p(p,WR,rhocj,ucj,pcj,delta)+WR.u-WL.u


dq = WL.q - WR.q
[rho_cj,u_cj,p_cj] = rho_u_p_CJ(WR,dq)

p_min = p_cj
p_max = 3*p_cj
p_mid = 0.5*(p_min+p_max)

u = Fp(p_mid,WL,WR,rho_cj,u_cj,p_cj)
while abs(u)>1e-8:
    u=Fp(p_mid,WL,WR,rho_cj,u_cj,p_cj)
    if u>0:
        p_max = p_mid
        p_mid = 0.5*(p_min+p_max)
    else:
        p_min = p_mid
        p_mid = 0.5*(p_min+p_max)

u_mid = WL.u - f_L_p(p_mid,WL)
# print(u_mid)
# print(pL)
# print(p_mid)
# print(p_cj)


x = np.linspace(x_L,x_R,N)
p_star = p_mid
u_star = u_mid
# 左侧的激波 或者 稀疏波
S = u_star
rho = np.zeros(N)
u = np.zeros(N)
p = np.zeros(N)


if p_star > pL:
    # 激波
    mu2 = (gammaL - 1) / (gammaL + 1)
    p_r = p_star/pL
    rhoL_star = WL.rho * (p_r + mu2) / (mu2 * p_r + 1)
    SL = WL.u - WL.a() * ((gammaL+1)*p_star/(2*gammaL*pL) + (gammaL-1)/(2*gammaL))**0.5
    for i in range(N):
        if x[i]-x_c < SL*t_end:
            rho[i] = WL.rho
            u[i] = WL.u
            p[i] = WL.p
        elif x[i]-x_c < S*t_end:
            rho[i] = rhoL_star
            u[i] = u_star
            p[i] = p_star
# 右侧是一个爆轰激波
M = - (p_star-pR)/(u_star-uR)
SR = uR - M/rhoR
rhoR_star = M/(u_star-SR)
for i in range(N):
    if (x[i]-x_c) > SR*t_end:
        rho[i] = rhoR
        u[i] = uR
        p[i] = pR
    elif ((x[i]-x_c) >= S*t_end) and ((x[i]-x_c) <= SR*t_end):
        rho[i] = rhoR_star
        u[i] = u_star
        p[i] = p_star
# 右侧的解 这是一个激波

print(SL)
print(SR)
print(u_star)
print(rhoL)
print(rhoL_star)
print(rhoR_star)
print(rhoR)
plt.plot(x,rho)
plt.show()
plt.clf()
plt.plot(x,u)
plt.show()
plt.clf()
plt.plot(x,p)
plt.show()







# 对完整解进行绘图

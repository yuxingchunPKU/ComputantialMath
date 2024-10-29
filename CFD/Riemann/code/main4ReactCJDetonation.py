import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import FluidReact as FR
from typing import *


rhoL = 0.974514
uL = 1586.05
pL = 4.04266e6
qL = 0.0
gammaL = 1.27

rhoR = 0.601
uR = 0.0
pR = 1.0e5
qR = 242000/0.018
gammaR = 1.27

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
    gamma = WR.gamma
    mu2 = (WR.gamma - 1) / (WR.gamma + 1)
    tau0 = 1/WR.rho
    p0 = WR.p
    u0 = WR.u
    taucj = 1/rhocj
    if p >= pcj:
        return (p-p0)*(((1-mu2)*tau0-2*mu2*delta/(p0-p))/(mu2*p0+p))**(1/2)
    else:
        part1 = (pcj-p0)*(((1-mu2)*tau0-2*mu2*delta/(p0-pcj))/(mu2*p0+pcj))**(1/2)
        part2 = 2*(math.sqrt(gamma)/(gamma-1))*math.sqrt(pcj*taucj)*((p/pcj)**((gamma-1)/(2*gamma))-1)
        return part1+part2

def Fp(p,WL,WR,rhocj,ucj,pcj):
    delta = WL.q - WR.q
    return f_L_p(p,WL) + f_R_p(p,WR,rhocj,ucj,pcj,delta)+WR.u-WL.u


dq = WL.q - WR.q
[rho_cj,u_cj,p_cj] = rho_u_p_CJ(WR,dq)

p_min = 0.5*p_cj
p_max = 1*p_cj
p_mid = 0.98*p_cj

u = Fp(p_mid,WL,WR,rho_cj,u_cj,p_cj)
print(u)
while abs(u)>1e-4:
    u=Fp(p_mid,WL,WR,rho_cj,u_cj,p_cj)
    if u>0:
        p_max = p_mid
        p_mid = 0.5*(p_min+p_max)
    else:
        p_min = p_mid
        p_mid = 0.5*(p_min+p_max)


u_star = WL.u - f_L_p(p_mid,WL)
p_star = p_mid
S = u_star
print(u_star)
print(p_star)
# # 左侧是稀疏波 右侧是稀疏波连着激波
x = np.linspace(x_L,x_R,N)
rho = np.zeros(N)
u = np.zeros(N)
p = np.zeros(N)

mu2 = (gammaL - 1) / (gammaL + 1)
p_r = p_star / pL
rhoL_star = WL.rho * (p_r + mu2) / (mu2 * p_r + 1)
SL = WL.u - WL.a() * ((gammaL + 1) * p_star / (2 * gammaL * pL) + (gammaL - 1) / (2 * gammaL)) ** 0.5
for i in range(N):
    if x[i] - x_c < SL * t_end:
        rho[i] = WL.rho
        u[i] = WL.u
        p[i] = WL.p
    elif x[i] - x_c < S * t_end:
        rho[i] = rhoL_star
        u[i] = u_star
        p[i] = p_star

rhoR_star = rho_cj * (p_star/p_cj)**(1/gammaR)
a_cj = (gammaR*p_cj/rho_cj)**0.5
aR_star = a_cj*(p_star/p_cj)**((gammaR-1)/(2*gammaR))
SHR = u_cj + a_cj
STR = u_star + aR_star
for i in range(N):
    if x[i] - x_c > SHR*t_end:
        rho[i] = rhoR
        u[i] = uR
        p[i] = pR
    elif x[i] - x_c > STR*t_end:
        x_t = (x[i] - x_c)/t_end
        rho[i] = rho_cj*(2/(gammaR+1)-((gammaR-1)/(a_cj*(gammaR+1)))*(u_cj-x_t))**(2/(gammaR-1))
        u[i] = 2/(gammaR+1)*(-a_cj+0.5*(gammaR-1)*u_cj+x_t)
        p[i] = p_cj*(2/(gammaR+1)-((gammaR-1)/(a_cj*(gammaR+1)))*(u_cj-x_t))**(2*gammaR/(gammaR-1))
    elif x[i] - x_c > S*t_end:
        rho[i] = rhoR_star
        u[i] = u_star
        p[i] = p_star

M = - (p_cj-pR)/(u_cj-uR)
SR = uR - M/rhoR
print(SR)
print(SHR)
print(STR)
print(S)
print(SL)
print("-------")
print(rhoR_star)
print(rhoL_star)
print(rhoL)
print("-------")
print("-------")
print(p_cj)
print(p_star)
print(pL)
plt.plot(x,rho)
plt.grid()
plt.show()
plt.clf()
plt.plot(x,u)
plt.grid()
plt.show()
plt.clf()
plt.plot(x,p)
plt.grid()
plt.show()

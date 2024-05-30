'''
理想气体的Euler格式
'''
import matplotlib.pyplot as plt
import numpy as np
import Fluid1d as Fd
import NumericalFlux as NF
import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.append("../Riemann")
import Fluid as Fdexact
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS
def init_fluid_data(U_data, Ele_c):
    gamma = 1.4
    for i in range(len(Ele_c)):
        if Ele_c[i] < 0.5:
            rho = 1
            u = 0
            P = 1.0
            U_data[0,i] = rho
            U_data[1, i] = rho * u
            U_data[2, i] = P / (gamma - 1) + 0.5 * rho * u * u
        else:
            rho = 0.125
            u = 0
            P = 0.1
            U_data[0,i] = rho
            U_data[1,i] = rho * u
            U_data[2,i] = P / (gamma - 1) + 0.5 * rho * u * u

DIM = 1
CFL = 0.2
U_init = 1e-8 * np.ones(DIM + 2, dtype=float)
gamma = 1.4
x_min = 0
x_max = 1
Nx = 1200
max_t = 0.15
point = np.linspace(x_min , x_max, Nx + 1)
n_edge = np.shape(point)[0]
n_ele = n_edge - 1
ele_vol = (x_max - x_min) / n_ele
edge_mark = np.zeros(n_edge, dtype=int)
edge_mark[0] = 1
edge_mark[-1] = 1

ele2edge = np.array([np.arange(0, n_edge - 1), np.arange(1, n_edge)])
ele2edge = ele2edge.transpose()
ele2point = ele2edge
# edge2cell = np.array([n_edge,2])
edge2cell = np.array([np.arange(-1, n_ele), np.arange(0, n_ele + 1)])
edge2cell[-1, -1] = -1
edge2cell = edge2cell.transpose()
# 将向量组合在一起 可能不再是array了
ele_centroid = 0.5 * (point[0:-1] + point[1:])

# 对数据进行初始化
U_data = np.zeros([DIM+2,Nx],dtype=float)
# 守恒量 初始值
init_fluid_data(U_data, ele_centroid)

t = 0
is_stop = False
while (not is_stop):
    max_speed = 0
    Fluxes = np.zeros([DIM + 2, n_edge],dtype=float)
    # lax-Friedriches
    for i in range(n_edge):
        if edge2cell[i, 0] == -1:
            Ur = U_data[:,edge2cell[i, 1]]
            Ul = Ur
        elif edge2cell[i, 1] == -1:
            Ul = U_data[:,edge2cell[i, 0]]
            Ur = Ul
        else:
            Ul = U_data[:,edge2cell[i, 0]]
            Ur = U_data[:,edge2cell[i, 1]]
        Ul1d = Fd.Fluid_1d(Ul)
        Ur1d = Fd.Fluid_1d(Ur)
        NF1 = NF.NumericalFlux(Ul1d,Ur1d)
        max_speed = max(max_speed, Ul1d.max_speed())
        max_speed = max(max_speed, Ur1d.max_speed())
        Fluxes[:, i] = NF1.HLLC()
    #判断是否停止
    dt = CFL * ele_vol / max_speed
    if (t+dt>=max_t and t<max_t):
        dt = max_t-t
        is_stop= True
    t = t + dt
    # 更新守恒量
    U_data=U_data-(dt / ele_vol)*(Fluxes[:,1:]-Fluxes[:,:-1])

print(t)
Rho = np.zeros(n_ele)
V = np.zeros(n_ele)
P = np.zeros(n_ele)

for i in range(n_ele):
    U1d = Fd.Fluid_1d(U_data[:,i])
    Rho[i] = U1d.U[0]
    V[i] = U1d.U[1] / U1d.U[0]
    P[i] = U1d.p

print(Rho)
rho_L = 1.0
u_L = 0.0
p_L = 1.0
gamma_L = 1.4
p_inf_L = 0.0
# right data
rho_R = 0.125
u_R = 0.0
p_R = 0.1
gamma_R = 1.4
p_inf_R = 0.0

x_L = -0.5
x_R = 0.5

N=1000
tol = 1e-12
maxit = 1000
WL = Fdexact.Fluid(rho_L,u_L,p_L,gamma_L,p_inf_L)
WR = Fdexact.Fluid(rho_R,u_R,p_R,gamma_R,p_inf_R)
ERS_solver = ERS.ExactRiemannSolver(WL,WR,tol,maxit)
[p_star,u_star]=ERS_solver.solver()
Complete_solver = ERCS.Riemann_Complete_solve(p_star,u_star,WL,WR,t,x_L,x_R,N)
[x,rho,u,p] = Complete_solver.solve()

plt.figure()
plt.plot(x+0.5,rho,label='Exact')
plt.plot(ele_centroid, Rho,label='Numerical')
plt.grid()
plt.legend()
plt.ylabel(r"$\rho$")
plt.savefig("rho.png",dpi=300)
plt.clf()
plt.plot(x+0.5,u,label='Exact')
plt.plot(ele_centroid, V,label='Numerical')
plt.grid()
plt.legend()
plt.ylabel(r"$u$")
plt.savefig("u.png",dpi=300)
plt.clf()
plt.plot(x+0.5,p,label='Exact')
plt.plot(ele_centroid, P,label='Numerical')
plt.grid()
plt.legend()
plt.ylabel(r"$p$")
plt.savefig("p.png",dpi=300)
plt.clf()
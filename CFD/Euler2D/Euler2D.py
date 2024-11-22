import numpy as np
import meshio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from fealpy.mesh import TriangleMesh
import Fluid2d as Fd
import NumericalFlux2d as NF
import sys
sys.path.append("../Riemann")
import Fluid as Fdexact
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS


mesh_gmsh =meshio.read("./Gmsh2.msh",file_format='gmsh')
pnt = mesh_gmsh.points[:,0:2]
cell = mesh_gmsh.cells_dict['triangle']
triangle = mtri.Triangulation(pnt[:, 0], pnt[:, 1], cell)
mesh = TriangleMesh(pnt,cell)

'''
几何信息
'''
n_node = mesh.number_of_nodes()
n_edge = mesh.number_of_edges()
n_cell = mesh.number_of_cells()
#边与单元的关系
cell2edge = mesh.ds.cell_to_edge()
edge2cell = mesh.ds.edge_to_cell()
isBDedge = mesh.ds.boundary_edge_flag() #是否是边界点的向量
edge_length = mesh.entity_measure('edge')
tri_vol = mesh.entity_measure('cell')
tri_center = mesh.entity_barycenter('cell')
edge_norm = mesh.edge_unit_normal() #单位外法向
# 计算内接圆的半径
sum_edge_len = edge_length[cell2edge[:,0]]+edge_length[cell2edge[:,1]]+edge_length[cell2edge[:,2]]
inter_cir_raddi = 2*tri_vol/sum_edge_len

DIM = 2
CFL = 0.1
U_init = 1e-8 * np.ones(DIM + 2, dtype=float)

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
max_t = 0.15
'''
初始值
'''
U_data = np.zeros([DIM+2,n_cell],dtype=float)
for i in range(n_cell):
    if tri_center[i,0]<0.5:
        U_data[0,i] = rho_L
        U_data[1,i] = rho_L*u_L
        U_data[2,i] = 0
        U_data[3,i] = p_L/(gamma_L-1) + 0.5*rho_L*u_L*u_L
    else:
        U_data[0, i] = rho_R
        U_data[1, i] = rho_R * u_R
        U_data[2, i] = 0
        U_data[3, i] = p_R / (gamma_R - 1) + 0.5 * rho_R * u_R * u_R

'''
开始循环
'''
t= 0
is_stop = False
while (not  is_stop):
    max_speed =0
    dU = np.zeros([DIM+2,n_cell],dtype=float)
    #计算各个边界的通量
    for i in range(n_edge):
        '''
        判断是否是边界
        '''
        Uin = np.zeros([DIM+2],dtype=float)
        Uout = np.zeros([DIM+2], dtype=float)
        if (isBDedge[i]):
            in_cell_idx = edge2cell[i,0]
            Uin[:]=U_data[:,in_cell_idx]
            # 透射边界条件
            Uout[:]=Uin
        else:
            in_cell_idx = edge2cell[i,0]
            out_cell_idx = edge2cell[i,1]
            Uin[:] = U_data[:, in_cell_idx]
            Uout[:] = U_data[:, out_cell_idx]
        Uin2d = Fd.Fluid_2d(Uin)
        Uout2d = Fd.Fluid_2d(Uout)
        #计算最大特征速度
        max_speed = max(max_speed,Uin2d.max_speed())
        max_speed = max(max_speed,Uout2d.max_speed())
        NF2 = NF.NumericalFlux(Uin2d,Uout2d,edge_norm[i,:])
        # 计算通量
        Flux= NF2.LLF()
        Flux = Flux*edge_length[i]
        # 更新
        dU[:,edge2cell[i, 0]] += Flux
        if (not isBDedge[i]):
            dU[:,edge2cell[i,1]] -= Flux


    # 计算时间步长
    dt = CFL*np.min(inter_cir_raddi)/max_speed
    if (t+dt>=max_t and t<max_t):
        dt = max_t-t
        is_stop= True
    t = t + dt
    # 更新守恒量
    U_data=U_data - dt*(dU/tri_vol)
    print("dt="+str(dt)+",   "+"t="+str(t))

Rho = np.zeros(n_cell)
V = np.zeros(n_cell)
P = np.zeros(n_cell)

for i in range(n_cell):
    U = Fd.Fluid_2d(U_data[:,i])
    Rho[i] =U.rho
    V[i] = np.linalg.norm(U.u)
    P[i] = U.p

#抽样 一维的数据进行计算
Rho1d = []
V1d = []
P1d = []
X = []
for i in range(n_cell):
    U = Fd.Fluid_2d(U_data[:, i])
    if (tri_center[i,1]-0.05)<0.0005:
        Rho1d.append(U.rho)
        V1d.append(np.linalg.norm(U.u))
        P1d.append(U.p)
        X.append(tri_center[i,0])

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

plt.plot(X,Rho1d,color='blue', marker='o',linewidth=0,markerfacecolor='none')
plt.plot(x+0.5,rho,label='Exact')
plt.show()

plt.clf()
plt.plot(X,V1d,color='blue', marker='o',linewidth=0,markerfacecolor='none')
plt.plot(x+0.5,u,label='Exact')
plt.show()

plt.clf()
plt.plot(X,P1d,color='blue', marker='o',linewidth=0,markerfacecolor='none')
plt.plot(x+0.5,p,label='Exact')
plt.show()
# plt.tripcolor(triangle,Rho, edgecolors='none', cmap=plt.cm.rainbow, alpha=0.8)
# plt.show()

'''
2024年8月15日
算的不对 还是有问题

'''


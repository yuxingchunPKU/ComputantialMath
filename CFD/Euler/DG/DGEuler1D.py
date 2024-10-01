from scipy.special import roots_legendre
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.append("../")
import Fluid1d as Fd1d
import NumericalFlux as NF
sys.path.append("../../Riemann")
import Fluid as Fdexact
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS
from pyinstrument import Profiler

TVB_para = 0
M_b = 2
# minmod 函数
def minmod(a,b,c,M=10,dx=1):
    bc_min = np.minimum(b*np.sign(a),c*np.sign(a))
    return a*(np.abs(a) <= M*dx**2)+(np.sign(a)*np.maximum(0,np.minimum(np.abs(a),bc_min)))*(np.abs(a) > M*dx**2)

def bais(x,x_i,dx):
    N_x = len(x)
    B = np.zeros([M_b, N_x], dtype=float)
    B[0, :] = 1 * np.ones(N_x, dtype=float)
    match M_b:
        case 2:
            B[1, :] = (x - x_i) / (0.5 * dx)
        case 3:
            B[1, :] = (x - x_i) / (0.5 * dx)
            B[2, :] = ((x - x_i) / (0.5 * dx)) ** 2 - 1 / 3
    return B

def bais_x(x,x_i,dx):
    N_x = len(x)
    B = np.zeros([M_b, N_x], dtype=float)
    match M_b:
        case 2:
            B[1, :] = np.ones(N_x, dtype=float) / (0.5 * dx)
        case 3:
            B[1, :] = np.ones(N_x, dtype=float) / (0.5 * dx)
            B[2, :] = 2*(x-x_i)*np.ones(N_x,dtype=float)/((0.5*dx)**2)
    return B

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
CFL = 0.1
x_min = 0
x_max = 1

gamma = 1.4
Nx = 200
max_t = 0.15
point = np.linspace(x_min, x_max, Nx + 1)
n_edge = np.shape(point)[0]
n_ele = n_edge - 1
ele_vol = (x_max - x_min) / n_ele
edge_mark = np.ones(n_edge, dtype=int)
edge_mark[0] = -1
edge_mark[-1] = 0

ele2edge = np.array([np.arange(0, n_edge - 1), np.arange(1, n_edge)])
ele2edge = ele2edge.transpose()
ele2point = ele2edge
edge2cell = np.array([np.arange(-1, n_ele), np.arange(0, n_ele + 1)])
edge2cell[-1, -1] = -1
edge2cell = edge2cell.transpose()
ele_centroid = 0.5 * (point[0:-1] + point[1:])

# 对数据进行初始化
U_data = np.zeros([DIM+2,M_b,Nx],dtype=float)
# 守恒量 初始值
init_fluid_data(U_data[:,0,:], ele_centroid)
'''
需要对初始值解方程
'''
W = 4
x_int_ref, wi = roots_legendre(W)
wi *= 0.5
# for i in range(Nx):
#     x_int = 0.5 * ele_vol * x_int_ref + ele_centroid[i]
#     B = bais(x_int, ele_centroid[i], ele_vol)
#     M = (B*wi)@np.transpose(B)*ele_vol
#     U = ele_vol*np.dot(B*init_fluid_data(x_int),wi)
#     U_data[:,i] = np.linalg.inv(M)@U

t = 0
is_stop = False

profile = Profiler()
profile.start()
while (not is_stop):
    U_data_cp = U_data
    def step_forward(U_data,max_speed):
        dU = np.zeros([DIM+2,M_b,Nx],dtype=float)
        edge_bias_r = bais(point[0:-1],ele_centroid,ele_vol)
        edge_bias_l = bais(point[1:],ele_centroid,ele_vol)
        Ul = np.zeros([DIM+2,n_edge],dtype=float)
        Ur = np.zeros([DIM+2,n_edge],dtype=float)
        Ul[:,1:] = np.sum(edge_bias_l*U_data,axis=1)
        Ur[:,0:-1] = np.sum(edge_bias_r*U_data,axis=1)
        # 周期边界条件
        # Ul[:,0] = Ul[:,-1]
        # Ur[:,-1] = Ur[:,0]
        Ul[:,0] = Ur[:,0]
        Ur[:,-1] = Ul[:,-1]
        # 需要在遍历 单元边界的时候 计算通量 和 最大传播速度
        Fu_edge = np.zeros([DIM+2, n_edge],dtype=float)
        for i in range(n_edge):
            Ul1d = Fd1d.Fluid_1d(Ul[:,i])
            Ur1d = Fd1d.Fluid_1d(Ur[:,i])
            NF1d = NF.NumericalFlux(Ul1d,Ur1d)
            max_speed = max(max_speed, Ul1d.max_speed())
            max_speed = max(max_speed, Ur1d.max_speed())
            Fu_edge[:,i] = NF1d.LLF()
            #Fu_edge[:,i] = Ul1d.flux()
        # 遍历每一个单元 然后在单元的内部计算增加量
        for i in range(n_ele):
            # 左右端点的坐标 和 基函数的值
            ele_point_lr = [ele_centroid[i]-0.5*ele_vol,ele_centroid[i]+0.5*ele_vol]
            ele_bais_lr = bais(ele_point_lr,ele_centroid[i],ele_vol)
            # 计算边界上的通量
            V1 = np.outer(Fu_edge[:,i+1],ele_bais_lr[:,1])-np.outer(Fu_edge[:,i],ele_bais_lr[:,0])
            '''
            1.取积分点
            2.计算基函数在积分点上的值
            3.组合成函数值 并计算通量
            4.计算导数的梯度值 并乘以权重
            5.求和
            '''
            x_int = 0.5 * ele_vol * x_int_ref + ele_centroid[i]
            bais_val = bais(x_int,ele_centroid[i],ele_vol)
            # 计算导数的梯度值 乘权重
            bais_val_x = bais_x(x_int,ele_centroid[i],ele_vol)
            bais_val_x_wi = bais_val_x * wi
            # 内部的守恒量
            u = U_data[:,:,i] @ bais_val
            V2 = np.zeros([DIM+2,M_b],dtype=float)
            for j in range(W):
                U1d_in = Fd1d.Fluid_1d(u[:,j])
                FU1d = U1d_in.flux()
                V2[0, :] -= ele_vol * FU1d[0]*bais_val_x_wi[:, j]
                V2[1, :] -= ele_vol * FU1d[1]*bais_val_x_wi[:, j]
                V2[2, :] -= ele_vol * FU1d[2]*bais_val_x_wi[:, j]
                # V2[:,:] -= ele_vol * np.outer(FU1d,bais_val_x_wi[:,j])

            '''
            计算矩阵 M
            要改为右乘 矩阵M的逆
            '''
            M=(bais_val * wi) @ np.transpose(bais_val) * ele_vol
            dU[:,:,i] = (V1+V2)/np.diagonal(M)

        return dU,max_speed

    def TVB_limiter1(U_data):
        '''
        限制线性函数的限制器
        限制在单元边界上的梯度 以要求其满足局部极值原理 从而实现L^\infty 稳定性
        限制器的实现 来自于文章
        Runge–Kutta discontinuous Galerkin methods for compressible two-medium flow simulations: One-dimensional case
        :param U_data:
        :return:
        '''
        DU_p = np.zeros([DIM+2,n_ele],dtype=float)
        DU_m = np.zeros([DIM + 2, n_ele], dtype=float)
        DU_p[:,0:-1] = U_data[:,0,1:]-U_data[:,0,0:-1]
        DU_m[:,1:] = U_data[:,0,1:]-U_data[:,0,0:-1]
        # 自然边界条件 空的
        # 重构单元边界的值
        # 1. 基函数 单元内的作用
        # 2. 组合成值
        # 3. 计算梯度: 减去 守恒量的值
        # 4. 限制梯度
        # 5. 根据限制梯度的值求线性代数方程组
        ele_bias_r = bais(point[1:],ele_centroid,ele_vol)
        U_ele_r = np.sum(ele_bias_r*U_data,axis=1)
        U0 = U_data[:,0,:]
        dUp = U_ele_r - U0
        # dUp_mod = np.minimum(np.abs(dUp),np.abs(DUm))*((dUp*DUm)>0)
        # dUp_mod = np.minimum(np.abs(dUp_mod),np.abs(DUp))*((dUp*DUp)>0)
        # dUp_mod *= np.sign(dUp)
        # V1
        # dUp_mod = minmod(dUp,DUp,DUm,0,ele_vol)
        # V2 耗散大 但是稳定性好
        dUp_mod = minmod(dUp, DU_p, DU_m, 0, ele_vol)

        # 局部求解线性代数方程组 直接相等即可
        U_data[:,1,:] = dUp_mod
        return U_data

    def TVB_limiter2(U_data):
        '''
        限制二次函数的限制器
        限制在单元边界上的梯度 以要求其满足局部极值原理 从而实现L^\inty 稳定性
        限制器的实现 来自于文章
        Runge–Kutta discontinuous Galerkin methods for compressible two-medium flow simulations: One-dimensional case
        :param U_data:
        :return:
        '''
        DU_p = np.zeros([DIM+2,n_ele],dtype=float)
        DU_m = np.zeros([DIM + 2, n_ele], dtype=float)
        DU_p[:,0:-1] = U_data[:,0,1:]-U_data[:,0,0:-1]
        DU_m[:,1:] = U_data[:,0,1:]-U_data[:,0,0:-1]
        # 重构单元边界的值
        # 1. 基函数 单元内的作用
        # 2. 组合成值
        # 3. 计算梯度: 减去 守恒量的值
        # 4. 限制梯度
        # 5. 根据限制梯度的值求线性代数方程组
        bais_ele_l = bais(point[0:-1],ele_centroid,ele_vol)
        bais_ele_r = bais(point[1:],ele_centroid,ele_vol)
        # 单元边界上的重构值
        U_ele_l = np.sum(bais_ele_l*U_data,axis=1)
        U_ele_r = np.sum(bais_ele_r*U_data,axis=1)
        # 单元内的守恒量
        U_aver = U_data[:,0,:]
        dUr = U_ele_r - U_aver
        dUl = U_aver - U_ele_l
        #基于特征变量的限制器
        for i in range(n_ele):
            #建立1D对象取出一些辅助变量
            U1d = Fd1d.Fluid_1d(U_aver[:,i])
            rho = U1d.rho
            u=U1d.u
            c=U1d.sound_speed()
            p=U1d.p
            H = (U_aver[2,i]+p)/rho
            R = np.array([[1,1,1],[u-c,u,u+c],[H-u*c,0.5*u**2,H+u*c]])
            R_inv = np.linalg.inv(R)
            #调用minmod 函数来实现
            dUr_mod = R@minmod(R_inv@dUr[:,i],R_inv@DU_p[:,i],R_inv@DU_m[:,i],TVB_para,ele_vol)
            dUl_mod = R@minmod(R_inv@dUl[:,i],R_inv@DU_p[:,i],R_inv@DU_m[:,i],TVB_para,ele_vol)
            # 解出来U的值
            U_data[0, 1, i] = 0.5 * (dUr_mod[0] + dUl_mod[0])
            U_data[0, 2, i] = (3 / 4) * (dUr_mod[0] - dUl_mod[0])
            U_data[1, 1, i] = 0.5 * (dUr_mod[1] + dUl_mod[1])
            U_data[1, 2, i] = (3 / 4) * (dUr_mod[1] - dUl_mod[1])
            U_data[2, 1, i] = 0.5 * (dUr_mod[2] + dUl_mod[2])
            U_data[2, 2, i] = (3 / 4) * (dUr_mod[2] - dUl_mod[2])
            # U_data[1, 1, :] = 0.5 * (dUr_mod_rhou + dUl_mod_rhou)
            # U_data[1, 2, :] = (3 / 4) * (dUr_mod_rhou - dUl_mod_rhou)
            # U_data[2, 1, :] = 0.5 * (dUr_mod_E + dUl_mod_E)
            # U_data[2, 2, :] = (3 / 4) * (dUr_mod_E - dUl_mod_E)
        # dUr_mod_rho = minmod(dUr[0,:],DU_p[0,:],DU_m[0,:],TVB_para,ele_vol)
        # dUl_mod_rho = minmod(dUl[0,:],DU_p[0,:],DU_m[0,:],TVB_para,ele_vol)
        # dUr_mod_rhou = minmod(dUr[1,:],DU_p[1,:],DU_m[1,:],TVB_para,ele_vol)
        # dUl_mod_rhou = minmod(dUl[1,:],DU_p[1,:],DU_m[1,:],TVB_para,ele_vol)
        # dUr_mod_E = minmod(dUr[2,:],DU_p[2,:],DU_m[2,:],TVB_para,ele_vol)
        # dUl_mod_E = minmod(dUl[2,:],DU_p[2,:],DU_m[2,:],TVB_para,ele_vol)
        # 局部求解线性代数方程组
        return U_data
    max_speed = 0
    dU,max_speed = step_forward(U_data,max_speed)
    dt = CFL * ele_vol / max_speed
    if (t + dt >= max_t and t < max_t):
        dt = max_t - t
        is_stop = True
    t = t + dt
    # 一阶时间精度
    U_data = U_data - dt * dU

    match M_b:
        case 2:
            U_data = TVB_limiter1(U_data)
            dU, max_speed = step_forward(U_data, max_speed)
            U_data = 0.5 * U_data_cp + 0.5 * (U_data - dt * dU)
            U_data = TVB_limiter1(U_data)
        case 3:
            U_data = TVB_limiter2(U_data)
            dU, max_speed = step_forward(U_data, max_speed)
            U_data = 0.75 * U_data_cp + 0.25 * (U_data - dt * dU)
            U_data = TVB_limiter2(U_data)
            dU, max_speed = step_forward(U_data, max_speed)
            U_data = (1 / 3) * U_data_cp + (2 / 3) * (U_data - dt * dU)
            U_data = TVB_limiter2(U_data)
    # 二阶时间精度
    # U_data = TVB_limiter1(U_data)
    # dU, max_speed = step_forward(U_data, max_speed)
    # U_data = 0.5*U_data_cp + 0.5*(U_data-dt*dU)
    # U_data = TVB_limiter1(U_data)

    # 三阶时间精度
    # U_data = TVB_limiter2(U_data)
    #
    # dU, max_speed = step_forward(U_data, max_speed)
    # U_data = 0.75*U_data_cp + 0.25*(U_data-dt*dU)
    # U_data = TVB_limiter2(U_data)
    #
    # dU, max_speed = step_forward(U_data, max_speed)
    # U_data = (1 / 3) * U_data_cp + (2 / 3) * (U_data - dt * dU)
    # U_data = TVB_limiter2(U_data)


profile.stop()
profile.print()

# Bc = bais(ele_centroid, ele_centroid, ele_vol)
# Rho = Bc[0,:]*U_data[0,0,:]
# U = Bc[0,:]*U_data[1,0,:]/Rho
# P = (gamma-1)*(Bc[0,:]*U_data[2,0,:]-0.5*Rho*(U**2))


Bc = bais(ele_centroid, ele_centroid, ele_vol)
Rho = np.sum(Bc[0,:]*U_data[0,:,:],axis=0)
U = np.sum(Bc[0,:]*U_data[1,:,:],axis=0)/Rho
P = (gamma-1)*(np.sum((Bc[0,:]*U_data[2,:,:]),axis=0) -0.5*Rho*(U**2))



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



# U_exact = np.sin(2*np.pi*(ele_centroid-a*max_t))
plt.figure()
plt.plot(x+0.5,rho,label='Exact')
plt.plot(ele_centroid,Rho,'*-',label='DG P2')
plt.legend()
plt.show()
plt.clf()

plt.plot(x+0.5,u,label='Exact')
plt.plot(ele_centroid,U,'*-',label='DG P2')
plt.legend()
plt.show()
plt.clf()


plt.plot(x+0.5,p,label='Exact')
plt.plot(ele_centroid,P,'*-',label='DG P2')
plt.legend()
plt.show()



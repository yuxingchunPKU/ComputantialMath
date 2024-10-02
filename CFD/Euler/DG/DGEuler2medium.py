#一维两介质的求解
#在单介质黎曼问题的基础上，增加了两介质的求解
#1. 两种流体各自更新
#2. 界面上求解多介质黎曼问题
import math
import numpy as np
from scipy.special import roots_legendre
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import sys
sys.path.append("../")
import Fluid1d as Fd1d
import NumericalFlux as NF
sys.path.append("../../Riemann")
import Fluid as Fdexact
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS

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

def bais_single(x,x_i,dx):
    B = np.zeros(M_b, dtype=float)
    B[0] = 1
    match M_b:
        case 2:
            B[1] = (x - x_i) / (0.5 * dx)
        case 3:
            B[1] = (x - x_i) / (0.5 * dx)
            B[2] = ((x - x_i) / (0.5 * dx)) ** 2 - 1 / 3
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

def bais_x_single(x,x_i,dx):
    B = np.zeros(M_b, dtype=float)
    match M_b:
        case 2:
            B[1] = 1 / (0.5 * dx)
        case 3:
            B[1] = 1 / (0.5 * dx)
            B[2] = 2*(x-x_i)/((0.5*dx)**2)
    return B

# 两相的初值
def init_fluid_data(U_data0,U_data1,Ele_c):
    for i in range(n_ele):
        x_int = 0.5 * ele_vol * x_int_ref + ele_centroid[i]
        B = bais(x_int, ele_centroid[i], ele_vol)
        M = (B * wi) @ np.transpose(B) * ele_vol
        gamma = 1.4
        rho_ele = 1 + 0.2 * np.sin(np.pi * x_int)
        m_ele = rho_ele * 1
        E_ele = (1 / (gamma - 1)) + 0.5 * rho_ele * (1 ** 2)
        rho_f = ele_vol * np.dot(B * rho_ele, wi)
        m_f = ele_vol * np.dot(B * 1 * m_ele, wi)
        E_f = ele_vol * np.dot(B * E_ele, wi)
        # 初始时刻界面的位置在 1.0
        if Ele_c[i] < 1.0:
            U_data0[0,:,i] = rho_f/np.diagonal(M)
            U_data0[1,:,i] = m_f/np.diagonal(M)
            U_data0[2,:,i] = E_f/np.diagonal(M)
            U_data1[:,:,i] = 0
        else:
            U_data0[:,:,i] = 0
            U_data1[0,:,i] = rho_f/np.diagonal(M)
            U_data1[1,:,i] = m_f/np.diagonal(M)
            U_data1[2,:,i] = E_f/np.diagonal(M)
def interface_cell_idx(phi_input):
    # 标记界面点和界面单元 小于等于0的点是相0 大于0的点是相1
    N_edge = len(phi_input)
    point_label[:] = np.zeros(N_edge,dtype=int)*((phi_input<=0))+np.ones(N_edge,dtype=int)*((phi_input>0))
    point_label_add = point_label[0:-1]+point_label[1:] # 两个相加 0相0 1是界面单元 2是相1
    ele_label[:] = point_label_add
    idx_out = np.where(ele_label == 1)[0][0]
    return idx_out
def interface_cell_idx_new(phi_input):
    #界面移动之后的标记
    N_edge = len(phi_input)
    point_label_new[:] = np.zeros(N_edge,dtype=int)*((phi_input<=0))+np.ones(N_edge,dtype=int)*((phi_input>0))
    point_label_add = point_label_new[0:-1]+point_label_new[1:]
    ele_label_new[:] = point_label_add
    idx_out = np.where(ele_label_new == 1)[0][0]
    return idx_out
def update_phase_info(x_i):
    # 更新相的信息 相的体积和质心
    for i in range(n_ele):
        match ele_label[i]:
            case 0:
                phase_vol[0,i] = ele_vol
                phase_vol[1,i] = 0
                phase_centroid[0,i] = ele_centroid[i]
                phase_centroid[1,i] = ele_centroid[i]
            case 1:
                phase_vol[0,i] = math.fabs(x_i-point[i])
                phase_vol[1,i] = math.fabs(point[i+1]-x_i)
                phase_centroid[0,i] = (x_i + point[i])/2
                phase_centroid[1,i] = (point[i+1] + x_i)/2
            case 2:
                phase_vol[0,i] = 0
                phase_vol[1,i] = ele_vol
                phase_centroid[0,i] = ele_centroid[i]
                phase_centroid[1,i] = ele_centroid[i]
def update_phase_info_new(x_i):
    # 更新相的信息 相的体积和质心
    for i in range(n_ele):
        match ele_label_new[i]:
            case 0:
                phase_vol_new[0,i] = ele_vol
                phase_vol_new[1,i] = 0
                phase_centroid_new[0,i] = ele_centroid[i]
                phase_centroid_new[1,i] = ele_centroid[i]
            case 1:
                phase_vol_new[0,i] = math.fabs(x_i-point[i])
                phase_vol_new[1,i] = math.fabs(point[i+1]-x_i)
                phase_centroid_new[0,i] = (x_i + point[i])/2
                phase_centroid_new[1,i] = (point[i+1] + x_i)/2
            case 2:
                phase_vol_new[0,i] = 0
                phase_vol_new[1,i] = ele_vol
                phase_centroid_new[0,i] = ele_centroid[i]
                phase_centroid_new[1,i] = ele_centroid[i]

def update_interface_info():
    # 更新界面单元的信息
    IC0_v = np.sum(phase_vol[0, IC_idx - 1:IC_idx + 2])
    IC1_v = np.sum(phase_vol[1, IC_idx - 1:IC_idx + 2])
    IC0_c = np.dot(phase_centroid[0, IC_idx - 1:IC_idx + 2], phase_vol[0, IC_idx - 1:IC_idx + 2]) / IC0_v
    IC1_c = np.dot(phase_centroid[1, IC_idx - 1:IC_idx + 2], phase_vol[1, IC_idx - 1:IC_idx + 2]) / IC1_v
    return IC0_v,IC1_v,IC0_c,IC1_c

def update_interface_info_new():
    # 更新界面单元的信息
    IC0_v_n = np.sum(phase_vol_new[0, IC_idx - 1:IC_idx + 2])
    IC1_v_n = np.sum(phase_vol_new[1, IC_idx - 1:IC_idx + 2])
    IC0_c_n = np.dot(phase_centroid_new[0, IC_idx - 1:IC_idx + 2], phase_vol_new[0, IC_idx - 1:IC_idx + 2]) / IC0_v_n
    IC1_c_n = np.dot(phase_centroid_new[1, IC_idx - 1:IC_idx + 2], phase_vol_new[1, IC_idx - 1:IC_idx + 2]) / IC1_v_n
    return IC0_v_n,IC1_v_n,IC0_c_n,IC1_c_n
def update_interface_val():
    # 先执行相0 然后 执行相1
    # p 表示+  m 表示 -
    x_int_m1 = 0.5 * ele_vol * x_int_ref + ele_centroid[IC_idx - 1]
    ele_bias_m1 = bais(x_int_m1, ele_centroid[IC_idx - 1], ele_vol)
    # 界面单元上的基函数在本单元的取值
    ele_bias_m1_IC = bais(x_int_m1, IC0_centroid, IC0_vol)
    Um1 = (U_data0[:,:,IC_idx-1] @ ele_bias_m1)@ np.transpose(ele_bias_m1_IC*wi) *ele_vol
    Um0 = np.zeros([DIM+2,M_b],dtype=float)
    if phase_vol[0,IC_idx]> eps:
        x_int_m0 = 0.5 * phase_vol[0,IC_idx] * x_int_ref + phase_centroid[0,IC_idx]
        ele_bias_m0 = bais(x_int_m0, phase_centroid[0,IC_idx], phase_vol[0,IC_idx])
        ele_bias_m0_IC = bais(x_int_m0, IC0_centroid, IC0_vol)
        Um0[:,:] = (U_data0[:,:,IC_idx] @ ele_bias_m0) @ np.transpose(ele_bias_m0_IC*wi) * phase_vol[0,IC_idx]
    # Um1 和 Um2 的求和就是右端项
    x_int_IC0 = 0.5 * IC0_vol * x_int_ref + IC0_centroid
    ele_bias_IC0 = bais(x_int_IC0, IC0_centroid, IC0_vol)
    # 界面单元上的刚度矩阵
    M = (ele_bias_IC0 * wi) @ np.transpose(ele_bias_IC0) * IC0_vol
    # 得到界面单元上的新值
    U_IC0[:,:] = (Um0+Um1)/np.diagonal(M)
    # 类似地 实现对相1的更新
    x_int_p1 = 0.5 * ele_vol * x_int_ref + ele_centroid[IC_idx+1]
    ele_bias_p1 = bais(x_int_p1, ele_centroid[IC_idx+1], ele_vol)
    ele_bias_p1_IC = bais(x_int_p1, IC1_centroid, IC1_vol)
    Up1 = (U_data1[:,:,IC_idx+1] @ ele_bias_p1)@ np.transpose(ele_bias_p1_IC*wi) *ele_vol
    Up0 = np.zeros([DIM+2,M_b],dtype=float)
    if phase_vol[1,IC_idx] > eps:
        x_int_p0 = 0.5 * phase_vol[1,IC_idx] * x_int_ref + phase_centroid[1,IC_idx]
        ele_bias_p0 = bais(x_int_p0, phase_centroid[1,IC_idx], phase_vol[1,IC_idx])
        ele_bias_p0_IC = bais(x_int_p0, IC1_centroid, IC1_vol)
        Up0[:,:] = (U_data1[:,:,IC_idx] @ ele_bias_p0) @ np.transpose(ele_bias_p0_IC*wi) * phase_vol[1,IC_idx]
    x_int_IC1 = 0.5 * IC1_vol * x_int_ref + IC1_centroid
    ele_bias_IC1 = bais(x_int_IC1, IC1_centroid, IC1_vol)
    # 刚度矩阵
    M = (ele_bias_IC1 * wi) @ np.transpose(ele_bias_IC1) * IC1_vol
    # 得到界面单元上的新值
    U_IC1[:,:] = (Up0+Up1)/np.diagonal(M)

def oneD_Riemann(Ul,Ur):
    # 调用多介质黎曼问题求解器 获得一维黎曼问题的精确解 并返回界面的速度
    UlIC1d = Fd1d.Fluid_1d(Ul,gamma=gamma0)
    UrIC1d = Fd1d.Fluid_1d(Ur,gamma=gamma1)
    WL = Fdexact.Fluid(UlIC1d.rho,UlIC1d.u,UlIC1d.p,gamma0,0)
    WR = Fdexact.Fluid(UrIC1d.rho,UrIC1d.u,UrIC1d.p,gamma1,0)
    ERS_solver = ERS.ExactRiemannSolver(WL, WR, tol, maxit)
    [p, u] = ERS_solver.solver()
    IC_flux[1] = p
    IC_flux[2] = p * u
    return u

def get_flux(U_data0,U_data1,max_speed):
    dU = np.zeros([DIM+2,M_b,Nx],dtype=float)
    dUIC = np.zeros([DIM+2,M_b,2],dtype=float)
    # 重构出单元边界上的值
    # 从0到倒数第二个顶点的右边界 从1到最后一个顶点的左边界
    edge_bias_r = bais(point[0:-1], ele_centroid, ele_vol)
    edge_bias_l = bais(point[1:], ele_centroid, ele_vol)
    Ul = np.zeros([DIM + 2, n_edge], dtype=float)
    Ur = np.zeros([DIM + 2, n_edge], dtype=float)
    #重构出边界上的值
    Ul[:, 1:IC_idx] = np.sum(edge_bias_l[:, 0:IC_idx - 1] * U_data0[:, :, 0:IC_idx - 1], axis=1)
    Ur[:, 0:IC_idx - 1] = np.sum(edge_bias_r[:, 0:IC_idx - 1] * U_data0[:, :, 0:IC_idx - 1], axis=1)
    Ul[:, IC_idx + 3:] = np.sum(edge_bias_l[:, IC_idx + 2:] * U_data1[:, :, IC_idx + 2:], axis=1)
    Ur[:, IC_idx + 2:-1] = np.sum(edge_bias_r[:, IC_idx + 2:] * U_data1[:, :, IC_idx + 2:], axis=1)
    # IC0的左端点 和 IC1的右端点 上的值
    edge_bias_r_IC0 = bais_single(point[IC_idx - 1], IC0_centroid, IC0_vol)
    edge_bias_l_IC1 = bais_single(point[IC_idx + 2], IC1_centroid, IC1_vol)
    Ur[:, IC_idx - 1] = np.sum(edge_bias_r_IC0 * U_IC0, axis=1)
    Ul[:, IC_idx + 2] = np.sum(edge_bias_l_IC1 * U_IC1, axis=1)
    # 周期边界
    Ul[:, 0] = Ul[:, -1]
    Ur[:, -1] = Ur[:, 0]
    # 计算通量
    Fu_edge = np.zeros([DIM + 2, n_edge], dtype=float)
    for i in range(n_edge):
        if i>=IC_idx and i<=IC_idx+1:
            pass
        else:
            Ul1d = Fd1d.Fluid_1d(Ul[:, i])
            Ur1d = Fd1d.Fluid_1d(Ur[:, i])
            NF1d = NF.NumericalFlux(Ul1d, Ur1d)
            max_speed = max(max_speed, Ul1d.max_speed())
            max_speed = max(max_speed, Ur1d.max_speed())
            Fu_edge[:, i] = NF1d.LLF()
    # 遍历非界面单元 在单元的内部计算增加量
    for i in range(n_ele):
        if i>=IC_idx-1 and i<=IC_idx+1:
            pass
        else:
            #建立单元内左右端点的值
            point_lr = [ele_centroid[i] - 0.5 * ele_vol, ele_centroid[i] + 0.5 * ele_vol]
            bais_lr = bais(point_lr, ele_centroid[i], ele_vol)
            V1 = np.outer(Fu_edge[:, i + 1], bais_lr[:, 1]) - np.outer(Fu_edge[:, i], bais_lr[:, 0])
            #内部的积分点值
            x_int = 0.5 * ele_vol * x_int_ref + ele_centroid[i]
            bais_val = bais(x_int, ele_centroid[i], ele_vol)
            bais_val_x = bais_x(x_int, ele_centroid[i], ele_vol)
            bais_val_x_wi = bais_val_x * wi
            u = np.zeros([DIM+2,W],dtype=float)
            if i<IC_idx-1:
                u[:,:] = U_data0[:, :, i]@ bais_val
            else:
                u[:,:] = U_data1[:, :, i]@ bais_val
            V2 = np.zeros([DIM + 2, M_b], dtype=float)
            for j in range(W):
                U1d_in = Fd1d.Fluid_1d(u[:, j])
                FU1d = U1d_in.flux()
                V2[:, :] -= ele_vol * np.outer(FU1d, bais_val_x_wi[:, j])
            M = (bais_val * wi) @ np.transpose(bais_val) * ele_vol
            dU[:, :, i] = (V1 + V2) / np.diagonal(M)
    # 计算界面单元内的增量
    point_lr_IC0 = [IC0_centroid - 0.5 * IC0_vol, IC0_centroid + 0.5 * IC0_vol]
    bais_lr_IC0 = bais(point_lr_IC0, IC0_centroid, IC0_vol)
    V1_IC0 = np.outer(IC_flux, bais_lr_IC0[:, 1]) - np.outer(Fu_edge[:, IC_idx-1], bais_lr_IC0[:, 0])

    point_lr_IC1 = [IC1_centroid - 0.5 * IC1_vol, IC1_centroid + 0.5 * IC1_vol]
    bais_lr_IC1 = bais(point_lr_IC1, IC1_centroid, IC1_vol)
    V1_IC1 = np.outer(Fu_edge[:, IC_idx+2], bais_lr_IC1[:, 1]) - np.outer(IC_flux, bais_lr_IC1[:, 0])
    # 内部积分点的值
    x_int_IC0 = 0.5 * IC0_vol * x_int_ref + IC0_centroid
    bais_val_IC0 = bais(x_int_IC0, IC0_centroid, IC0_vol)
    bais_val_x_IC0 = bais_x(x_int_IC0, IC0_centroid, IC0_vol)
    bais_val_x_wi_IC0 = bais_val_x_IC0 * wi
    u_IC0 = U_IC0 @ bais_val_IC0
    V2_IC0 = np.zeros([DIM + 2, M_b], dtype=float)

    x_int_IC1 = 0.5 * IC1_vol * x_int_ref + IC1_centroid
    bais_val_IC1 = bais(x_int_IC1, IC1_centroid, IC1_vol)
    bais_val_x_IC1 = bais_x(x_int_IC1, IC1_centroid, IC1_vol)
    bais_val_x_wi_IC1 = bais_val_x_IC1 * wi
    u_IC1 = U_IC1 @ bais_val_IC1
    V2_IC1 = np.zeros([DIM + 2, M_b], dtype=float)
    for j in range(W):
        U1d_in_IC0 = Fd1d.Fluid_1d(u_IC0[:, j])
        FU1d_IC0= U1d_in_IC0.flux()
        V2_IC0[:, :] -= IC0_vol * np.outer(FU1d_IC0, bais_val_x_wi_IC0[:, j])
        U1d_in_IC1 = Fd1d.Fluid_1d(u_IC1[:, j])
        FU1d_IC1 = U1d_in_IC1.flux()
        V2_IC1[:, :] -= IC1_vol * np.outer(FU1d_IC1, bais_val_x_wi_IC1[:, j])
    #求逆
    # M_IC0 = (bais_val_IC0 * wi) @ np.transpose(bais_val_IC0) * IC0_vol
    # M_IC1 = (bais_val_IC1 * wi) @ np.transpose(bais_val_IC1) * IC1_vol
    dUIC[:, :, 0] = (V1_IC0 + V2_IC0)
    dUIC[:, :, 1] = (V1_IC1 + V2_IC1)
    return dU,dUIC,max_speed

def rho_exact(x,t):
    return 1+0.2*np.sin(np.pi*(x-t))


#建立主函数的
if __name__ == '__main__':
    M_b = 1
    Nx = 320
    max_t = 0.5
    DIM = 1
    CFL = 0.1
    x_min = 0
    x_max = 2
    eps = 1e-12
    # 流体的参数信息
    max_speed = 0
    gamma = 1.4
    gamma0 = 1.4
    gamma1 = 1.4

    # 单元的几何信息
    point = np.linspace(x_min, x_max, Nx + 1)
    n_edge = np.shape(point)[0]
    n_ele = n_edge - 1
    ele_vol = (x_max - x_min) / n_ele
    # 单元之间的对应关系
    ele2edge = np.array([np.arange(0, n_edge - 1), np.arange(1, n_edge)])
    ele2edge = ele2edge.transpose()
    ele2point = ele2edge
    edge2cell = np.array([np.arange(-1, n_ele), np.arange(0, n_ele + 1)])
    edge2cell[-1, -1] = -1
    edge2cell = edge2cell.transpose()
    ele_centroid = 0.5 * (point[0:-1] + point[1:])
    # 积分点的信息
    W = 4
    x_int_ref, wi = roots_legendre(W)
    wi *= 0.5
    # 对数据进行初始化
    U_data0 = np.zeros([DIM+2,M_b,Nx],dtype=float)
    U_data1 = np.zeros([DIM+2,M_b,Nx],dtype=float)
    init_fluid_data(U_data0, U_data1, ele_centroid)
    '''
    界面相关的几何信息
    '''
    IC_idx = -1  # 界面所但在单元的指标
    IC_idx_new = -1
    point_label = -np.ones(n_edge, dtype=int)  # 顶点的标记
    ele_label = -np.ones(n_ele, dtype=int)  # 单元的标记
    # 存界面单元的值
    U_IC0 = np.zeros([DIM + 2, M_b], dtype=float)
    U_IC1 = np.zeros([DIM + 2, M_b], dtype=float)
    # 界面单元的体积和质心
    IC0_vol = 0
    IC1_vol = 0
    IC0_centroid = 0
    IC1_centroid = 0
    # 相体积和相重心
    phase_vol = np.zeros([2, n_ele], dtype=float)
    phase_centroid = np.zeros([2, n_ele], dtype=float)
    # 备份
    point_label_new = -np.ones(n_edge, dtype=int)
    ele_label_new = -np.ones(n_ele, dtype=int)
    IC0_vol_new = 0
    IC1_vol_new = 0
    IC0_centroid_new = 0
    IC1_centroid_new = 0
    phase_vol_new = np.zeros([2, n_ele], dtype=float)
    phase_centroid_new = np.zeros([2, n_ele], dtype=float)
    # 界面上的通量（移动边界上的通量）
    IC_flux = np.zeros(DIM + 2, dtype=float)
    # 多介质黎曼问题的参数
    tol = 1e-12
    maxit = 1000
    # 界面的初始位置 与 水平集函数的初值 phi
    x_I = 1.0
    phi = point - x_I

    #开始计时 准备运算
    t = 0
    is_stop = False
    iter = 0
    while (not is_stop):
        # 备份解
        U_data_cp0 = U_data0
        U_data_cp1 = U_data1
        # 确定界面的指标
        IC_idx = interface_cell_idx(phi)
        # 更新相信息
        update_phase_info(x_I)
        IC0_vol,IC1_vol,IC0_centroid,IC1_centroid=update_interface_info()
        # 计算界面单元的初值
        update_interface_val()
        # 求解多介质黎曼问题并得到通量
        u_star = oneD_Riemann(U_IC0[:, 0], U_IC1[:, 0])
        # 计算通量 演化方程
        max_speed = 0
        dU, dUIC, max_speed = get_flux(U_data0, U_data1, max_speed)
        # 计算时间步长
        dt = CFL * ele_vol / max_speed
        if (t + dt > max_t and t <= max_t):
            dt = max_t - t
            is_stop = True
        t += dt
        # 界面的位置发生了移动
        x_I_new = x_I + u_star * dt
        phi_new = point - x_I_new
        # 更新几何信息 需要备份啊
        IC_idx_new = interface_cell_idx_new(phi_new)
        update_phase_info_new(x_I_new)
        IC0_vol_new,IC1_vol_new,IC0_centroid_new,IC1_centroid_new = update_interface_info_new()
        # 更新变量 先考虑一阶格式
        U_data0[:, :, 0:IC_idx - 1] = U_data0[:, :, 0:IC_idx - 1] - dt * dU[:, :, 0:IC_idx - 1]
        U_data1[:, :, IC_idx + 2:] = U_data1[:, :, IC_idx + 2:] - dt * dU[:, :, IC_idx + 2:]
        # 更新新界面单元上的值
        x_int_IC0 = 0.5 * IC0_vol * x_int_ref + IC0_centroid
        x_int_IC0_new = 0.5 * IC0_vol_new * x_int_ref + IC0_centroid_new
        bais_val_IC0 = bais(x_int_IC0, IC0_centroid, IC0_vol)
        bais_val_IC0_new = bais(x_int_IC0_new, IC0_centroid_new, IC0_vol_new)
        x_int_IC1 = 0.5 * IC1_vol * x_int_ref + IC1_centroid
        x_int_IC1_new = 0.5 * IC1_vol_new * x_int_ref + IC1_centroid_new
        bais_val_IC1 = bais(x_int_IC1, IC1_centroid, IC1_vol)
        bais_val_IC1_new = bais(x_int_IC1_new, IC1_centroid_new, IC1_vol_new)

        M_IC0 = (bais_val_IC0 * wi) @ np.transpose(bais_val_IC0) * IC0_vol
        M_IC0_new = (bais_val_IC0_new * wi) @ np.transpose(bais_val_IC0_new) * IC0_vol_new
        M_IC1 = (bais_val_IC1 * wi) @ np.transpose(bais_val_IC1) * IC1_vol
        M_IC1_new = (bais_val_IC1_new * wi) @ np.transpose(bais_val_IC1_new) * IC1_vol_new
        U_IC0[:, :] = (U_IC0[:, :] * np.diagonal(M_IC0) - dt * dUIC[:, :, 0]) / np.diagonal(M_IC0_new)
        U_IC1[:, :] = (U_IC1[:, :] * np.diagonal(M_IC1) - dt * dUIC[:, :, 1]) / np.diagonal(M_IC1_new)
        # 重新分配守恒量 间断起见直接赋值吧
        for i in range(IC_idx - 1, IC_idx + 2):
            if phase_vol_new[0, i] > eps:
                U_data0[:, :, i] = U_IC0
            else:
                U_data0[:, :, i] = 0
            if phase_vol_new[1, i] > eps:
                U_data1[:, :, i] = U_IC1
            else:
                U_data1[:, :, i] = 0
        # 解已经完成了更新
        phi = phi_new
        x_I = x_I_new
        IC_idx = IC_idx_new
        if iter ==100:
            break
        iter += 1


x_int_ref, wi = roots_legendre(5)
wi *= 0.5
sumErrorL1 = 0
sumErrorLinf = 0
for i in range(n_ele):
    # 在积分点上计算误差 并求和
    x_int = 0.5 * ele_vol * x_int_ref + ele_centroid[i] # 积分点
    B = bais(x_int, ele_centroid[i], ele_vol)
    rho_ele0 = U_data0[0,:,i]@B
    rho_ele1 = U_data1[0,:,i]@B
    rho_exact_ele = rho_exact(x_int,t)
    temp1 = 0
    for j in range(len(x_int)):
        if x_I-x_int[j]>0:
            temp1 += np.abs(rho_ele0[j]-rho_exact_ele[j])*wi[j]*ele_vol
            sumErrorLinf = max(sumErrorLinf,np.abs(rho_ele0[j]-rho_exact_ele[j]))
        else:
            temp1 += np.abs(rho_ele1[j]-rho_exact_ele[j])*wi[j]*ele_vol
            sumErrorLinf = max(sumErrorLinf, np.abs(rho_ele1[j] - rho_exact_ele[j]))
    sumErrorL1 += temp1

print('sum_errorL1:'+str(sumErrorL1))
print('sum_errorLinf:'+str(sumErrorLinf))

#输出结果的展示
Bc = bais(ele_centroid, ele_centroid, ele_vol)
Rho0 = np.sum(Bc[0,:]*U_data0[0,:,:],axis=0)
Rho1 = np.sum(Bc[0,:]*U_data1[0,:,:],axis=0)
Rho = np.maximum(Rho0,Rho1)

plt.figure()
plt.plot(ele_centroid,rho_exact(ele_centroid,t),label='Exact')
plt.plot(ele_centroid,Rho,'*-',label='DG P2')
# plt.plot(ele_centroid,Rho1,'*-',label='DG P2')
plt.legend()
plt.show()
plt.clf()

# 需要再补充一个计算误差的函数
# 先看一阶精度的下的误差阶 近似有1阶精度 即可
# 然后考虑二阶近似 主要是在解的分配上

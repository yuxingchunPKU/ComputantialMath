import math
'''
一维原始变量的流体类型
状态方程为刚性气体状态方程
$$
p = (\gamma-1)\rho e - \gamma p_{\infty}
$$
内置了一个初始化和计算局部声速的函数
'''
class Fluid:
    rho: float
    u: float
    p: float
    gamma: float
    p_inf: float
    def __init__(self,rho,u,p,gamma,p_inf=0.0):
        '''
        :param rho:密度
        :param u: 速度
        :param p: 压强
        :param gamma: 热力学参数 1
        :param p_inf: 热力学参数 2 刚性气体
        '''
        self.rho = rho
        self.u = u
        self.p = p
        self.gamma = gamma
        self.p_inf = p_inf
    def a(self) -> float:
        '''
        :return: 声速
        '''
        return math.sqrt(self.gamma*(self.p+self.p_inf)/self.rho)


import math


class FluidReact:
    '''
    一维反应流中使用的流体类型（理想气体 + 单位质量化学反应热 q）

    状态量：
        rho   密度
        u     速度
        p     压强
        gamma 比热比
        q     单位质量的化学能（未反应气体 q>0，已反应气体 q=0）

    状态方程为理想气体状态方程
    $$
    p = (\gamma-1)\rho e
    $$
    声速为
    $$
    a = \sqrt{\gamma p/\rho}
    $$
    '''

    rho: float
    u: float
    p: float
    gamma: float
    q: float

    def __init__(self, rho, u, p, gamma=1.4, q=0.0):
        '''
        :param rho: 密度
        :param u: 速度
        :param p: 压强
        :param gamma: 比热比
        :param q: 单位质量化学能
        '''
        self.rho = rho
        self.u = u
        self.p = p
        self.gamma = gamma
        self.q = q

    def a(self) -> float:
        '''
        :return: 声速
        '''
        return math.sqrt(self.gamma * self.p / self.rho)

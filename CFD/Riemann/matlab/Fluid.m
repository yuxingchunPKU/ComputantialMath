classdef Fluid
    %FLUID 一维原始变量的流体类型（刚性气体状态方程）
    %
    %   状态方程
    %       p = (gamma - 1) * rho * e - gamma * p_inf
    %
    %   对应 Python 版本 code/Fluid.py。
    properties
        rho      % 密度
        u        % 速度
        p        % 压强
        gamma    % 热力学参数 1（比热比）
        p_inf    % 热力学参数 2（刚性气体压力）
    end

    methods
        function obj = Fluid(rho, u, p, gamma, p_inf)
            if nargin < 5
                p_inf = 0.0;
            end
            obj.rho = rho;
            obj.u = u;
            obj.p = p;
            obj.gamma = gamma;
            obj.p_inf = p_inf;
        end

        function a = sound_speed(obj)
            % 局部声速
            a = sqrt(obj.gamma * (obj.p + obj.p_inf) / obj.rho);
        end
    end
end

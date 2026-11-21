classdef FluidReact
    %FLUIDREACT 一维反应流中使用的流体类型（理想气体 + 单位质量化学反应热 q）
    %
    %   状态量：
    %       rho   密度
    %       u     速度
    %       p     压强
    %       gamma 比热比
    %       q     单位质量的化学能（未反应气体 q>0，已反应气体 q=0）
    %
    %   状态方程为理想气体状态方程
    %       p = (gamma - 1) * rho * e
    %   声速为
    %       a = sqrt(gamma * p / rho)
    %
    %   对应 Python 版本 code/FluidReact.py。
    properties
        rho
        u
        p
        gamma
        q
    end

    methods
        function obj = FluidReact(rho, u, p, gamma, q)
            if nargin < 4 || isempty(gamma)
                gamma = 1.4;
            end
            if nargin < 5 || isempty(q)
                q = 0.0;
            end
            obj.rho = rho;
            obj.u = u;
            obj.p = p;
            obj.gamma = gamma;
            obj.q = q;
        end

        function a = sound_speed(obj)
            a = sqrt(obj.gamma * obj.p / obj.rho);
        end
    end
end

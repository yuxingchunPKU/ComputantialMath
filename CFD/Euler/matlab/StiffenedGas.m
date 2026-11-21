classdef StiffenedGas
    %STIFFENEDGAS 常 gamma、常 p_inf 的刚性气体状态方程
    %
    %   守恒变量 U = [rho; rho*u; E]，状态方程
    %       p = (gamma - 1) * (E - 0.5 * rho * u^2) - gamma * p_inf
    %
    %   输入 U 为 2 维数组（3 x n），各方法按列广播。
    %   对应 Python 版本 euler_unified/eos.py。
    properties
        gamma
        p_inf
    end

    methods
        function obj = StiffenedGas(gamma, p_inf)
            if nargin < 1 || isempty(gamma)
                gamma = 1.4;
            end
            if nargin < 2 || isempty(p_inf)
                p_inf = 0.0;
            end
            if gamma <= 1.0
                error('StiffenedGas:gamma', 'gamma must be greater than 1');
            end
            obj.gamma = gamma;
            obj.p_inf = p_inf;
        end

        function [rho, u, p] = primitive(obj, U)
            % 由守恒变量计算原始变量 (rho, u, p)
            rho = max(U(1, :), 1.0e-14);
            u = U(2, :) ./ rho;
            p = (obj.gamma - 1.0) * (U(3, :) - 0.5 * rho .* u .* u) ...
                - obj.gamma * obj.p_inf;
            p = max(p, 1.0e-14);
        end

        function U = conservative(obj, rho, u, p)
            % 由原始变量计算守恒变量
            E = (p + obj.gamma * obj.p_inf) / (obj.gamma - 1.0) ...
                + 0.5 * rho .* u .* u;
            U = [rho; rho .* u; E];
        end

        function a = sound_speed(obj, U)
            [rho, ~, p] = obj.primitive(U);
            a = sqrt(obj.gamma * (p + obj.p_inf) ./ rho);
        end

        function H = specific_enthalpy(obj, U)
            [rho, ~, p] = obj.primitive(U);
            H = (U(3, :) + p) ./ rho;
        end

        function s = max_speed(obj, U)
            [~, u, ~] = obj.primitive(U);
            a = obj.sound_speed(U);
            s = max(abs(u(:)) + a(:));
        end

        function F = flux(obj, U)
            % 物理 Euler 通量
            [~, u, p] = obj.primitive(U);
            F = zeros(size(U));
            F(1, :) = U(2, :);
            F(2, :) = U(2, :) .* u + p;
            F(3, :) = (U(3, :) + p) .* u;
        end
    end
end

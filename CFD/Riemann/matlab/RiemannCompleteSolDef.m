classdef RiemannCompleteSolDef < handle
    %RIEMANNCOMPLETESOLDEF 爆燃波黎曼问题的完全解
    %
    %   从左到右的波系为
    %       左端状态 WL -> 非反应波 -> 接触间断 -> 爆燃波 ->
    %       状态 0 -> 非反应波 -> 右端状态 WR
    %   其中求解器给出的是状态 1（火焰波后、接触间断一侧）与状态 0（火焰波前）。
    %   对应 Python 版本 code/Exact_Riemann_Combustion.py 中的
    %   RiemannCompleteSolDef 类。
    properties
        p_star
        u_star
        p0
        rho0
        WL
        WR
        x_L
        x_R
        x_c
        N
        t_final
        quiet
        V
        rho1
        u0
    end

    methods
        function obj = RiemannCompleteSolDef(p_star, u_star, p0, rho0, WL, WR, ...
                                             t_final, x_L, x_R, x_c, N, quiet, ...
                                             V, rho1, u0)
            if nargin < 8 || isempty(x_L), x_L = -0.5; end
            if nargin < 9 || isempty(x_R), x_R = 0.5; end
            if nargin < 10 || isempty(x_c), x_c = 0.0; end
            if nargin < 11 || isempty(N), N = 200; end
            if nargin < 12 || isempty(quiet), quiet = true; end
            if nargin < 13, V = []; end
            if nargin < 14, rho1 = []; end
            if nargin < 15, u0 = []; end
            obj.p_star = p_star;
            obj.u_star = u_star;
            obj.p0 = p0;
            obj.rho0 = rho0;
            obj.WL = WL;
            obj.WR = WR;
            obj.t_final = t_final;
            obj.x_L = x_L;
            obj.x_R = x_R;
            obj.x_c = x_c;
            obj.N = N;
            obj.quiet = quiet;
            obj.V = V;
            obj.rho1 = rho1;
            obj.u0 = u0;
        end

        function u0 = state0_velocity(obj)
            % 由火焰波前压强 p0 和右端状态 WR 计算 u0
            WR = obj.WR;
            mu2 = (WR.gamma - 1) / (WR.gamma + 1);
            if obj.p0 >= WR.p
                phi = (obj.p0 - WR.p) * ...
                      sqrt((1 - mu2) / (WR.rho * (obj.p0 + mu2 * WR.p)));
            else
                phi = 2 * WR.sound_speed() / (WR.gamma - 1) * ...
                      ((obj.p0 / WR.p) ^ ((WR.gamma - 1) / (2 * WR.gamma)) - 1);
            end
            u0 = WR.u + phi;
        end

        function [x, rho, u, p] = complete_solution(obj)
            if isempty(obj.V)
                error('RiemannCompleteSolDef:V', ...
                      'RiemannCompleteSolDef 需要爆燃波速度 V');
            end
            x = linspace(obj.x_L, obj.x_R, obj.N + 1);
            rho = obj.WL.rho * ones(size(x));
            u = obj.WL.u * ones(size(x));
            p = obj.WL.p * ones(size(x));

            si = (x - obj.x_c) / obj.t_final;
            S = obj.u_star;                           % 接触间断速度
            p1 = obj.p_star;
            if isempty(obj.u0)
                u0 = obj.state0_velocity();
            else
                u0 = obj.u0;
            end
            if isempty(obj.rho1)
                rho1 = obj.rho0 * (obj.V - u0) / (obj.V - obj.u_star);
            else
                rho1 = obj.rho1;
            end
            if ~(obj.V > S)
                error('RiemannCompleteSolDef:order', ...
                      '爆燃波速度 V 必须大于接触间断速度 u*，否则波系顺序不成立');
            end

            WL = obj.WL;
            gL = WL.gamma;
            mu2L = (gL - 1) / (gL + 1);

            % ---------------- 左侧非反应波 ----------------
            if p1 >= WL.p
                % 激波
                rho_star_L = WL.rho * (p1 + mu2L * WL.p) / (WL.p + mu2L * p1);
                if abs(rho_star_L - WL.rho) > 1e-14 * WL.rho
                    S_L = (rho_star_L * obj.u_star - WL.rho * WL.u) / ...
                          (rho_star_L - WL.rho);
                else
                    % 零强度波
                    S_L = WL.u - WL.sound_speed();
                end
                m = (si > S_L) & (si <= S);
                rho(m) = rho_star_L;
                u(m) = obj.u_star;
                p(m) = p1;
                if ~obj.quiet
                    fprintf('Left Shock Wave\n');
                    fprintf('rho_star_L %.16f\n', rho_star_L);
                    fprintf('S_L %.16f\n', S_L);
                end
            else
                % 稀疏波
                aL = WL.sound_speed();
                rho_star_L = WL.rho * (p1 / WL.p) ^ (1 / gL);
                a_star_L = aL * (p1 / WL.p) ^ ((gL - 1) / (2 * gL));
                S_HL = WL.u - aL;
                S_TL = obj.u_star - a_star_L;

                m = (si > S_HL) & (si <= S_TL);
                base = 2 / (gL + 1) + (gL - 1) * (WL.u - si(m)) / ((gL + 1) * aL);
                rho(m) = WL.rho * base .^ (2 / (gL - 1));
                u(m) = 2 * (aL + (gL - 1) * WL.u / 2 + si(m)) / (gL + 1);
                p(m) = WL.p * base .^ (2 * gL / (gL - 1));

                m = (si > S_TL) & (si <= S);
                rho(m) = rho_star_L;
                u(m) = obj.u_star;
                p(m) = p1;
                if ~obj.quiet
                    fprintf('Left Rarefaction Wave\n');
                    fprintf('rho_star_L %.16f\n', rho_star_L);
                    fprintf('S_HL %.16f\n', S_HL);
                    fprintf('S_TL %.16f\n', S_TL);
                end
            end

            % ---------------- 接触间断与火焰之间的状态 1 ----------------
            m = (si > S) & (si < obj.V);
            rho(m) = rho1;
            u(m) = obj.u_star;
            p(m) = p1;

            % ---------------- 火焰波后的状态 0 与右侧非反应波 ----------------
            WR = obj.WR;
            gR = WR.gamma;
            aR = WR.sound_speed();
            if obj.p0 >= WR.p
                % 激波
                tau_R = 1 / WR.rho;
                tau_0 = 1 / obj.rho0;
                if abs(tau_R - tau_0) > 1e-14 * tau_R
                    S_R0 = WR.u + sqrt((obj.p0 - WR.p) / (tau_R - tau_0)) / WR.rho;
                else
                    S_R0 = WR.u + WR.sound_speed();
                end
                if ~(S_R0 > obj.V)
                    error('RiemannCompleteSolDef:order', ...
                          '右侧非反应波速度必须大于爆燃波速度 V，否则波系顺序不成立');
                end
                m = (si >= obj.V) & (si < S_R0);
                rho(m) = obj.rho0;
                u(m) = u0;
                p(m) = obj.p0;
                m = (si >= S_R0);
                rho(m) = WR.rho;
                u(m) = WR.u;
                p(m) = WR.p;
                if ~obj.quiet
                    fprintf('Right Non-reactive Shock Wave\n');
                    fprintf('rho0 %.16f\n', obj.rho0);
                    fprintf('S_R0 %.16f\n', S_R0);
                end
            else
                % 稀疏波
                a0 = sqrt(gR * obj.p0 / obj.rho0);
                S_TR0 = u0 + a0;
                S_HR = WR.u + aR;
                if ~(S_TR0 > obj.V)
                    error('RiemannCompleteSolDef:order', ...
                          '右侧非反应波速度必须大于爆燃波速度 V，否则波系顺序不成立');
                end
                m = (si >= obj.V) & (si < S_TR0);
                rho(m) = obj.rho0;
                u(m) = u0;
                p(m) = obj.p0;

                m = (si >= S_TR0) & (si < S_HR);
                base = 2 / (gR + 1) - (gR - 1) * (WR.u - si(m)) / ((gR + 1) * aR);
                rho(m) = WR.rho * base .^ (2 / (gR - 1));
                u(m) = 2 * (-aR + (gR - 1) * WR.u / 2 + si(m)) / (gR + 1);
                p(m) = WR.p * base .^ (2 * gR / (gR - 1));

                m = (si >= S_HR);
                rho(m) = WR.rho;
                u(m) = WR.u;
                p(m) = WR.p;
                if ~obj.quiet
                    fprintf('Right Non-reactive Rarefaction Wave\n');
                    fprintf('rho0 %.16f\n', obj.rho0);
                    fprintf('S_TR0 %.16f\n', S_TR0);
                    fprintf('S_HR %.16f\n', S_HR);
                end
            end

            if ~obj.quiet
                fprintf('Deflagration Wave\n');
                fprintf('rho1 %.16f\n', rho1);
                fprintf('V %.16f\n', obj.V);
            end
        end
    end
end

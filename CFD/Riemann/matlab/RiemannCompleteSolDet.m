classdef RiemannCompleteSolDet < handle
    %RIEMANNCOMPLETESOLDET 含爆轰波的黎曼问题完全解
    %
    %   从左到右的波系为：左端状态 WL -> 非反应波 -> 接触间断 ->
    %   爆轰波（强爆轰或 CJ 爆轰 + 稀疏波）-> 右端状态 WR。
    %   对应 Python 版本 code/Exact_Riemann_Combustion.py 中的
    %   RiemannCompleteSolDet 类。
    properties
        p_star
        u_star
        WL
        WR
        x_L
        x_R
        x_c
        N
        t_final
        quiet
    end

    methods
        function obj = RiemannCompleteSolDet(p_star, u_star, WL, WR, t_final, ...
                                             x_L, x_R, x_c, N, quiet)
            if nargin < 6 || isempty(x_L)
                x_L = -0.5;
            end
            if nargin < 7 || isempty(x_R)
                x_R = 0.5;
            end
            if nargin < 8 || isempty(x_c)
                x_c = 0.0;
            end
            if nargin < 9 || isempty(N)
                N = 200;
            end
            if nargin < 10 || isempty(quiet)
                quiet = true;
            end
            obj.p_star = p_star;
            obj.u_star = u_star;
            obj.WL = WL;
            obj.WR = WR;
            obj.t_final = t_final;
            obj.x_L = x_L;
            obj.x_R = x_R;
            obj.x_c = x_c;
            obj.N = N;
            obj.quiet = quiet;
        end

        function [x, rho, u, p] = complete_solution(obj)
            x = linspace(obj.x_L, obj.x_R, obj.N + 1);
            rho = zeros(size(x));
            u = zeros(size(x));
            p = zeros(size(x));

            WL = obj.WL;
            WR = obj.WR;
            ps = obj.p_star;
            us = obj.u_star;
            S = us;                                   % 接触间断速度
            si = (x - obj.x_c) / obj.t_final;

            ers = ExactRiemannSolverDetonation(obj.WL, obj.WR);
            [rho_cj, u_cj, p_cj] = ers.CJ_data();

            % ---------------- 左侧非反应波 ----------------
            if ps >= WL.p
                g = WL.gamma;
                mu2 = (g - 1) / (g + 1);
                p0 = WL.p;
                rho0 = WL.rho;
                u0 = WL.u;
                rho_star_L = rho0 * (ps + mu2 * p0) / (p0 + mu2 * ps);
                S_L = (rho_star_L * us - rho0 * u0) / (rho_star_L - rho0);

                if ~obj.quiet
                    fprintf('Left Shock Wave\n');
                    fprintf('rho_star_L %.16f\n', rho_star_L);
                    fprintf('S_L %.16f\n', S_L);
                end

                m = (si <= S_L);
                rho(m) = rho0;
                u(m) = u0;
                p(m) = p0;
                m = (si > S_L) & (si <= S);
                rho(m) = rho_star_L;
                u(m) = us;
                p(m) = ps;
            else
                g = WL.gamma;
                rho0 = WL.rho;
                u0 = WL.u;
                p0 = WL.p;
                a0 = WL.sound_speed();
                rho_star_L = rho0 * (ps / p0) ^ (1 / g);
                a_star_L = a0 * (ps / p0) ^ ((g - 1) / (2 * g));
                S_HL = u0 - a0;
                S_TL = us - a_star_L;

                if ~obj.quiet
                    fprintf('Left Rarefaction Wave\n');
                    fprintf('rho_star_L %.16f\n', rho_star_L);
                    fprintf('S_HL %.16f\n', S_HL);
                    fprintf('S_TL %.16f\n', S_TL);
                end

                m = (si <= S_HL);
                rho(m) = rho0;
                u(m) = u0;
                p(m) = p0;

                m = (si > S_HL) & (si <= S_TL);
                base = 2 / (g + 1) + (g - 1) * (u0 - si(m)) / ((g + 1) * a0);
                rho(m) = rho0 * base .^ (2 / (g - 1));
                u(m) = 2 * (a0 + (g - 1) * u0 / 2 + si(m)) / (g + 1);
                p(m) = p0 * base .^ (2 * g / (g - 1));

                m = (si > S_TL) & (si <= S);
                rho(m) = rho_star_L;
                u(m) = us;
                p(m) = ps;
            end

            % ---------------- 右侧燃烧波 ----------------
            if ps >= p_cj
                % 强爆轰分支
                g = WR.gamma;
                p0 = WR.p;
                rho0 = WR.rho;
                u0 = WR.u;
                M = -(ps - p0) / (us - u0);
                S_R = u0 - M / rho0;
                rho_star_R = M / (us - S_R);

                if ~obj.quiet
                    fprintf('Right Strong Detonation Wave\n');
                    fprintf('rho_star_R %.16f\n', rho_star_R);
                    fprintf('S_Detonation %.16f\n', S_R);
                end

                m = (si >= S_R);
                rho(m) = rho0;
                u(m) = u0;
                p(m) = p0;
                m = (si < S_R) & (si >= S);
                rho(m) = rho_star_R;
                u(m) = us;
                p(m) = ps;
            else
                % CJ 爆轰分支：激波 + 稀疏波
                g = WR.gamma;
                rho0 = WR.rho;
                u0 = WR.u;
                p0 = WR.p;
                rho_star_R = rho_cj * (ps / p_cj) ^ (1 / g);
                a_cj = sqrt(g * p_cj / rho_cj);
                a_star_R = a_cj * (ps / p_cj) ^ ((g - 1) / (2 * g));
                S_TR = us + a_star_R;
                S_HR = u_cj + a_cj;

                if ~obj.quiet
                    fprintf('Right CJ Detonation Wave\n');
                    fprintf('rho_star_R %.16f\n', rho_star_R);
                    fprintf('S_TR %.16f\n', S_TR);
                    fprintf('S_HR %.16f\n', S_HR);
                end

                m = (si <= S_TR) & (si > S);
                rho(m) = rho_star_R;
                u(m) = us;
                p(m) = ps;

                m = (si <= S_HR) & (si > S_TR);
                base = 2 / (g + 1) - (g - 1) * (u_cj - si(m)) / ((g + 1) * a_cj);
                rho(m) = rho_cj * base .^ (2 / (g - 1));
                u(m) = 2 * (-a_cj + (g - 1) * u_cj / 2 + si(m)) / (g + 1);
                p(m) = p_cj * base .^ (2 * g / (g - 1));

                m = (si > S_HR);
                rho(m) = rho0;
                u(m) = u0;
                p(m) = p0;
            end
        end
    end
end

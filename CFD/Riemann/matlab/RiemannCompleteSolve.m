classdef RiemannCompleteSolve
    %RIEMANNCOMPLETESOLVE 激波管问题的完全解（刚性气体）
    %
    %   给定中间状态 (p_star, u_star)，在均匀网格上采样精确解。
    %   对应 Python 版本 code/Exact_Riemann_complete_solve.py。
    properties
        p_star
        u_star
        WL
        WR
        x_L
        x_R
        N
        t_final
    end

    methods
        function obj = RiemannCompleteSolve(p_star, u_star, WL, WR, t_final, x_L, x_R, N)
            if nargin < 6 || isempty(x_L)
                x_L = -0.5;
            end
            if nargin < 7 || isempty(x_R)
                x_R = 0.5;
            end
            if nargin < 8 || isempty(N)
                N = 200;
            end
            obj.p_star = p_star;
            obj.u_star = u_star;
            obj.WL = WL;
            obj.WR = WR;
            obj.t_final = t_final;
            obj.x_L = x_L;
            obj.x_R = x_R;
            obj.N = N;
        end

        function [x, rho, u, p] = solve(obj)
            x = linspace(obj.x_L, obj.x_R, obj.N + 1);
            rho = zeros(size(x));
            u = zeros(size(x));
            p = zeros(size(x));

            WL = obj.WL;
            WR = obj.WR;
            ps = obj.p_star;
            us = obj.u_star;
            S = us;                                   % 接触间断速度
            si = x / obj.t_final;

            % ---------------- 左侧：激波或稀疏波 ----------------
            if ps >= WL.p
                part1 = 2 * WL.gamma * WL.p_inf + (WL.gamma + 1) * ps + ...
                        (WL.gamma - 1) * WL.p;
                part2 = 2 * (WL.p + WL.gamma * WL.p_inf) + ...
                        (WL.gamma - 1) * ps + (WL.gamma - 1) * WL.p;
                rho_star_L = WL.rho * part1 / part2;
                A_L = 2 / ((WL.gamma + 1) * WL.rho);
                B_L = (WL.gamma - 1) * (WL.p + WL.p_inf) / (WL.gamma + 1);
                S_L = WL.u - (1 / WL.rho) * sqrt((B_L + WL.p_inf + ps) / A_L);

                m = (si <= S_L);
                rho(m) = WL.rho;
                u(m) = WL.u;
                p(m) = WL.p;
                m = (si > S_L) & (si <= S);
                rho(m) = rho_star_L;
                u(m) = us;
                p(m) = ps;
            else
                g = WL.gamma;
                aL = WL.sound_speed();
                rho_star_L = WL.rho * ((ps + WL.p_inf) / (WL.p + WL.p_inf)) ^ (1 / g);
                aL_star = aL * ((ps + WL.p_inf) / (WL.p + WL.p_inf)) ^ ...
                          ((g - 1) / (2 * g));
                S_HL = WL.u - aL;
                S_TL = us - aL_star;

                m = (si <= S_HL);
                rho(m) = WL.rho;
                u(m) = WL.u;
                p(m) = WL.p;

                m = (si > S_HL) & (si <= S_TL);
                base = 2 / (g + 1) + (g - 1) * (WL.u - si(m)) / ((g + 1) * aL);
                rho(m) = WL.rho * base .^ (2 / (g - 1));
                u(m) = 2 * (aL + (g - 1) * WL.u / 2 + si(m)) / (g + 1);
                p(m) = (WL.p + WL.p_inf) * base .^ (2 * g / (g - 1)) - WL.p_inf;

                m = (si > S_TL) & (si <= S);
                rho(m) = rho_star_L;
                u(m) = us;
                p(m) = ps;
            end

            % ---------------- 右侧：激波或稀疏波 ----------------
            if ps >= WR.p
                part1 = 2 * WR.gamma * WR.p_inf + (WR.gamma + 1) * ps + ...
                        (WR.gamma - 1) * WR.p;
                part2 = 2 * (WR.p + WR.gamma * WR.p_inf) + ...
                        (WR.gamma - 1) * ps + (WR.gamma - 1) * WR.p;
                rho_star_R = WR.rho * part1 / part2;
                A_R = 2 / ((WR.gamma + 1) * WR.rho);
                B_R = (WR.gamma - 1) * (WR.p + WR.p_inf) / (WR.gamma + 1);
                S_R = WR.u + (1 / WR.rho) * sqrt((B_R + WR.p_inf + ps) / A_R);

                m = (si > S) & (si <= S_R);
                rho(m) = rho_star_R;
                u(m) = us;
                p(m) = ps;
                m = (si > S_R);
                rho(m) = WR.rho;
                u(m) = WR.u;
                p(m) = WR.p;
            else
                g = WR.gamma;
                aR = WR.sound_speed();
                rho_star_R = WR.rho * ((ps + WR.p_inf) / (WR.p + WR.p_inf)) ^ (1 / g);
                aR_star = aR * ((ps + WR.p_inf) / (WR.p + WR.p_inf)) ^ ...
                          ((g - 1) / (2 * g));
                S_HR = WR.u + aR;
                S_TR = us + aR_star;

                m = (si <= S_TR) & (si > S);
                rho(m) = rho_star_R;
                u(m) = us;
                p(m) = ps;

                m = (si <= S_HR) & (si > S_TR);
                base = 2 / (g + 1) - (g - 1) * (WR.u - si(m)) / ((g + 1) * aR);
                rho(m) = WR.rho * base .^ (2 / (g - 1));
                u(m) = 2 * (-aR + (g - 1) * WR.u / 2 + si(m)) / (g + 1);
                p(m) = (WR.p + WR.p_inf) * base .^ (2 * g / (g - 1)) - WR.p_inf;

                m = (si > S_HR);
                rho(m) = WR.rho;
                u(m) = WR.u;
                p(m) = WR.p;
            end
        end
    end
end

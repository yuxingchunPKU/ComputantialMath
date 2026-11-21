classdef ExactRiemannSolverDeflagration < handle
    %EXACTRIEMANNSOLVERDEFLAGRATION 含爆燃波的一维 Euler 精确 Riemann 求解器
    %
    %   右端为未燃气体（q>0），火焰速度满足
    %       V = u0 + K * (p0 / rho0)^Q
    %   火焰波前为状态 0，波后为状态 1（接触间断一侧）。
    %   对应 Python 版本 code/Exact_Riemann_Combustion.py 中的
    %   ExactRiemannSolverDeflagration 类。
    properties
        WL
        WR
        tol
        maxit
        p_star
        u_star
        p0
        rho0
        rho1
        u0
        p_cj
        u_cj
        rho_cj
        quiet
        dq
        V
        K
        Q
    end

    methods
        function obj = ExactRiemannSolverDeflagration(WL, WR, K, Q, tol, maxit, quiet)
            if nargin < 4 || isempty(Q)
                Q = 2;
            end
            if nargin < 5 || isempty(tol)
                tol = 1e-8;
            end
            if nargin < 6 || isempty(maxit)
                maxit = 10000;
            end
            if nargin < 7 || isempty(quiet)
                quiet = true;
            end
            obj.WL = WL;
            obj.WR = WR;
            obj.K = K;
            obj.Q = Q;
            obj.tol = tol;
            obj.maxit = maxit;
            obj.quiet = quiet;
            obj.dq = WL.q - WR.q;
        end

        function CJ_status(obj, W0)
            % 计算 CJ 状态（保留接口，供其它波系分析使用）
            dq = obj.dq;
            p0 = W0.p;
            rho0 = W0.rho;
            u0 = W0.u;
            g = W0.gamma;
            mu2 = (g - 1) / (g + 1);

            t1 = 2 * g * p0 / ((g ^ 2 - 1) * rho0 * dq);
            t2 = 1 - sqrt(1 - t1);
            pcj = p0 - (g - 1) * rho0 * dq * t2;
            rhocj = rho0 * (pcj * (1 + g) - p0) / (g * pcj);
            tau = 1 / rho0;
            t3 = ((1 - mu2) * tau - 2 * mu2 * dq / (p0 - pcj)) / (mu2 * p0 + pcj);
            ucj = u0 + (pcj - p0) * sqrt(t3);

            obj.p_cj = pcj;
            obj.rho_cj = rhocj;
            obj.u_cj = ucj;
        end

        function f = f_L_p(obj, p)
            % 左侧非反应波的特征函数
            W = obj.WL;
            p0 = W.p;
            tau0 = 1 / W.rho;
            g = W.gamma;
            mu2 = (g - 1) / (g + 1);
            if p >= p0
                A = (1 - mu2) * tau0;
                B = mu2 * p0;
                f = (p - p0) * sqrt(A / (p + B));
            else
                a0 = W.sound_speed();
                idx = (g - 1) / (2 * g);
                f = 2 * a0 / (g - 1) * ((p / p0) ^ idx - 1);
            end
        end

        function V = deflagration_speed(obj, rho0, u0, p0)
            % 爆燃波速度 V = u0 + K * (p0 / rho0)^Q
            V = u0 + obj.K * (p0 / rho0) ^ obj.Q;
        end

        function [rho0, u0] = state0_from_p0(obj, p0)
            % 由火焰前方压强 p0 计算状态 0（火焰波前）的密度和速度
            WR = obj.WR;
            rhoR = WR.rho;
            uR = WR.u;
            pR = WR.p;
            g = WR.gamma;
            mu2 = (g - 1) / (g + 1);
            if p0 >= pR
                % 激波分支
                rho0 = rhoR * (p0 + mu2 * pR) / (mu2 * p0 + pR);
                phi = (p0 - pR) * sqrt((1 - mu2) / (rhoR * (p0 + mu2 * pR)));
            else
                % 稀疏波分支
                rho0 = rhoR * (p0 / pR) ^ (1 / g);
                phi = 2 * WR.sound_speed() / (g - 1) * ...
                      ((p0 / pR) ^ ((g - 1) / (2 * g)) - 1);
            end
            u0 = uR + phi;
        end

        function [rho0, u0, V, p1, u1] = state1_from_p0(obj, p0)
            % 由火焰前方压强 p0 计算火焰波后的状态 1
            [rho0, u0] = obj.state0_from_p0(p0);
            V = obj.deflagration_speed(rho0, u0, p0);
            p1 = obj.compute_p1(rho0, u0, p0);
            tau0 = 1 / rho0;
            gR = obj.WR.gamma;
            mu2 = (gR - 1) / (gR + 1);
            phi_def = (p1 - p0) * sqrt(((1 - mu2) * tau0 - ...
                      2 * mu2 * obj.dq / (p0 - p1)) / (mu2 * p0 + p1));
            u1 = u0 + phi_def;
        end

        function f = f_R_p(obj, p)
            % 右波（非反应波 + 爆燃波）两侧的速度差 u1 - uR
            [~, ~, ~, ~, u1] = obj.state1_from_p0(p);
            f = u1 - obj.WR.u;
        end

        function p1 = compute_p1(obj, rho0, u0, p0)
            g = obj.WR.gamma;
            mu2 = (g - 1) / (g + 1);
            V = obj.deflagration_speed(rho0, u0, p0);
            tau0 = 1 / rho0;
            c0 = sqrt(g * p0 * tau0);
            v0 = u0 - V;
            M2 = (v0 / c0) ^ 2;
            p_ratio1 = (1 - mu2) * (1 + g * M2);
            p_ratio2 = sqrt(((1 + mu2) ^ 2) * ((1 - M2) ^ 2) + ...
                            8 * mu2 * (g ^ 2) * M2 * obj.dq / (c0 ^ 2));
            p1 = p0 * (p_ratio1 + p_ratio2) / 2;
        end

        function f = f_p(obj, p)
            % 匹配方程；求解变量 p 是火焰前方压强 p0
            [~, ~, ~, p1, ~] = obj.state1_from_p0(p);
            f = obj.f_L_p(p1) + obj.f_R_p(p) + obj.WR.u - obj.WL.u;
        end

        function p = p_init(obj)
            p = obj.WR.p;
        end

        function [a, b] = find_interval(obj, p0)
            p1 = p0;
            if obj.f_p(p0) <= 0
                for k = 1:200
                    if obj.f_p(p1) > 0
                        a = p0;
                        b = p1;
                        return;
                    end
                    p1 = 2 * p1;
                end
            else
                for k = 1:200
                    if obj.f_p(p1) <= 0
                        a = p1;
                        b = p0;
                        return;
                    end
                    p1 = 0.5 * p1;
                end
            end
            error('ExactRiemannSolverDeflagration:interval', '无法找到根的存在区间');
        end

        function [p_star, u_star, p0, rho0, V] = solver_brent(obj)
            p0 = obj.p_init();
            [a, b] = obj.find_interval(p0);
            fa = obj.f_p(a);
            fb = obj.f_p(b);
            if fa * fb > 0
                error('ExactRiemannSolverDeflagration:brent', ...
                      'Brent 方法失败：区间端点函数值同号');
            end
            if abs(fa) < abs(fb)
                tmp = a; a = b; b = tmp;
                tmp = fa; fa = fb; fb = tmp;
            end
            c = a;
            fc = fa;
            d = 0;
            mflag = true;
            for k = 1:obj.maxit
                if abs(b - a) / abs(a) < obj.tol
                    break;
                end
                if fb == 0
                    break;
                end
                if fa ~= fc && fb ~= fc
                    s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + ...
                        (b * fa * fc) / ((fb - fa) * (fb - fc)) + ...
                        (c * fa * fb) / ((fc - fa) * (fc - fb));
                else
                    s = b - fb * (b - a) / (fb - fa);
                end
                use_bisection = (s < 0.25 * (3 * a + b)) || (s > b) || ...
                    (mflag && abs(s - b) >= abs(b - c) / 2) || ...
                    (~mflag && abs(s - b) >= abs(c - d) / 2) || ...
                    (mflag && abs(b - c) < obj.tol) || ...
                    (~mflag && abs(c - d) < obj.tol);
                if use_bisection
                    s = 0.5 * (a + b);
                    mflag = true;
                else
                    mflag = false;
                end

                fs = obj.f_p(s);
                d = c;
                c = b;
                fc = fb;
                if fa * fs < 0
                    b = s;
                    fb = fs;
                else
                    a = s;
                    fa = fs;
                end
                if abs(fa) < abs(fb)
                    tmp = a; a = b; b = tmp;
                    tmp = fa; fa = fb; fb = tmp;
                end
            end

            obj.p0 = b;
            [rho0, u0, V, p1, u1] = obj.state1_from_p0(b);
            obj.rho0 = rho0;
            obj.u0 = u0;
            obj.V = V;
            obj.p_star = p1;
            obj.u_star = u1;
            % 由质量守恒得到火焰波后（中间状态）的密度
            obj.rho1 = rho0 * (V - u0) / (V - u1);

            if ~obj.quiet
                fprintf('p_star %.16f\n', obj.p_star);
                fprintf('u_star %.16f\n', obj.u_star);
                fprintf('p0 %.16f\n', obj.p0);
                fprintf('rho0 %.16f\n', obj.rho0);
                fprintf('rho1 %.16f\n', obj.rho1);
                fprintf('V %.16f\n', obj.V);
            end

            p_star = obj.p_star;
            u_star = obj.u_star;
            p0 = obj.p0;
            V = obj.V;
        end

        function data = solver_data(obj)
            % 求解得到的波和状态数据
            data = struct('p_star', obj.p_star, 'u_star', obj.u_star, ...
                          'p0', obj.p0, 'rho0', obj.rho0, 'u0', obj.u0, ...
                          'rho1', obj.rho1, 'V', obj.V);
        end
    end
end

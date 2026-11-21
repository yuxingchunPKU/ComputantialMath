classdef ExactRiemannSolver
    %EXACTRIEMANNSOLVER 刚性气体黎曼问题的精确求解（牛顿迭代 + Brent 兜底）
    %
    %   求解匹配方程  f_L(p) + f_R(p) + u_R - u_L = 0。
    %   对应 Python 版本 code/Exact_Riemann_solver.py。
    properties
        WL
        WR
        tol
        maxit
        p_star
        u_star
    end

    methods
        function obj = ExactRiemannSolver(WL, WR, tol, maxit)
            if nargin < 3 || isempty(tol)
                tol = 1e-12;
            end
            if nargin < 4 || isempty(maxit)
                maxit = 10000;
            end
            obj.WL = WL;
            obj.WR = WR;
            obj.tol = tol;
            obj.maxit = maxit;
        end

        function f = f_K_p(obj, p, W)
            % 计算 f_K(p)，K = L/R
            if p >= W.p
                A_K = 2 / ((W.gamma + 1) * W.rho);
                B_K = (W.gamma - 1) * (W.p + W.p_inf) / (W.gamma + 1);
                f = (p - W.p) * sqrt(A_K / (B_K + p + W.p_inf));
            else
                f = (2 * W.sound_speed() / (W.gamma - 1)) * ...
                    (((p + W.p_inf) / (W.p + W.p_inf)) ^ ...
                     ((W.gamma - 1) / (2 * W.gamma)) - 1);
            end
        end

        function f = f_prime_K_p(obj, p, W)
            % f_K(p) 关于 p 的导数
            if p >= W.p
                A_K = 2 / ((W.gamma + 1) * W.rho);
                B_K = (W.gamma - 1) * (W.p + W.p_inf) / (W.gamma + 1);
                part1 = sqrt(A_K / (B_K + p + W.p_inf));
                part2 = 1 - (p - W.p) / (2 * (B_K + p + W.p_inf));
                f = part1 * part2;
            else
                part1 = ((p + W.p_inf) / (W.p + W.p_inf)) ^ ...
                        (-(W.gamma + 1) / (2 * W.gamma));
                part2 = 1 / (W.rho * W.sound_speed());
                f = part1 * part2;
            end
        end

        function p = p_init(obj)
            % PVRS 估计；p_inf 很大导致估计值非正时退回算术平均
            p_pv = 0.5 * (obj.WL.p + obj.WR.p) - ...
                   0.125 * (obj.WR.u - obj.WL.u) * ...
                   (obj.WR.rho + obj.WL.rho) * ...
                   (obj.WR.sound_speed() + obj.WL.sound_speed());
            if p_pv <= obj.tol
                p_pv = 0.5 * (obj.WL.p + obj.WR.p);
            end
            p = max(obj.tol, p_pv);
        end

        function f = f_p(obj, p)
            % 黎曼问题的匹配方程 f(p)
            f = obj.f_K_p(p, obj.WL) + obj.f_K_p(p, obj.WR) + ...
                obj.WR.u - obj.WL.u;
        end

        function u = u_star_from_p(obj, p_star)
            % 由中间压强计算接触间断速度
            u = 0.5 * (obj.WL.u + obj.WR.u) + 0.5 * ...
                (obj.f_K_p(p_star, obj.WR) - obj.f_K_p(p_star, obj.WL));
        end

        function [p_star, u_star] = solver(obj)
            % 牛顿迭代；失败时退回 Brent 方法
            p = obj.p_init();
            converged = false;
            for k = 1:obj.maxit
                if p <= 0.0
                    break;
                end
                fp = obj.f_p(p);
                dfp = obj.f_prime_K_p(p, obj.WL) + obj.f_prime_K_p(p, obj.WR);
                p_new = p - fp / dfp;

                if p_new <= 0.0
                    % 阻尼：避免压强变为负值
                    p_new = 0.5 * p;
                end
                if 2 * abs(p_new - p) / (p + p_new) < obj.tol
                    p = p_new;
                    converged = true;
                    break;
                end
                p = p_new;
            end

            if ~converged || p <= 0.0
                [p_star, u_star] = obj.solver_brent();
                return;
            end

            p_star = p;
            u_star = obj.u_star_from_p(p_star);
        end

        function [a, b] = find_interval(obj, p0)
            % 寻找根的存在区间 [a, b]，满足 f(a) * f(b) <= 0
            p1 = p0;
            if obj.f_p(p0) <= 0.0
                for k = 1:200
                    if obj.f_p(p1) > 0.0
                        a = p0;
                        b = p1;
                        return;
                    end
                    p1 = 2.0 * p1;
                end
            else
                for k = 1:200
                    if obj.f_p(p1) <= 0.0
                        a = p1;
                        b = p0;
                        return;
                    end
                    p1 = 0.5 * p1;
                end
            end
            error('ExactRiemannSolver:interval', '无法找到根的存在区间');
        end

        function [p_star, u_star] = solver_brent(obj)
            % Brent 方法求解匹配方程
            p0 = obj.p_init();
            [a, b] = obj.find_interval(p0);
            fa = obj.f_p(a);
            fb = obj.f_p(b);
            if fa * fb > 0.0
                error('ExactRiemannSolver:brent', ...
                      'Brent 方法失败：区间端点函数值同号');
            end
            if abs(fa) < abs(fb)
                tmp = a; a = b; b = tmp;
                tmp = fa; fa = fb; fb = tmp;
            end

            c = a;
            fc = fa;
            d = 0.0;
            mflag = true;
            for k = 1:obj.maxit
                if abs(b - a) / abs(a) < obj.tol || fb == 0.0
                    break;
                end
                if fa ~= fc && fb ~= fc
                    % 逆二次插值
                    s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + ...
                        (b * fa * fc) / ((fb - fa) * (fb - fc)) + ...
                        (c * fa * fb) / ((fc - fa) * (fc - fb));
                else
                    % 割线法
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
                if fa * fs < 0.0
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

            p_star = b;
            u_star = obj.u_star_from_p(p_star);
        end
    end
end

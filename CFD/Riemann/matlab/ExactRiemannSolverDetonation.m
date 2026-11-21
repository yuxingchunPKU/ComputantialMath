classdef ExactRiemannSolverDetonation < handle
    %EXACTRIEMANNSOLVERDETONATION 含爆轰波的一维 Euler 精确 Riemann 求解器
    %
    %   右端为非反应气体（q>0），右波为强爆轰或 CJ 爆轰 + 稀疏波复合波。
    %   对应 Python 版本 code/Exact_Riemann_Combustion.py 中的
    %   ExactRiemannSolverDetonation 类。
    properties
        WL
        WR
        tol
        maxit
        p_star
        u_star
        p_cj
        u_cj
        rho_cj
        quiet
        dq
    end

    methods
        function obj = ExactRiemannSolverDetonation(WL, WR, tol, maxit, quiet)
            if nargin < 3 || isempty(tol)
                tol = 1e-8;
            end
            if nargin < 4 || isempty(maxit)
                maxit = 10000;
            end
            if nargin < 5 || isempty(quiet)
                quiet = true;
            end
            obj.WL = WL;
            obj.WR = WR;
            obj.tol = tol;
            obj.maxit = maxit;
            obj.quiet = quiet;
            obj.dq = WL.q - WR.q;
            obj.CJ_status();
        end

        function f = f_L_p(obj, p)
            % 左侧非反应波（激波或稀疏波）的特征函数
            W = obj.WL;
            p0 = W.p;
            tau0 = 1 / W.rho;
            g = W.gamma;
            mu2 = (g - 1) / (g + 1);
            if p >= p0
                % 激波分支
                A = (1 - mu2) * tau0;
                B = mu2 * p0;
                f = (p - p0) * sqrt(A / (p + B));
            else
                % 稀疏波分支
                a0 = W.sound_speed();
                idx = (g - 1) / (2 * g);
                f = 2 * a0 / (g - 1) * ((p / p0) ^ idx - 1);
            end
        end

        function f = f_prime_L_p(obj, p)
            W = obj.WL;
            p0 = W.p;
            tau0 = 1 / W.rho;
            g = W.gamma;
            mu2 = (g - 1) / (g + 1);
            if p >= p0
                A = (1 - mu2) * tau0;
                B = mu2 * p0;
                part1 = sqrt(A / (p + B));
                part2 = 1 - 0.5 * (p - p0) / (p + B);
                f = part1 * part2;
            else
                idx = -(g + 1) / (2 * g);
                a0 = W.sound_speed();
                f = (tau0 / a0) * ((p / p0) ^ idx);
            end
        end

        function CJ_status(obj)
            % 计算 CJ 状态
            WR = obj.WR;
            dq = obj.dq;
            p0 = WR.p;
            rho0 = WR.rho;
            u0 = WR.u;
            g = WR.gamma;
            mu2 = (g - 1) / (g + 1);

            t1 = 2 * g * p0 / ((g ^ 2 - 1) * rho0 * dq);
            t2 = 1 + sqrt(1 - t1);
            pcj = p0 - (g - 1) * rho0 * dq * t2;
            rhocj = rho0 * (pcj * (1 + g) - p0) / (g * pcj);
            tau = 1 / rho0;
            t3 = ((1 - mu2) * tau - 2 * mu2 * dq / (p0 - pcj)) / (mu2 * p0 + pcj);
            ucj = u0 + (pcj - p0) * sqrt(t3);

            obj.p_cj = pcj;
            obj.rho_cj = rhocj;
            obj.u_cj = ucj;
        end

        function f = f_R_p(obj, p)
            % 右侧燃烧波分支（分支取决于 p_cj）
            W = obj.WR;
            p0 = W.p;
            tau0 = 1 / W.rho;
            g = W.gamma;
            mu2 = (g - 1) / (g + 1);
            if p >= obj.p_cj
                % 强爆轰分支（激波）
                f = (p - p0) * sqrt(((1 - mu2) * tau0 - ...
                    2 * mu2 * obj.dq / (p0 - p)) / (mu2 * p0 + p));
            else
                % CJ 爆轰：激波 + 稀疏波的复合波
                part1 = (obj.p_cj - p0) * sqrt(((1 - mu2) * tau0 - ...
                        2 * mu2 * obj.dq / (p0 - obj.p_cj)) / ...
                        (mu2 * p0 + obj.p_cj));
                part2 = 2 * (sqrt(g) / (g - 1)) * ...
                        sqrt(obj.p_cj / obj.rho_cj) * ...
                        ((p / obj.p_cj) ^ ((g - 1) / (2 * g)) - 1);
                f = part1 + part2;
            end
        end

        function f = f_prime_R_p(obj, p)
            W = obj.WR;
            p0 = W.p;
            tau0 = 1 / W.rho;
            g = W.gamma;
            mu2 = (g - 1) / (g + 1);
            if p >= obj.p_cj
                tau_p = (tau0 * (p0 + mu2 * p) - 2 * mu2 * obj.dq) / ...
                        (p + mu2 * p0);
                tau_dp = ((mu2 ^ 2 - 1) * tau0 * p0 + ...
                          2 * mu2 * obj.dq) / (p + mu2 * p0) ^ 2;
                f = 0.5 * (tau_dp * (p0 - p) - (tau_p - tau0)) / ...
                    sqrt((tau_p - tau0) * (p0 - p));
            else
                tau_cj = 1 / obj.rho_cj;
                a_cj = sqrt(g * obj.p_cj * tau_cj);
                f = tau_cj * (p / obj.p_cj) ^ (-(g + 1) / (2 * g)) / a_cj;
            end
        end

        function f = f_p(obj, p)
            f = obj.f_L_p(p) + obj.f_R_p(p) + obj.WR.u - obj.WL.u;
        end

        function p = p_init(obj)
            p = obj.p_cj;
        end

        function [p_star, u_star] = solver(obj)
            % 牛顿迭代
            p = obj.p_init();
            for k = 1:obj.maxit
                fp = obj.f_p(p);
                dfp = obj.f_prime_L_p(p) + obj.f_prime_R_p(p);
                p_new = max(p - fp / dfp, obj.tol);
                if 2 * abs(p_new - p) / (p + p_new) < obj.tol
                    p = p_new;
                    break;
                else
                    p = p_new;
                end
            end

            obj.p_star = p;
            obj.u_star = 0.5 * (obj.WL.u + obj.WR.u) + ...
                         0.5 * (obj.f_R_p(p) - obj.f_L_p(p));

            if ~obj.quiet
                fprintf('u_star %.16f\n', obj.u_star);
                fprintf('p_star %.16f\n', obj.p_star);
                fprintf('rho_cj %.16f\n', obj.rho_cj);
                fprintf('u_cj %.16f\n', obj.u_cj);
                fprintf('p_cj %.16f\n', obj.p_cj);
            end

            p_star = obj.p_star;
            u_star = obj.u_star;
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
            error('ExactRiemannSolverDetonation:interval', '无法找到根的存在区间');
        end

        function [p_star, u_star] = solver_brent(obj)
            p0 = obj.p_init();
            [a, b] = obj.find_interval(p0);
            fa = obj.f_p(a);
            fb = obj.f_p(b);
            if fa * fb > 0
                error('ExactRiemannSolverDetonation:brent', ...
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

            obj.p_star = b;
            obj.u_star = 0.5 * (obj.WL.u + obj.WR.u) + ...
                         0.5 * (obj.f_R_p(b) - obj.f_L_p(b));

            if ~obj.quiet
                fprintf('u_star %.16f\n', obj.u_star);
                fprintf('p_star %.16f\n', obj.p_star);
                fprintf('rho_cj %.16f\n', obj.rho_cj);
                fprintf('u_cj %.16f\n', obj.u_cj);
                fprintf('p_cj %.16f\n', obj.p_cj);
            end

            p_star = obj.p_star;
            u_star = obj.u_star;
        end

        function [rho_cj, u_cj, p_cj] = CJ_data(obj)
            rho_cj = obj.rho_cj;
            u_cj = obj.u_cj;
            p_cj = obj.p_cj;
        end
    end
end

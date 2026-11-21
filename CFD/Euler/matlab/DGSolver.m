classdef DGSolver < handle
    %DGSOLVER 单介质间断有限元（DG）求解器，支持 P0/P1/P2
    %
    %   基函数为 Legendre 多项式 B0 = 1, B1 = xi, B2 = xi^2 - 1/3。
    %   对应 Python 版本 euler_unified/single.py 中的 DGSolver 类。
    properties
        cfg
        mesh
        gas
        order
        M
        flux_kind
        tvb
        W
        xq_ref
        wq
        U
    end

    methods
        function obj = DGSolver(cfg)
            obj.cfg = cfg;
            obj.mesh = euler_mesh(cfg);
            obj.gas = StiffenedGas(cfg.fluids.gamma, cfg.fluids.p_inf);
            obj.order = cfg.order;
            obj.M = obj.order + 1;
            obj.flux_kind = cfg.flux;
            obj.tvb = cfg.limiter.tvb;
            obj.W = max(4, obj.M + 1);
            [obj.xq_ref, w] = gauss_legendre(obj.W);
            obj.wq = 0.5 * w;
            obj.U = obj.initial_data();
        end

        % --------------------------------------------------------------
        % 基函数
        % --------------------------------------------------------------
        function B = basis(obj, x, xc, dx)
            x_arr = x(:).';
            xi = (x_arr - xc) / (0.5 * dx);
            B = zeros(obj.M, numel(x_arr));
            B(1, :) = 1.0;
            if obj.M > 1
                B(2, :) = xi;
            end
            if obj.M > 2
                B(3, :) = xi .* xi - 1.0 / 3.0;
            end
        end

        function B = basis_x(obj, x, xc, dx)
            x_arr = x(:).';
            B = zeros(obj.M, numel(x_arr));
            if obj.M > 1
                B(2, :) = 1.0 / (0.5 * dx);
            end
            if obj.M > 2
                B(3, :) = 2.0 * (x_arr - xc) / ((0.5 * dx) ^ 2);
            end
        end

        % --------------------------------------------------------------
        % 初始投影
        % --------------------------------------------------------------
        function U = initial_data(obj)
            nx = obj.mesh.nx;
            dx = obj.mesh.dx;
            W = obj.W;
            U = zeros(3, obj.M, nx);
            for i = 1:nx
                xq = 0.5 * dx * obj.xq_ref + obj.mesh.centers(i);  % W x 1
                [rho, u, p] = primitive_at(obj.cfg, xq);
                Uq = obj.gas.conservative(rho, u, p);              % 3 x W
                B = obj.basis(xq, obj.mesh.centers(i), dx);        % M x W
                Mmat = (B .* obj.wq.') * B.' * dx;
                U(:, :, i) = ((Uq .* obj.wq.') * B.' * dx) ./ ...
                             diag(Mmat).';
            end
        end

        % --------------------------------------------------------------
        function [UL, UR] = face_states(obj, U)
            n = obj.mesh.nx;
            Bl = obj.basis(obj.mesh.edges(1:end - 1), obj.mesh.centers, ...
                           obj.mesh.dx);
            Br = obj.basis(obj.mesh.edges(2:end), obj.mesh.centers, ...
                           obj.mesh.dx);
            UL = zeros(3, n + 1);
            UR = zeros(3, n + 1);
            % 面 j 分隔左单元 j-1 与右单元 j：左状态来自左单元的右端点，
            % 右状态来自右单元的左端点。
            UL(:, 2:end) = apply_basis(Br, U);
            UR(:, 1:end - 1) = apply_basis(Bl, U);
            switch obj.mesh.boundary
                case 'periodic'
                    UL(:, 1) = UL(:, end);
                    UR(:, end) = UR(:, 1);
                case 'transmissive'
                    UL(:, 1) = UR(:, 1);
                    UR(:, end) = UL(:, end);
                case 'reflective'
                    UL(:, 1) = reflect_state(UR(:, 1));
                    UR(:, end) = reflect_state(UL(:, end));
                otherwise
                    error('DGSolver:boundary', 'unknown boundary: %s', ...
                          obj.mesh.boundary);
            end
        end

        % --------------------------------------------------------------
        function [dU, speed] = rhs(obj, U)
            n = obj.mesh.nx;
            dx = obj.mesh.dx;
            [UL, UR] = obj.face_states(U);
            F = numerical_flux(UL, UR, obj.gas, obj.flux_kind);
            dU = zeros(size(U));
            for i = 1:n
                center = obj.mesh.centers(i);
                B_lr = obj.basis([obj.mesh.edges(i), obj.mesh.edges(i + 1)], ...
                                 center, dx);                       % M x 2
                V1 = F(:, i + 1) * B_lr(:, 2).' - F(:, i) * B_lr(:, 1).';
                xq = 0.5 * dx * obj.xq_ref + center;
                Bq = obj.basis(xq, center, dx);
                Bxq = obj.basis_x(xq, center, dx);
                Uq = reshape(U(:, :, i), 3, obj.M) * Bq;            % 3 x W
                Fq = obj.gas.flux(Uq);
                V2 = -(dx * (Fq * (Bxq .* obj.wq.').'));
                Mmat = (Bq .* obj.wq.') * Bq.' * dx;
                % Python 参考代码存的是负的时间导数；此处统一为 dU/dt。
                dU(:, :, i) = -(V1 + V2) ./ diag(Mmat).';
            end
            speed = max(obj.gas.max_speed(UL), obj.gas.max_speed(UR));
        end

        % --------------------------------------------------------------
        % 限制器
        % --------------------------------------------------------------
        function [DU_p, DU_m] = differences(obj)
            U = obj.U;
            n = obj.mesh.nx;
            U0 = reshape(U(:, 1, :), 3, n);
            DU_p = zeros(3, n);
            DU_m = zeros(3, n);
            DU_p(:, 1:n - 1) = U0(:, 2:n) - U0(:, 1:n - 1);
            DU_m(:, 2:n) = U0(:, 2:n) - U0(:, 1:n - 1);
            if strcmp(obj.mesh.boundary, 'periodic')
                DU_p(:, n) = U0(:, 1) - U0(:, n);
                DU_m(:, 1) = U0(:, 1) - U0(:, n);
            end
        end

        function U = limit(obj, U)
            obj.U = U;
            if obj.M == 1
                return;
            end
            n = obj.mesh.nx;
            dx = obj.mesh.dx;
            [DU_p, DU_m] = obj.differences();
            if obj.M == 2
                Br = obj.basis(obj.mesh.edges(2:end), obj.mesh.centers, dx);
                U_r = apply_basis(Br, U);
                U0 = reshape(U(:, 1, :), 3, n);
                dUp = U_r - U0;
                dUp_mod = minmod_tvb(dUp, DU_p, DU_m, obj.tvb, dx);
                U(:, 2, :) = reshape(dUp_mod, 3, 1, n);
                return;
            end

            % P2 特征变量 TVB 限制器
            Bl = obj.basis(obj.mesh.edges(1:end - 1), obj.mesh.centers, dx);
            Br = obj.basis(obj.mesh.edges(2:end), obj.mesh.centers, dx);
            U_l = apply_basis(Bl, U);
            U_r = apply_basis(Br, U);
            U0 = reshape(U(:, 1, :), 3, n);
            dUr = U_r - U0;
            dUl = U0 - U_l;
            for i = 1:n
                R = char_matrix(U0(:, i), obj.gas);
                dUr_mod = R * minmod_tvb(R \ dUr(:, i), R \ DU_p(:, i), ...
                                         R \ DU_m(:, i), obj.tvb, dx);
                dUl_mod = R * minmod_tvb(R \ dUl(:, i), R \ DU_p(:, i), ...
                                         R \ DU_m(:, i), obj.tvb, dx);
                U(:, 2, i) = 0.5 * (dUr_mod + dUl_mod);
                U(:, 3, i) = 0.75 * (dUr_mod - dUl_mod);
            end
        end

        % --------------------------------------------------------------
        function res = run(obj)
            U = obj.U;
            t_end = obj.cfg.time.t_end;
            cfl = obj.cfg.time.cfl;
            t = 0.0;
            dt = 0.0;
            steps = 0;
            while t < t_end
                [dU, speed] = obj.rhs(U);
                if ~isfinite(speed) || speed <= 0.0
                    error('DGSolver:speed', ...
                          'non-positive or non-finite maximum wave speed');
                end
                dt = cfl * obj.mesh.dx / speed;
                if t + dt >= t_end
                    dt = t_end - t;
                end
                if obj.M == 1
                    U = U + dt * dU;
                elseif obj.M == 2
                    U_cp = U;
                    U = obj.limit(U + dt * dU);
                    [dU1, ~] = obj.rhs(U);
                    U = obj.limit(0.5 * U_cp + 0.5 * (U + dt * dU1));
                else
                    U_cp = U;
                    U = obj.limit(U + dt * dU);
                    [dU1, ~] = obj.rhs(U);
                    U = obj.limit(0.75 * U_cp + 0.25 * (U + dt * dU1));
                    [dU2, ~] = obj.rhs(U);
                    U = obj.limit((1.0 / 3.0) * U_cp + ...
                                  (2.0 / 3.0) * (U + dt * dU2));
                end
                t = t + dt;
                steps = steps + 1;
            end
            obj.U = U;
            U0 = reshape(U(:, 1, :), 3, obj.mesh.nx);
            [rho, u, p] = obj.gas.primitive(U0);
            res = make_result(obj.cfg, obj.mesh.centers, rho, u, p, dt, steps);
        end
    end
end

classdef FVSolver < handle
    %FVSOLVER 单介质有限体积求解器（二阶为 MUSCL + minmod）
    %
    %   对应 Python 版本 euler_unified/single.py 中的 FVSolver 类。
    properties
        cfg
        mesh
        gas
        order
        flux_kind
        U
    end

    methods
        function obj = FVSolver(cfg)
            obj.cfg = cfg;
            obj.mesh = euler_mesh(cfg);
            obj.gas = StiffenedGas(cfg.fluids.gamma, cfg.fluids.p_inf);
            obj.order = cfg.order;
            obj.flux_kind = cfg.flux;
            obj.U = obj.initial_data();
        end

        function U = initial_data(obj)
            [rho, u, p] = primitive_at(obj.cfg, obj.mesh.centers);
            U = obj.gas.conservative(rho, u, p);
        end

        function Ue = extended(obj, U)
            % 与选定边界条件一致的两层 ghost 单元（此处每侧一层）
            n = size(U, 2);
            Ue = zeros(3, n + 2);
            Ue(:, 2:n + 1) = U;
            switch obj.mesh.boundary
                case 'periodic'
                    Ue(:, 1) = U(:, end);
                    Ue(:, end) = U(:, 1);
                case 'transmissive'
                    Ue(:, 1) = U(:, 1);
                    Ue(:, end) = U(:, end);
                case 'reflective'
                    Ue(:, 1) = reflect_state(U(:, 1));
                    Ue(:, end) = reflect_state(U(:, end));
                otherwise
                    error('FVSolver:boundary', 'unknown boundary: %s', ...
                          obj.mesh.boundary);
            end
        end

        function [UL, UR] = face_states(obj, U)
            Ue = obj.extended(U);
            if obj.order == 1
                s = zeros(size(Ue));
            else
                dL = Ue(:, 2:end - 1) - Ue(:, 1:end - 2);
                dR = Ue(:, 3:end) - Ue(:, 2:end - 1);
                s = zeros(size(Ue));
                s(:, 2:end - 1) = minmod2(dL, dR);
                s(:, 1) = s(:, 2);
                s(:, end) = s(:, end - 1);
            end
            UL = Ue(:, 1:end - 1) + 0.5 * s(:, 1:end - 1);
            UR = Ue(:, 2:end) - 0.5 * s(:, 2:end);
        end

        function [dU, speed] = rhs(obj, U)
            [UL, UR] = obj.face_states(U);
            F = numerical_flux(UL, UR, obj.gas, obj.flux_kind);
            dU = -(F(:, 2:end) - F(:, 1:end - 1)) / obj.mesh.dx;
            speed = max(obj.gas.max_speed(UL), obj.gas.max_speed(UR));
        end

        function res = run(obj)
            U = obj.U;
            t_end = obj.cfg.time.t_end;
            cfl = obj.cfg.time.cfl;
            dt = 0.0;
            t = 0.0;
            steps = 0;

            while t < t_end
                [dU, speed] = obj.rhs(U);
                if ~isfinite(speed) || speed <= 0.0
                    error('FVSolver:speed', ...
                          'non-positive or non-finite maximum wave speed');
                end
                dt = cfl * obj.mesh.dx / speed;
                if t + dt >= t_end
                    dt = t_end - t;
                end
                if obj.order == 1
                    U = U + dt * dU;
                else
                    U1 = U + dt * dU;
                    [dU1, ~] = obj.rhs(U1);
                    U = 0.5 * U + 0.5 * (U1 + dt * dU1);
                end
                t = t + dt;
                steps = steps + 1;
            end

            obj.U = U;
            [rho, u, p] = obj.gas.primitive(U);
            res = make_result(obj.cfg, obj.mesh.centers, rho, u, p, dt, steps);
        end
    end
end

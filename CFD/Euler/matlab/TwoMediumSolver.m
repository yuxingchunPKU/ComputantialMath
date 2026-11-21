classdef TwoMediumSolver < handle
    %TWOMEDIUMSOLVER 固定 Euler 网格上的 sharp-interface 两介质 cut-cell 求解器
    %
    %   含界面的单元被拆成两个相单元，界面用精确 Riemann 求解器推进。
    %   order = 0 时为同一套算法的一阶有限体积版本。
    %   对应 Python 版本 euler_unified/two_medium.py。
    %
    %   每个物理量按相存放：U{ph} 为 3 x M x nx，U_IC{ph} 为 3 x M。
    properties
        cfg
        x_min
        x_max
        nx
        dx
        point
        n_edge
        ele_vol
        ele_centroid
        boundary
        order
        M
        flux_kind
        tvb
        eps
        gas
        W
        xq_ref
        wq
        x_I
        x_I_cp
        x_I_new
        phi
        point_label
        point_label_new
        ele_label
        ele_label_new
        idx
        idx_new
        phase_vol
        phase_centroid
        phase_vol_new
        phase_centroid_new
        phase_vol_cp
        phase_centroid_cp
        IC_vol
        IC_centroid
        IC_vol_new
        IC_centroid_new
        U
        U_data_cp
        U_IC
        U_IC_cp
        IC_flux
    end

    methods
        function obj = TwoMediumSolver(cfg)
            obj.cfg = cfg;
            m = cfg.mesh;
            obj.x_min = m.x_min;
            obj.x_max = m.x_max;
            obj.nx = m.nx;
            obj.dx = (m.x_max - m.x_min) / m.nx;
            obj.point = linspace(m.x_min, m.x_max, m.nx + 1);      % 1 x (nx+1)
            obj.n_edge = obj.nx + 1;
            obj.ele_vol = obj.dx;
            obj.ele_centroid = 0.5 * (obj.point(1:end - 1) + obj.point(2:end));
            obj.boundary = m.boundary;

            % 两介质的一阶有限体积是同一套 cut-cell 算法的 P0 版本。
            if strcmp(cfg.scheme, 'fv')
                obj.order = 0;
            else
                obj.order = cfg.order;
            end
            obj.M = obj.order + 1;
            obj.flux_kind = cfg.flux;
            obj.tvb = cfg.limiter.tvb;
            obj.eps = 1.0e-12;

            obj.gas = {StiffenedGas(cfg.fluids.left.gamma, ...
                                    cfg.fluids.left.p_inf), ...
                       StiffenedGas(cfg.fluids.right.gamma, ...
                                    cfg.fluids.right.p_inf)};

            obj.W = max(4, obj.M + 1);
            [obj.xq_ref, w] = gauss_legendre(obj.W);
            obj.wq = 0.5 * w;

            % 界面几何
            obj.x_I = obj.safe_interface(cfg.initial.x_interface);
            obj.x_I_cp = obj.x_I;
            obj.phi = obj.point - obj.x_I;

            obj.point_label = zeros(1, obj.n_edge);
            obj.point_label_new = zeros(1, obj.n_edge);
            obj.ele_label = zeros(1, obj.nx);
            obj.ele_label_new = zeros(1, obj.nx);
            obj.idx = 0;
            obj.idx_new = 0;

            obj.phase_vol = zeros(2, obj.nx);
            obj.phase_centroid = zeros(2, obj.nx);
            obj.phase_vol_new = zeros(2, obj.nx);
            obj.phase_centroid_new = zeros(2, obj.nx);
            obj.phase_vol_cp = zeros(2, obj.nx);
            obj.phase_centroid_cp = zeros(2, obj.nx);
            obj.IC_vol = zeros(2, 1);
            obj.IC_centroid = zeros(2, 1);
            obj.IC_vol_new = zeros(2, 1);
            obj.IC_centroid_new = zeros(2, 1);

            obj.U = {zeros(3, obj.M, obj.nx), zeros(3, obj.M, obj.nx)};
            obj.U_data_cp = {zeros(3, obj.M, obj.nx), ...
                             zeros(3, obj.M, obj.nx)};
            obj.U_IC = {zeros(3, obj.M), zeros(3, obj.M)};
            obj.U_IC_cp = {zeros(3, obj.M), zeros(3, obj.M)};
            obj.IC_flux = zeros(3, 1);

            obj.mark(obj.phi, false);
            obj.phase_info(obj.x_I, false);
            obj.interface_info(false, obj.idx);
            obj.init_data();
            obj.compose();
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

        function B = basis_single(obj, x, xc, dx)
            B = obj.basis(x, xc, dx);
            B = B(:, 1);
        end

        % --------------------------------------------------------------
        % 几何 / 标记
        % --------------------------------------------------------------
        function x_i = safe_interface(obj, x_i)
            % 避免界面恰好落在网格点上：极小移动保证 cut cell 两个相体积均为正。
            x_i = double(x_i);
            for k = 2:obj.nx          % point(2:end-1)
                edge = obj.point(k);
                if abs(edge - x_i) < 1.0e-12 * obj.dx
                    x_i = edge + 1.0e-8 * obj.dx;
                    return;
                end
            end
        end

        function mark(obj, phi, isnew)
            ph = double(phi > 0.0);
            el = ph(1:end - 1) + ph(2:end);
            idx = find(el == 1);
            if isempty(idx)
                error('TwoMediumSolver:interface', ...
                      'the material interface is not inside the domain');
            end
            if isnew
                obj.point_label_new = ph;
                obj.ele_label_new = el;
                obj.idx_new = idx(1);
            else
                obj.point_label = ph;
                obj.ele_label = el;
                obj.idx = idx(1);
            end
        end

        function phase_info(obj, x_i, isnew)
            if nargin < 3
                isnew = false;
            end
            if isnew
                vol = obj.phase_vol_new;
                cen = obj.phase_centroid_new;
                lab = obj.ele_label_new;
            else
                vol = obj.phase_vol;
                cen = obj.phase_centroid;
                lab = obj.ele_label;
            end
            cen(1, :) = obj.ele_centroid;
            cen(2, :) = obj.ele_centroid;
            vol(1, :) = obj.ele_vol * (lab < 1);
            vol(2, :) = obj.ele_vol * (lab > 1);
            for i = 1:obj.nx
                if lab(i) == 1
                    vol(1, i) = abs(x_i - obj.point(i));
                    vol(2, i) = abs(obj.point(i + 1) - x_i);
                    cen(1, i) = 0.5 * (x_i + obj.point(i));
                    cen(2, i) = 0.5 * (obj.point(i + 1) + x_i);
                end
            end
            if isnew
                obj.phase_vol_new = vol;
                obj.phase_centroid_new = cen;
            else
                obj.phase_vol = vol;
                obj.phase_centroid = cen;
            end
        end

        function interface_info(obj, isnew, idx)
            if nargin < 2
                isnew = false;
            end
            if nargin < 3 || isempty(idx)
                if isnew
                    idx = obj.idx_new;
                else
                    idx = obj.idx;
                end
            end
            if isnew
                vol = obj.phase_vol_new;
                cen = obj.phase_centroid_new;
            else
                vol = obj.phase_vol;
                cen = obj.phase_centroid;
            end
            sl = idx - 1:idx + 1;
            v = sum(vol(:, sl), 2);
            c = sum(cen(:, sl) .* vol(:, sl), 2) ./ v;
            if isnew
                obj.IC_vol_new = v;
                obj.IC_centroid_new = c;
            else
                obj.IC_vol = v;
                obj.IC_centroid = c;
            end
        end

        % --------------------------------------------------------------
        % 初始数据
        % --------------------------------------------------------------
        function init_data(obj)
            initial = obj.cfg.initial;
            left = initial.left;
            right = initial.right;
            for i = 1:obj.nx
                for ph = 1:2
                    vol = obj.phase_vol(ph, i);
                    if vol <= obj.eps
                        obj.set_u(i, ph, zeros(3, obj.M));
                        continue;
                    end
                    gas = obj.gas{ph};
                    if strcmp(initial.type, 'smooth')
                        cen = obj.phase_centroid(ph, i);
                        xq = (0.5 * vol * obj.xq_ref + cen).';
                        rho = initial.rho0 + initial.rho_amp * ...
                              sin(initial.k * pi * xq);
                        u = initial.u0 * ones(size(xq));
                        p = initial.p0 * ones(size(xq));
                        Uq = gas.conservative(rho, u, p);
                        B = obj.basis(xq, cen, vol);
                        Mmat = (B .* obj.wq.') * B.' * vol;
                        val = ((Uq .* obj.wq.') * B.' * vol) ./ ...
                              diag(Mmat).';
                    else
                        if ph == 1
                            state = left;
                        else
                            state = right;
                        end
                        Uc = gas.conservative(state.rho, state.u, state.p);
                        val = zeros(3, obj.M);
                        val(:, 1) = Uc;
                    end
                    obj.set_u(i, ph, val);
                end
            end
        end

        function set_u(obj, i, ph, val)
            U = obj.U{ph};
            U(:, :, i) = val;
            obj.U{ph} = U;
        end

        function set_uic(obj, ph, val)
            U = obj.U_IC{ph};
            U(:, :) = val;
            obj.U_IC{ph} = U;
        end

        % --------------------------------------------------------------
        % 界面单元投影
        % --------------------------------------------------------------
        function compose(obj)
            for ph = 1:2
                Uf = zeros(3, obj.M, 3);
                for j = 1:3
                    i = obj.idx - 2 + j;
                    vol = obj.phase_vol(ph, i);
                    if vol > obj.eps
                        cen = obj.phase_centroid(ph, i);
                        xq = (0.5 * vol * obj.xq_ref + cen).';
                        Bs = obj.basis(xq, cen, vol);
                        Bi = obj.basis(xq, obj.IC_centroid(ph), ...
                                       obj.IC_vol(ph));
                        Umat = obj.U{ph}(:, :, i);
                        Uf(:, :, j) = (Umat * Bs) * ...
                                      (Bi .* obj.wq.').' * vol;
                    end
                end
                xq = (0.5 * obj.IC_vol(ph) * obj.xq_ref + ...
                      obj.IC_centroid(ph)).';
                Bi = obj.basis(xq, obj.IC_centroid(ph), obj.IC_vol(ph));
                Mmat = (Bi .* obj.wq.') * Bi.' * obj.IC_vol(ph);
                obj.set_uic(ph, sum(Uf, 3) ./ diag(Mmat).');
            end
        end

        function decompose(obj, idx)
            for ph = 1:2
                for i = idx - 1:idx + 1
                    vol = obj.phase_vol_new(ph, i);
                    if vol > obj.eps
                        cen = obj.phase_centroid_new(ph, i);
                        xq = (0.5 * vol * obj.xq_ref + cen).';
                        Bs = obj.basis(xq, cen, vol);
                        Bi = obj.basis(xq, obj.IC_centroid_new(ph), ...
                                       obj.IC_vol_new(ph));
                        F = (obj.U_IC{ph} * Bi) * ...
                            (Bs .* obj.wq.').' * vol;
                        Mi = (Bs .* obj.wq.') * Bs.' * vol;
                        val = F ./ diag(Mi).';
                    else
                        val = zeros(3, obj.M);
                    end
                    obj.set_u(i, ph, val);
                end
            end
        end

        % --------------------------------------------------------------
        function m = uic_mass(obj, vol2, cen2)
            m = zeros(2, 1);
            for ph = 1:2
                xq = (0.5 * vol2(ph) * obj.xq_ref + cen2(ph)).';
                B = obj.basis(xq, cen2(ph), vol2(ph));
                Mmat = (B .* obj.wq.') * B.' * vol2(ph);
                m(ph) = Mmat(1, 1);
            end
        end

        function update_uic(obj, dt, dUIC, stage)
            % dUIC 为积分矩 d/dt int U B dx 的时间导数，因此显式出现新旧质量矩阵。
            M_old = obj.uic_mass(obj.IC_vol, obj.IC_centroid);
            M_new = obj.uic_mass(obj.IC_vol_new, obj.IC_centroid_new);
            for ph = 1:2
                old = obj.U_IC{ph} * M_old(ph);
                if isnumeric(stage) && stage == 1
                    value = old + dt * dUIC{ph};
                elseif isnumeric(stage) && stage == 2
                    value = 0.5 * obj.U_IC_cp{ph} * M_old(ph) + ...
                            0.5 * (old + dt * dUIC{ph});
                elseif ischar(stage) && strcmp(stage, '3_2')
                    value = 0.75 * obj.U_IC_cp{ph} * M_old(ph) + ...
                            0.25 * (old + dt * dUIC{ph});
                elseif ischar(stage) && strcmp(stage, '3_3')
                    value = (1.0 / 3.0) * obj.U_IC_cp{ph} * M_old(ph) + ...
                            (2.0 / 3.0) * (old + dt * dUIC{ph});
                else
                    error('TwoMediumSolver:stage', 'unknown stage');
                end
                obj.set_uic(ph, value / M_new(ph));
            end
        end

        function backup_uic_cp(obj, idx_new)
            % 旧（步初）几何
            sl = idx_new - 1:idx_new + 1;
            vol_cp = obj.phase_vol_cp(:, sl);
            cen_cp = obj.phase_centroid_cp(:, sl);
            temp_v = sum(vol_cp, 2);
            temp_c = sum(cen_cp .* vol_cp, 2) ./ temp_v;
            temp_v_new = sum(obj.phase_vol_new(:, sl), 2);
            temp_c_new = sum(obj.phase_centroid_new(:, sl) .* ...
                             obj.phase_vol_new(:, sl), 2) ./ temp_v_new;
            for ph = 1:2
                Uf = zeros(3, obj.M, 3);
                for j = 1:3
                    i = idx_new - 2 + j;
                    if obj.phase_vol_cp(ph, i) > obj.eps
                        vol = obj.phase_vol_cp(ph, i);
                        cen = obj.phase_centroid_cp(ph, i);
                        xq = (0.5 * vol * obj.xq_ref + cen).';
                        Bs = obj.basis(xq, cen, vol);
                        Bi = obj.basis(xq, temp_c(ph), temp_v(ph));
                        Umat = obj.U_data_cp{ph}(:, :, i);
                        Uf(:, :, j) = (Umat * Bs) * ...
                                      (Bi .* obj.wq.').' * vol;
                    end
                end
                xq = (0.5 * temp_v(ph) * obj.xq_ref + temp_c(ph)).';
                Bi = obj.basis(xq, temp_c(ph), temp_v(ph));
                M_temp = (Bi .* obj.wq.') * Bi.' * temp_v(ph);
                uic = sum(Uf, 3) ./ diag(M_temp).';
                % 将多项式重新缩放到新的界面单元。
                xq_new = (0.5 * temp_v_new(ph) * obj.xq_ref + ...
                          temp_c_new(ph)).';
                Bi_new = obj.basis(xq_new, temp_c_new(ph), ...
                                   temp_v_new(ph));
                M_new = (Bi_new .* obj.wq.') * Bi_new.' * ...
                        temp_v_new(ph);
                obj.U_IC_cp{ph} = uic .* diag(M_temp).' ./ diag(M_new).';
            end
        end

        % --------------------------------------------------------------
        % 界面精确 Riemann 问题
        % --------------------------------------------------------------
        function u_star = interface_velocity(obj)
            [rho0, u0, p0] = obj.gas{1}.primitive(obj.U_IC{1}(:, 1));
            [rho1, u1, p1] = obj.gas{2}.primitive(obj.U_IC{2}(:, 1));
            WL = Fluid(rho0, u0, p0, obj.gas{1}.gamma, obj.gas{1}.p_inf);
            WR = Fluid(rho1, u1, p1, obj.gas{2}.gamma, obj.gas{2}.p_inf);
            try
                solver = ExactRiemannSolver(WL, WR, 1.0e-12, 1000);
                [p_star, u_star] = solver.solver();
            catch
                p_star = 0.5 * (p0 + p1);
                u_star = 0.5 * (u0 + u1);
            end
            obj.IC_flux = [0.0; p_star; p_star * u_star];
        end

        % --------------------------------------------------------------
        % 空间残差
        % --------------------------------------------------------------
        function [dU, dUIC, max_speed] = rhs(obj)
            n = obj.nx;
            J = obj.idx;
            dU = {zeros(3, obj.M, n), zeros(3, obj.M, n)};
            dUIC = {zeros(3, obj.M), zeros(3, obj.M)};

            edge_l = obj.basis(obj.point(2:end), obj.ele_centroid, ...
                               obj.ele_vol);              % M x n（右端点）
            edge_r = obj.basis(obj.point(1:end - 1), obj.ele_centroid, ...
                               obj.ele_vol);              % M x n（左端点）
            Ul = zeros(3, obj.n_edge);
            Ur = zeros(3, obj.n_edge);

            % 界面单元左侧的相 1 单元
            if J - 2 >= 1
                k = J - 2;
                Ul(:, 2:J - 1) = apply_basis(edge_l(:, 1:k), ...
                                             obj.U{1}(:, :, 1:k));
                Ur(:, 1:J - 2) = apply_basis(edge_r(:, 1:k), ...
                                             obj.U{1}(:, :, 1:k));
            end
            % 界面单元右侧的相 2 单元
            if J + 2 <= n
                Ul(:, J + 3:end) = apply_basis(edge_l(:, J + 2:end), ...
                                               obj.U{2}(:, :, J + 2:end));
                Ur(:, J + 2:end - 1) = apply_basis(edge_r(:, J + 2:end), ...
                                                   obj.U{2}(:, :, J + 2:end));
            end

            % 界面单元的外侧边
            Br0 = obj.basis_single(obj.point(J - 1), obj.IC_centroid(1), ...
                                   obj.IC_vol(1));
            Bl1 = obj.basis_single(obj.point(J + 2), obj.IC_centroid(2), ...
                                   obj.IC_vol(2));
            Ur(:, J - 1) = obj.U_IC{1} * Br0;
            Ul(:, J + 2) = obj.U_IC{2} * Bl1;

            % 区域边界
            if strcmp(obj.boundary, 'periodic')
                Ul(:, 1) = Ul(:, end);
                Ur(:, end) = Ur(:, 1);
            else  % transmissive
                Ul(:, 1) = Ur(:, 1);
                Ur(:, end) = Ul(:, end);
            end

            Fu = zeros(3, obj.n_edge);
            max_speed = 0.0;
            for i = 1:obj.n_edge
                if i >= J && i <= J + 1
                    continue;
                end
                if i <= J - 1
                    gas = obj.gas{1};
                else
                    gas = obj.gas{2};
                end
                Fu(:, i) = numerical_flux(Ul(:, i), Ur(:, i), gas, ...
                                          obj.flux_kind);
                max_speed = max([max_speed, ...
                                 gas.max_speed(Ul(:, i)), ...
                                 gas.max_speed(Ur(:, i))]);
            end

            % 常规单元
            for i = 1:n
                if i >= J - 1 && i <= J + 1
                    continue;
                end
                if i < J - 1
                    ph = 1;
                else
                    ph = 2;
                end
                gas = obj.gas{ph};
                cen = obj.ele_centroid(i);
                Blr = obj.basis([obj.point(i), obj.point(i + 1)], cen, ...
                                obj.ele_vol);
                V1 = Fu(:, i + 1) * Blr(:, 2).' - Fu(:, i) * Blr(:, 1).';
                xq = (0.5 * obj.ele_vol * obj.xq_ref + cen).';
                Bq = obj.basis(xq, cen, obj.ele_vol);
                Bxq = obj.basis_x(xq, cen, obj.ele_vol);
                Uq = obj.U{ph}(:, :, i) * Bq;
                Fq = gas.flux(Uq);
                V2 = -(obj.ele_vol * (Fq * (Bxq .* obj.wq.').'));
                Mmat = (Bq .* obj.wq.') * Bq.' * obj.ele_vol;
                val = -(V1 + V2) ./ diag(Mmat).';
                tmp = dU{ph};
                tmp(:, :, i) = val;
                dU{ph} = tmp;
            end

            % 界面单元：每个相一个方程
            edge_idx = [J - 1, J + 2];
            for ph = 1:2
                gas = obj.gas{ph};
                vol = obj.IC_vol(ph);
                cen = obj.IC_centroid(ph);
                Blr = obj.basis([cen - 0.5 * vol, cen + 0.5 * vol], cen, vol);
                if ph == 1
                    V1 = obj.IC_flux * Blr(:, 2).' - ...
                         Fu(:, edge_idx(1)) * Blr(:, 1).';
                else
                    V1 = Fu(:, edge_idx(2)) * Blr(:, 2).' - ...
                         obj.IC_flux * Blr(:, 1).';
                end
                xq = (0.5 * vol * obj.xq_ref + cen).';
                Bq = obj.basis(xq, cen, vol);
                Bxq = obj.basis_x(xq, cen, vol);
                Uq = obj.U_IC{ph} * Bq;
                Fq = gas.flux(Uq);
                V2 = -(vol * (Fq * (Bxq .* obj.wq.').'));
                dUIC{ph} = -(V1 + V2);
            end
        end

        % --------------------------------------------------------------
        % 限制器
        % --------------------------------------------------------------
        function limit(obj)
            if obj.M == 1
                return;
            end
            n = obj.nx;
            J = obj.idx;
            DU_p = {zeros(3, n), zeros(3, n)};
            DU_m = {zeros(3, n), zeros(3, n)};
            for ph = 1:2
                U0 = reshape(obj.U{ph}(:, 1, :), 3, n);
                DU_p{ph}(:, 1:n - 1) = U0(:, 2:n) - U0(:, 1:n - 1);
                DU_m{ph}(:, 2:n) = U0(:, 2:n) - U0(:, 1:n - 1);
            end
            if J - 2 >= 1
                DU_p{1}(:, J - 2) = obj.U_IC{1}(:, 1) - obj.U{1}(:, 1, J - 2);
            end
            if J + 2 <= n
                DU_m{2}(:, J + 2) = obj.U{2}(:, 1, J + 2) - obj.U_IC{2}(:, 1);
            end

            if obj.M == 2
                Br = obj.basis(obj.point(2:end), obj.ele_centroid, ...
                               obj.ele_vol);
                U_r1 = apply_basis(Br, obj.U{1});
                U_r2 = apply_basis(Br, obj.U{2});
                U0_1 = reshape(obj.U{1}(:, 1, :), 3, n);
                U0_2 = reshape(obj.U{2}(:, 1, :), 3, n);
                dUp1 = U_r1 - U0_1;
                dUp2 = U_r2 - U0_2;
                dUp_mod1 = minmod_tvb(dUp1, DU_p{1}, DU_m{1}, obj.tvb, ...
                                      obj.ele_vol);
                dUp_mod2 = minmod_tvb(dUp2, DU_p{2}, DU_m{2}, obj.tvb, ...
                                      obj.ele_vol);
                if J - 2 >= 1
                    tmp = obj.U{1};
                    tmp(:, 2, 1:J - 2) = reshape(dUp_mod1(:, 1:J - 2), ...
                                                 3, 1, []);
                    obj.U{1} = tmp;
                end
                if J + 2 <= n
                    tmp = obj.U{2};
                    tmp(:, 2, J + 2:n) = reshape(dUp_mod2(:, J + 2:n), ...
                                                 3, 1, []);
                    obj.U{2} = tmp;
                end

                U0IC0 = obj.U_IC{1}(:, 1);
                U0IC1 = obj.U_IC{2}(:, 1);
                Bl0 = obj.basis_single(obj.point(J - 1), ...
                                       obj.IC_centroid_new(1), ...
                                       obj.IC_vol_new(1));
                Br1 = obj.basis_single(obj.point(J + 2), ...
                                       obj.IC_centroid_new(2), ...
                                       obj.IC_vol_new(2));
                U_IC0_l = obj.U_IC{1} * Bl0;
                U_IC1_r = obj.U_IC{2} * Br1;
                dUp_IC0 = U0IC0 - U_IC0_l;
                dUp_IC1 = U_IC1_r - U0IC1;
                jump = U0IC1 - U0IC0;
                dUp_IC0_mod = minmod_tvb(dUp_IC0, ...
                                         DU_p{1}(:, J - 2), jump, obj.tvb, ...
                                         obj.IC_vol_new(1));
                dUp_IC1_mod = minmod_tvb(dUp_IC1, jump, ...
                                         DU_m{2}(:, J + 2), obj.tvb, ...
                                         obj.IC_vol_new(2));
                tmp1 = obj.U_IC{1};
                tmp1(:, 2) = dUp_IC0_mod;
                obj.U_IC{1} = tmp1;
                tmp2 = obj.U_IC{2};
                tmp2(:, 2) = dUp_IC1_mod;
                obj.U_IC{2} = tmp2;
                return;
            end

            % P2 特征变量 TVB 限制器
            Bl = obj.basis(obj.point(1:end - 1), obj.ele_centroid, ...
                           obj.ele_vol);
            Br = obj.basis(obj.point(2:end), obj.ele_centroid, obj.ele_vol);
            U_l1 = apply_basis(Bl, obj.U{1});
            U_l2 = apply_basis(Bl, obj.U{2});
            U_r1 = apply_basis(Br, obj.U{1});
            U_r2 = apply_basis(Br, obj.U{2});
            U0_1 = reshape(obj.U{1}(:, 1, :), 3, n);
            U0_2 = reshape(obj.U{2}(:, 1, :), 3, n);
            dUr1 = U_r1 - U0_1;
            dUr2 = U_r2 - U0_2;
            dUl1 = U0_1 - U_l1;
            dUl2 = U0_2 - U_l2;

            U1 = obj.U{1};
            for i = 1:J - 2
                R = char_matrix(U0_1(:, i), obj.gas{1});
                dUr_mod = R * minmod_tvb(R \ dUr1(:, i), ...
                                         R \ DU_p{1}(:, i), ...
                                         R \ DU_m{1}(:, i), obj.tvb, ...
                                         obj.ele_vol);
                dUl_mod = R * minmod_tvb(R \ dUl1(:, i), ...
                                         R \ DU_p{1}(:, i), ...
                                         R \ DU_m{1}(:, i), obj.tvb, ...
                                         obj.ele_vol);
                U1(:, 2, i) = 0.5 * (dUr_mod + dUl_mod);
                U1(:, 3, i) = 0.75 * (dUr_mod - dUl_mod);
            end
            obj.U{1} = U1;
            U2 = obj.U{2};
            for i = J + 2:n
                R = char_matrix(U0_2(:, i), obj.gas{2});
                dUr_mod = R * minmod_tvb(R \ dUr2(:, i), ...
                                         R \ DU_p{2}(:, i), ...
                                         R \ DU_m{2}(:, i), obj.tvb, ...
                                         obj.ele_vol);
                dUl_mod = R * minmod_tvb(R \ dUl2(:, i), ...
                                         R \ DU_p{2}(:, i), ...
                                         R \ DU_m{2}(:, i), obj.tvb, ...
                                         obj.ele_vol);
                U2(:, 2, i) = 0.5 * (dUr_mod + dUl_mod);
                U2(:, 3, i) = 0.75 * (dUr_mod - dUl_mod);
            end
            obj.U{2} = U2;

            % 界面单元
            U0IC0 = obj.U_IC{1}(:, 1);
            U0IC1 = obj.U_IC{2}(:, 1);
            Bl0 = obj.basis_single(obj.point(J - 1), ...
                                   obj.IC_centroid_new(1), ...
                                   obj.IC_vol_new(1));
            Br0 = obj.basis_single(obj.x_I, obj.IC_centroid_new(1), ...
                                   obj.IC_vol_new(1));
            Bl1 = obj.basis_single(obj.x_I, obj.IC_centroid_new(2), ...
                                   obj.IC_vol_new(2));
            Br1 = obj.basis_single(obj.point(J + 2), ...
                                   obj.IC_centroid_new(2), ...
                                   obj.IC_vol_new(2));
            dUr_1 = obj.U_IC{1} * Br0 - U0IC0;
            dUl_1 = U0IC0 - obj.U_IC{1} * Bl0;
            dUr_2 = obj.U_IC{2} * Br1 - U0IC1;
            dUl_2 = U0IC1 - obj.U_IC{2} * Bl1;
            jump = U0IC1 - U0IC0;

            for ph = 1:2
                if ph == 1
                    dUrp = dUr_1;
                    dUlp = dUl_1;
                    U0p = U0IC0;
                    b = DU_p{1}(:, J - 2);
                    c = jump;
                else
                    dUrp = dUr_2;
                    dUlp = dUl_2;
                    U0p = U0IC1;
                    b = jump;
                    c = DU_m{2}(:, J + 2);
                end
                R = char_matrix(U0p, obj.gas{ph});
                dUr_mod = R * minmod_tvb(R \ dUrp, R \ b, R \ c, ...
                                         obj.tvb, obj.IC_vol_new(ph));
                dUl_mod = R * minmod_tvb(R \ dUlp, R \ b, R \ c, ...
                                         obj.tvb, obj.IC_vol_new(ph));
                tmp = obj.U_IC{ph};
                tmp(:, 2) = 0.5 * (dUr_mod + dUl_mod);
                tmp(:, 3) = 0.75 * (dUr_mod - dUl_mod);
                obj.U_IC{ph} = tmp;
            end
        end

        % --------------------------------------------------------------
        % 驱动
        % --------------------------------------------------------------
        function res = run(obj)
            t_end = obj.cfg.time.t_end;
            cfl = obj.cfg.time.cfl;
            t = 0.0;
            dt = 0.0;
            steps = 0;

            while t < t_end
                U_cp = obj.U;
                UIC_cp = obj.U_IC;
                x_I_cp = obj.x_I;
                obj.U_data_cp = U_cp;
                obj.U_IC_cp = UIC_cp;
                obj.x_I_cp = x_I_cp;
                obj.phase_vol_cp = obj.phase_vol;
                obj.phase_centroid_cp = obj.phase_centroid;

                % ---- 第 1 级 ------------------------------------------
                obj.compose();
                u_star = obj.interface_velocity();
                [dU, dUIC, speed] = obj.rhs();
                speed = max(speed, abs(u_star));
                if ~isfinite(speed) || speed <= 0.0
                    error('TwoMediumSolver:speed', ...
                          'non-positive or non-finite maximum wave speed');
                end
                dt = cfl * obj.ele_vol / speed;
                if t + dt >= t_end
                    dt = t_end - t;
                end
                obj.x_I_new = obj.x_I + u_star * dt;
                obj.phi = obj.point - obj.x_I_new;
                obj.mark(obj.phi, true);
                obj.phase_info(obj.x_I_new, true);
                % 级更新与分解仍使用旧的界面单元模板，级末再重新居中。
                obj.interface_info(true, obj.idx);
                for ph = 1:2
                    obj.U{ph} = obj.U{ph} + dt * dU{ph};
                end
                obj.update_uic(dt, dUIC, 1);
                obj.limit();
                obj.decompose(obj.idx);
                if obj.M > 1
                    obj.backup_uic_cp(obj.idx_new);
                end
                obj.x_I = obj.x_I_new;
                obj.idx = obj.idx_new;
                obj.mark(obj.point - obj.x_I, false);
                obj.phase_info(obj.x_I, false);
                obj.interface_info(false, obj.idx);

                % ---- 第 2 级 ------------------------------------------
                if obj.M > 1
                    obj.compose();
                    u_star = obj.interface_velocity();
                    [dU, dUIC, ~] = obj.rhs();
                    if obj.M == 2
                        coef_cp = 0.5;
                        stage = 2;
                    else
                        coef_cp = 0.75;
                        stage = '3_2';
                    end
                    obj.x_I_new = coef_cp * x_I_cp + (1.0 - coef_cp) * ...
                                  (obj.x_I + u_star * dt);
                    obj.phi = obj.point - obj.x_I_new;
                    obj.mark(obj.phi, true);
                    obj.phase_info(obj.x_I_new, true);
                    obj.interface_info(true, obj.idx);
                    for ph = 1:2
                        obj.U{ph} = coef_cp * U_cp{ph} + ...
                                    (1.0 - coef_cp) * ...
                                    (obj.U{ph} + dt * dU{ph});
                    end
                    obj.update_uic(dt, dUIC, stage);
                    obj.limit();
                    obj.decompose(obj.idx);
                    if obj.M > 2
                        obj.backup_uic_cp(obj.idx_new);
                    end
                    obj.x_I = obj.x_I_new;
                    obj.idx = obj.idx_new;
                    obj.mark(obj.point - obj.x_I, false);
                    obj.phase_info(obj.x_I, false);
                    obj.interface_info(false, obj.idx);
                end

                % ---- 第 3 级 ------------------------------------------
                if obj.M > 2
                    obj.compose();
                    u_star = obj.interface_velocity();
                    [dU, dUIC, ~] = obj.rhs();
                    obj.x_I_new = (1.0 / 3.0) * x_I_cp + ...
                                  (2.0 / 3.0) * (obj.x_I + u_star * dt);
                    obj.phi = obj.point - obj.x_I_new;
                    obj.mark(obj.phi, true);
                    obj.phase_info(obj.x_I_new, true);
                    obj.interface_info(true, obj.idx);
                    for ph = 1:2
                        obj.U{ph} = (1.0 / 3.0) * U_cp{ph} + ...
                                    (2.0 / 3.0) * ...
                                    (obj.U{ph} + dt * dU{ph});
                    end
                    obj.update_uic(dt, dUIC, '3_3');
                    obj.limit();
                    obj.decompose(obj.idx);
                    obj.x_I = obj.x_I_new;
                    obj.idx = obj.idx_new;
                    obj.mark(obj.point - obj.x_I, false);
                end

                % 更新当前界面单元的几何与系数
                obj.phase_info(obj.x_I, false);
                obj.interface_info(false, obj.idx);
                obj.compose();
                t = t + dt;
                steps = steps + 1;
            end

            [rho, u, p] = obj.fields();
            res = make_result(obj.cfg, obj.ele_centroid, rho, u, p, dt, steps);
            res.interface = obj.x_I;
        end

        function [rho, u, p] = fields(obj)
            U1 = reshape(obj.U{1}(:, 1, :), 3, obj.nx);
            U2 = reshape(obj.U{2}(:, 1, :), 3, obj.nx);
            [rho0, u0, p0] = obj.gas{1}.primitive(U1);
            [rho1, u1, p1] = obj.gas{2}.primitive(U2);
            v0 = obj.phase_vol(1, :);
            v1 = obj.phase_vol(2, :);
            vtot = max(v0 + v1, obj.eps);
            rho = rho0 .* v0 ./ vtot + rho1 .* v1 ./ vtot;
            mom = rho0 .* u0 .* v0 ./ vtot + rho1 .* u1 .* v1 ./ vtot;
            u = mom ./ max(rho, obj.eps);
            p = p0 .* v0 ./ vtot + p1 .* v1 ./ vtot;
        end
    end
end

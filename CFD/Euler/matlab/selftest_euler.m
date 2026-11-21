function selftest_euler()
%SELFTEST_EULER 一维 Euler 求解器的快速自检
%
%   在 CFD/Euler/matlab 目录下运行：
%
%       selftest_euler
%
%   检查内容：
%     1. Gauss-Legendre 求积节点与权重；
%     2. minmod 限制器；
%     3. 刚性气体状态方程的守恒/原始变量互转与声速；
%     4. 经典 Sod 问题精确解的端点值与中间状态；
%     5. 完整算例：FV 单介质 Sod（与 Python 版本结果对比）；
%     6. DG P1 / P2 与双介质求解器的小规模短时运行。
%
%   全部检查通过时返回；失败项会逐条打印并以非零错误结束。
add_euler_paths();
root = euler_matlab_root();
chk = struct('failures', 0, 'total', 0);

fprintf('== 1. Gauss-Legendre 求积 ==\n');
[xg, wg] = gauss_legendre(4);
chk = check_close(chk, '节点个数', numel(xg), 4, 0);
chk = check_close(chk, '权重之和', sum(wg), 2, 1e-12);
chk = check_close(chk, '积分 x^2', sum(wg .* xg .^ 2), 2 / 3, 1e-12);
chk = check_close(chk, '积分 x^6', sum(wg .* xg .^ 6), 2 / 7, 1e-12);

fprintf('== 2. minmod 限制器 ==\n');
chk = check_close(chk, 'minmod2(2, 3)', minmod2(2, 3), 2, 0);
chk = check_close(chk, 'minmod2(-2, -3)', minmod2(-2, -3), -2, 0);
chk = check_close(chk, 'minmod2(1, -1)', minmod2(1, -1), 0, 0);

fprintf('== 3. 刚性气体状态方程 ==\n');
gas = StiffenedGas(1.4, 0.5);
U = gas.conservative(2.0, 3.0, 1.5);
[rho, u, p] = gas.primitive(U);
chk = check_close(chk, 'rho 往返', rho, 2.0, 1e-12);
chk = check_close(chk, 'u 往返', u, 3.0, 1e-12);
chk = check_close(chk, 'p 往返', p, 1.5, 1e-12);
chk = check_close(chk, '声速', gas.sound_speed(U), ...
                  sqrt(1.4 * (1.5 + 0.5) / 2.0), 1e-12);

fprintf('== 4. Sod 精确解 ==\n');
left = struct('rho', 1.0, 'u', 0.0, 'p', 1.0, 'gamma', 1.4);
right = struct('rho', 0.125, 'u', 0.0, 'p', 0.1, 'gamma', 1.4);
xq = [-0.5, -0.4, 0.0, 0.4, 0.5];
[re, ue, pe] = exact_riemann(left, right, xq, 0.2, 0.0);
chk = check_close(chk, '左端 rho', re(1), 1.0, 1e-12);
chk = check_close(chk, '左端 p', pe(1), 1.0, 1e-12);
chk = check_close(chk, '右端 rho', re(end), 0.125, 1e-12);
chk = check_close(chk, '右端 p', pe(end), 0.1, 1e-12);
chk = check_close(chk, '星区 p（x=0）', pe(3), 0.303130178050647, 1e-9);
chk = check_close(chk, '星区 rho_L（x=0）', re(3), 0.426319428178495, 1e-9);
chk = check_close(chk, '星区 u（x=0）', ue(3), 0.927452620048950, 1e-9);

fprintf('== 5. FV 单介质 Sod 完整算例 ==\n');
cfg = load_config(fullfile(root, 'configs', 'fv_single_sod.json'));
solver = FVSolver(cfg);
res = solver.run();
chk = check_positive(chk, 'FV Sod 结果', res.rho, res.u, res.p);
chk = check_range(chk, 'FV Sod 步数', res.steps, [320, 335]);
if isempty(res.exact_error)
    l1 = mean(abs(res.rho(:) - res.rho_exact(:)));
    chk = check_range(chk, 'FV Sod L1(rho)', l1, [2.5e-3, 3.7e-3]);
else
    chk = check_fail(chk, 'FV Sod 精确解', res.exact_error);
end

fprintf('== 6. DG / 双介质小规模算例 ==\n');
cfg = load_config(fullfile(root, 'configs', 'dg_single_sod.json'));
cfg.mesh.nx = 40;
cfg.time.t_end = 0.02;
cfg.time.cfl = 0.2;
cfg.order = 1;
res = DGSolver(cfg).run();
chk = check_positive(chk, 'DG P1 短时结果', res.rho, res.u, res.p);

cfg.order = 2;
res = DGSolver(cfg).run();
chk = check_positive(chk, 'DG P2 短时结果', res.rho, res.u, res.p);

cfg = load_config(fullfile(root, 'configs', 'fv_two_sod.json'));
cfg.mesh.nx = 40;
cfg.time.t_end = 0.03;
res = TwoMediumSolver(cfg).run();
chk = check_positive(chk, '双介质 FV 短时结果', res.rho, res.u, res.p);
% 界面随接触间断向右移动：0.5 + u* * t_end ≈ 0.528
chk = check_range(chk, '双介质界面位置', res.interface, [0.5, 0.56]);

fprintf('\n自检完成: %d/%d 项通过\n', ...
        chk.total - chk.failures, chk.total);
if chk.failures > 0
    error('selftest_euler:failed', '有 %d 项自检未通过', chk.failures);
end
end

% ----------------------------------------------------------------------
function chk = check_close(chk, label, got, want, rtol)
chk.total = chk.total + 1;
if ~isscalar(got) || ~isfinite(got)
    chk = check_fail(chk, label, '结果不是有限标量');
    return;
end
scale = max(abs(want), 1.0);
err = abs(got - want) / scale;
if err <= rtol
    fprintf('  [PASS] %-24s %.12g\n', label, got);
else
    fprintf('  [FAIL] %-24s got %.12g, want %.12g (err %.3g)\n', ...
            label, got, want, err);
    chk.failures = chk.failures + 1;
end
end

function chk = check_range(chk, label, got, bounds)
chk.total = chk.total + 1;
if isfinite(got) && got >= bounds(1) && got <= bounds(2)
    fprintf('  [PASS] %-24s %.12g\n', label, got);
else
    fprintf('  [FAIL] %-24s got %.12g, want [%g, %g]\n', ...
            label, got, bounds(1), bounds(2));
    chk.failures = chk.failures + 1;
end
end

function chk = check_positive(chk, label, rho, u, p)
chk.total = chk.total + 1;
ok = all(isfinite(rho)) && all(isfinite(u)) && all(isfinite(p)) && ...
     all(rho > 0) && all(p > 0);
if ok
    fprintf('  [PASS] %-24s rho in [%g, %g], p in [%g, %g]\n', ...
            label, min(rho), max(rho), min(p), max(p));
else
    fprintf('  [FAIL] %-24s 出现非有限值或非正密度/压强\n', label);
    chk.failures = chk.failures + 1;
end
end

function chk = check_fail(chk, label, msg)
chk.total = chk.total + 1;
chk.failures = chk.failures + 1;
fprintf('  [FAIL] %-24s %s\n', label, msg);
end

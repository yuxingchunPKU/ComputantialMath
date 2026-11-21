function selftest_riemann()
%SELFTEST_RIEMANN 黎曼精确解求解器的快速自检
%
%   在 CFD/Riemann/matlab 目录下运行：
%
%       selftest_riemann
%
%   检查内容：
%     1. 经典 Sod 问题的精确解（两侧 gamma = 1.4）；
%     2. Newton 与 Brent 两种求解方法的一致性；
%     3. 刚性气体匹配方程的残差；
%     4. 全部算例（非反应/爆轰/爆燃）的 p*、u* 与 Python 版本结果对比；
%     5. 各算例解的分辨率、有限性与正性。
%
%   全部检查通过时返回；失败项会逐条打印并以非零错误结束。
add_riemann_paths();
paths = riemann_discover_configs({});
chk = struct('failures', 0, 'total', 0);

fprintf('== 1. 经典 Sod 精确解（gamma = 1.4）==\n');
WL = Fluid(1.0, 0.0, 1.0, 1.4, 0.0);
WR = Fluid(0.125, 0.0, 0.1, 1.4, 0.0);
solver = ExactRiemannSolver(WL, WR, 1.0e-12, 1000);
[p_star, u_star] = solver.solver();
chk = check_close(chk, 'p*', p_star, 0.303130178050647, 1e-9);
chk = check_close(chk, 'u*', u_star, 0.927452620048950, 1e-9);

fprintf('== 2. Newton / Brent 一致性 ==\n');
[p_brent, u_brent] = solver.solver_brent();
chk = check_close(chk, 'p* (Brent)', p_brent, p_star, 1e-8);
chk = check_close(chk, 'u* (Brent)', u_brent, u_star, 1e-8);

fprintf('== 3. 刚性气体匹配方程残差 ==\n');
WLs = Fluid(1.0, 0.0, 1.0, 1.4, 0.5);
WRs = Fluid(0.125, 0.0, 0.1, 1.667, 0.2);
solver_s = ExactRiemannSolver(WLs, WRs, 1.0e-12, 1000);
[ps, us] = solver_s.solver();
residual = solver_s.f_K_p(ps, WLs) + solver_s.f_K_p(ps, WRs) ...
           + WRs.u - WLs.u;
chk = check_close(chk, '匹配方程残差', residual, 0.0, 1e-10);
% 左波关系: u* = u_L - f_L(p*)
chk = check_close(chk, '左波速度关系', ...
                  us + solver_s.f_K_p(ps, WLs), WLs.u, 1e-10);

fprintf('== 4. 全部算例与 Python 结果对比 ==\n');
refs = reference_values();
for k = 1:numel(paths)
    cfg = load_riemann_config(paths{k});
    cases = cfg.cases;
    for j = 1:numel(cases)
        cs = cases{j};
        switch cs.type
            case 'inert'
                [x, rho, u, p, star, waves] = compute_inert(cs);
            case 'detonation'
                [x, rho, u, p, star, waves] = compute_detonation(cs);
            case 'deflagration'
                [x, rho, u, p, star, waves] = compute_deflagration(cs);
            otherwise
                error('selftest_riemann:type', '未知算例类型: %s', cs.type);
        end
        ref = refs.(cs.name);
        chk = check_close(chk, [cs.name ' p*'], star.p_star, ref.p, 1e-6);
        chk = check_close(chk, [cs.name ' u*'], star.u_star, ref.u, 1e-6);
        chk = check_profile(chk, cs.name, x, rho, u, p);
        chk = check_waves(chk, cs.name, waves);
        if isfield(star, 'rho_star_L')
            chk = check_positive_scalar(chk, [cs.name ' rho*_L'], ...
                                        star.rho_star_L);
        end
        if isfield(star, 'rho_star_R')
            chk = check_positive_scalar(chk, [cs.name ' rho*_R'], ...
                                        star.rho_star_R);
        end
    end
end

fprintf('\n自检完成: %d/%d 项通过\n', ...
        chk.total - chk.failures, chk.total);
if chk.failures > 0
    error('selftest_riemann:failed', '有 %d 项自检未通过', chk.failures);
end
end

% ----------------------------------------------------------------------
function refs = reference_values()
refs = struct();
refs.inert_sod = struct('p', 0.3143966584, 'u', 0.9013775087);
refs.inert_stiffened_gas = struct('p', 0.2938748334, 'u', 0.6295798596);
refs.detonation_strong = struct('p', 7.70752512e6, 'u', 2.79981870e3);
refs.detonation_cj = struct('p', 4.04266012e6, 'u', 1.58604994e3);
refs.deflagration = struct('p', 9.45695111e4, 'u', -1.81018081e2);
refs.deflagration_uL0 = struct('p', 1.15181901e5, 'u', -1.37909786e2);
end

function chk = check_close(chk, label, got, want, rtol)
chk.total = chk.total + 1;
if ~isscalar(got) || ~isfinite(got)
    chk = check_fail(chk, label, '结果不是有限标量');
    return;
end
scale = max(abs(want), 1.0);
err = abs(got - want) / scale;
if err <= rtol
    fprintf('  [PASS] %-28s %.12g\n', label, got);
else
    fprintf('  [FAIL] %-28s got %.12g, want %.12g (err %.3g)\n', ...
            label, got, want, err);
    chk.failures = chk.failures + 1;
end
end

function chk = check_positive_scalar(chk, label, value)
chk.total = chk.total + 1;
if isscalar(value) && isfinite(value) && value > 0
    fprintf('  [PASS] %-28s %.12g\n', label, value);
else
    fprintf('  [FAIL] %-28s 非正或非有限值\n', label);
    chk.failures = chk.failures + 1;
end
end

function chk = check_profile(chk, name, x, rho, u, p)
chk.total = chk.total + 1;
n = numel(x);
ok = numel(rho) == n && numel(u) == n && numel(p) == n && ...
     all(isfinite(x)) && all(isfinite(rho)) && all(isfinite(u)) && ...
     all(isfinite(p)) && all(rho > 0) && all(p > 0);
if ok
    fprintf('  [PASS] %-28s N=%d, rho in [%g, %g]\n', ...
            [name ' 解'], n, min(rho), max(rho));
else
    fprintf('  [FAIL] %-28s 长度不一致或出现非有限/非正值\n', [name ' 解']);
    chk.failures = chk.failures + 1;
end
end

function chk = check_waves(chk, name, waves)
chk.total = chk.total + 1;
ok = true;
for k = 1:numel(waves)
    fns = fieldnames(waves{k}.speeds);
    for j = 1:numel(fns)
        ok = ok && isfinite(waves{k}.speeds.(fns{j}));
    end
end
if ok
    fprintf('  [PASS] %-28s %d 道波\n', [name ' 波速'], numel(waves));
else
    fprintf('  [FAIL] %-28s 存在非有限波速\n', [name ' 波速']);
    chk.failures = chk.failures + 1;
end
end

function chk = check_fail(chk, label, msg)
chk.total = chk.total + 1;
chk.failures = chk.failures + 1;
fprintf('  [FAIL] %-28s %s\n', label, msg);
end

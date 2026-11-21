function result = run_riemann_case(cs, fig_dir, plot_cfg, verbose)
%RUN_RIEMANN_CASE 求解一个算例、检查结果并生成图像
%
%   对应 Python 版本 code/run_cases.py 中的 run_case。
if nargin < 3 || isempty(plot_cfg)
    plot_cfg = struct();
end
if nargin < 4
    verbose = true;
end

switch cs.type
    case 'inert'
        [x, rho, u, p, star, waves, info] = compute_inert(cs);
    case 'detonation'
        [x, rho, u, p, star, waves, info] = compute_detonation(cs);
    case 'deflagration'
        [x, rho, u, p, star, waves, info] = compute_deflagration(cs);
    otherwise
        error('run_riemann_case:type', '未知算例类型: %s', cs.type);
end

if ~all(isfinite([rho(:); u(:); p(:)]))
    error('run_riemann_case:finite', ...
          '算例 %s 的计算结果包含非有限值', cs.name);
end
if any(rho <= 0.0)
    error('run_riemann_case:rho', '算例 %s 出现非正密度', cs.name);
end
if any(p <= 0.0)
    error('run_riemann_case:p', '算例 %s 出现非正压强', cs.name);
end

case_fig_dir = fullfile(fig_dir, cs.name);
if exist(case_fig_dir, 'dir') ~= 7
    mkdir(case_fig_dir);
end
figures = plot_riemann_case(cs, x, rho, u, p, waves, case_fig_dir, plot_cfg);

if verbose
    fprintf('[%s] p* = %.6e, u* = %.6e\n', cs.name, star.p_star, ...
            star.u_star);
    for k = 1:numel(waves)
        w = waves{k};
        fns = fieldnames(w.speeds);
        parts = cell(1, numel(fns));
        for j = 1:numel(fns)
            parts{j} = sprintf('%s=%.6e', fns{j}, w.speeds.(fns{j}));
        end
        fprintf('    %s (%s): %s\n', w.name, w.type, strjoin(parts, ', '));
    end
    fprintf('    图像已保存: %s\n', case_fig_dir);
end

result = struct();
result.name = cs.name;
result.description = cs.description;
result.type = cs.type;
result.case = cs;
result.star = star;
result.waves = waves;
result.info = info;
result.figures = figures;
result.solution_range = struct( ...
    'rho', [min(rho), max(rho)], ...
    'u', [min(u), max(u)], ...
    'p', [min(p), max(p)], ...
    'N', numel(x));
end

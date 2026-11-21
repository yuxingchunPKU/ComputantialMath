function item = run_euler_case(cfg, make_plots)
%RUN_EULER_CASE 运行一个算例，并（可选）生成图像
%
%   对应 Python 版本 euler_unified/run.py 中的 run_case。
if nargin < 2
    make_plots = true;
end
add_euler_paths();

if strcmp(cfg.medium, 'single')
    if strcmp(cfg.scheme, 'fv')
        solver = FVSolver(cfg);
    else
        solver = DGSolver(cfg);
    end
else
    solver = TwoMediumSolver(cfg);
end

res = solver.run();
figures = struct();
if make_plots
    figures = plot_euler_case(cfg, res);
end

item = struct();
item.cfg = cfg;
item.result = res;
item.figures = figures;
end

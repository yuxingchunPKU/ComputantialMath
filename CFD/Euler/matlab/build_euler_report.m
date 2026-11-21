function text = build_euler_report(items, report_path, config_paths)
%BUILD_EULER_REPORT 生成 Markdown 测试报告
%
%   items 为 run_euler_case 返回值组成的元胞数组。
%   对应 Python 版本 euler_unified/run.py 中的 build_report。
if nargin < 3
    config_paths = {};
end

now = datestr(now, 'yyyy-mm-dd HH:MM:SS');
names = cell(1, numel(config_paths));
for k = 1:numel(config_paths)
    [~, stem, ext] = fileparts(char(config_paths{k}));
    names{k} = ['`', stem, ext, '`'];
end
config_list = strjoin(names, '、');

lines = {};
lines{end + 1} = '# 一维 Euler 方程统一求解器测试报告';
lines{end + 1} = '';
lines{end + 1} = ['本报告由统一的 FV/DG 求解器自动生成。程序支持单介质与双介质' ...
                  '（单个移动界面）一维 Euler 方程组，数值格式、网格、流体参数' ...
                  '与初始条件均由 JSON 配置。'];
lines{end + 1} = '';
lines{end + 1} = sprintf('> 由 `run_euler.m` 读取 %s 自动生成。', config_list);
lines{end + 1} = sprintf('> 生成时间：%s', now);
lines{end + 1} = '';
lines{end + 1} = '## 算例汇总';
lines{end + 1} = '';
lines{end + 1} = ['| # | 算例 | 格式 | 介质 | 阶数 | 通量 | 单元数 | 终止时间 | ' ...
                  '步数 | 界面位置 | rho L1 | rho L∞ |'];
lines{end + 1} = ['| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ' ...
                  '---: | ---: | ---: |'];
for index = 1:numel(items)
    lines{end + 1} = summary_row(index, items{index}); %#ok<AGROW>
end
lines{end + 1} = '';
lines{end + 1} = ['数值解以蓝色实线表示，精确解以橙色虚线表示。双介质算例的界面位置由' ...
                  '界面黎曼问题推进，并在表中给出终了位置。'];
lines{end + 1} = '';
lines{end + 1} = '## 数值方法';
lines{end + 1} = '';
lines{end + 1} = ['- 单介质有限体积：一阶 LLF/HLL/HLLC，或 MUSCL + minmod 二阶重构；' ...
                  '时间推进采用 SSP-RK2。'];
lines{end + 1} = ['- 单介质 DG：P0/P1/P2 Legendre 基，LLF/HLL/HLLC 数值通量，' ...
                  '特征变量 TVB/minmod 限制器，SSP-RK2/RK3。'];
lines{end + 1} = ['- 双介质：界面所在单元用两相切割单元表示，界面速度由刚性气体精确' ...
                  '黎曼求解器给出，相内数据采用 P0（等价于一阶 FV）或 P1/P2 DG 多项式。'];
lines{end + 1} = '';
lines{end + 1} = '## 算例结果';
lines{end + 1} = '';
for index = 1:numel(items)
    section = case_section(index, items{index}, report_path);
    lines = [lines, section]; %#ok<AGROW>
    lines{end + 1} = ''; %#ok<AGROW>
end

lines{end + 1} = '## 复现方式';
lines{end + 1} = '';
lines{end + 1} = '```matlab';
lines{end + 1} = 'cd CFD/Euler/matlab';
lines{end + 1} = 'run_euler                       % 运行 configs/ 下全部算例并生成 report.md';
lines{end + 1} = 'run_euler --list                % 列出全部算例';
lines{end + 1} = 'run_euler --case fv_single_sod';
lines{end + 1} = 'run_euler --config configs/fv_single_sod.json';
lines{end + 1} = '```';
lines{end + 1} = '';
lines{end + 1} = '新增算例只需在 `configs/` 下添加 JSON 文件，无需修改求解器代码。';
lines{end + 1} = '';

text = strjoin(lines, sprintf('\n'));
end

% ----------------------------------------------------------------------
function row = summary_row(index, item)
cfg = item.cfg;
res = item.result;
two = strcmp(cfg.medium, 'two');
if two
    interface = fmt_num(get_field(res, 'interface', []));
else
    interface = '—';
end
err = euler_error_metrics(res);
row = sprintf('| %d | %s | %s | %s | %d | %s | %d | %s | %d | %s | %s | %s |', ...
              index, cfg.name, upper(cfg.scheme), ...
              choose(two, '多介质', '单介质'), cfg.order, ...
              upper(cfg.flux), cfg.mesh.nx, fmt_num(cfg.time.t_end), ...
              res.steps, interface, ...
              fmt_num(err.rho(1)), fmt_num(err.rho(2)));
end

function section = case_section(index, item, report_path)
cfg = item.cfg;
res = item.result;
figures = item.figures;
two = strcmp(cfg.medium, 'two');
err = euler_error_metrics(res);

lines = {};
lines{end + 1} = sprintf('## %d. %s', index, cfg.name);
lines{end + 1} = '';
if isfield(cfg, 'description') && ~isempty(cfg.description)
    lines{end + 1} = cfg.description;
    lines{end + 1} = '';
end

lines{end + 1} = '**计算参数**';
lines{end + 1} = '';
lines{end + 1} = '| 参数 | 值 |';
lines{end + 1} = '| --- | ---: |';
lines{end + 1} = sprintf('| 格式 | %s，%d 阶 |', upper(cfg.scheme), cfg.order);
lines{end + 1} = sprintf('| 介质 | %s |', ...
                         choose(two, '双介质（单个移动界面）', '单介质'));
lines{end + 1} = sprintf('| 数值通量 | %s |', upper(cfg.flux));
lines{end + 1} = sprintf('| 计算区域 | $x\\in[%s,%s]$，%d 个单元 |', ...
                         fmt_num(cfg.mesh.x_min), fmt_num(cfg.mesh.x_max), ...
                         cfg.mesh.nx);
lines{end + 1} = sprintf('| 边界条件 | %s |', cfg.mesh.boundary);
lines{end + 1} = sprintf('| 终止时间 / CFL | %s / %s |', ...
                         fmt_num(cfg.time.t_end), fmt_num(cfg.time.cfl));
lines{end + 1} = sprintf('| 时间步数 | %d |', res.steps);
if two
    lines{end + 1} = sprintf('| 界面终了位置 | %s |', ...
                             fmt_num(get_field(res, 'interface', [])));
end
lines{end + 1} = '';
lines{end + 1} = '**流体参数**';
lines{end + 1} = '';
lines = [lines, fluid_table(cfg)]; %#ok<AGROW>
lines{end + 1} = '**初始条件**';
lines{end + 1} = '';
lines = [lines, state_table(cfg)]; %#ok<AGROW>
lines{end + 1} = '**误差（相对精确解）**';
lines{end + 1} = '';
lines{end + 1} = '| 量 | L1 | L∞ |';
lines{end + 1} = '| --- | ---: | ---: |';
labels = struct('rho', '密度', 'u', '速度', 'p', '压强');
keys = {'rho', 'u', 'p'};
for k = 1:numel(keys)
    key = keys{k};
    lines{end + 1} = sprintf('| %s | %s | %s |', labels.(key), ...
                             fmt_num(err.(key)(1)), fmt_num(err.(key)(2))); %#ok<AGROW>
end
if isfield(res, 'exact_error') && ~isempty(res.exact_error)
    lines{end + 1} = '';
    lines{end + 1} = sprintf('> 精确解未能计算：`%s`', res.exact_error);
end
lines{end + 1} = '';
lines{end + 1} = '**结果图像**';
lines{end + 1} = '';

if isfield(figures, 'rho') && isfield(figures, 'u') && isfield(figures, 'p')
    base = fileparts(absolute_path(report_path));
    rho = relative_to(figures.rho, base);
    u = relative_to(figures.u, base);
    p = relative_to(figures.p, base);
    lines{end + 1} = '| 密度 $\rho$ | 速度 $u$ | 压强 $p$ |';
    lines{end + 1} = '| :---: | :---: | :---: |';
    lines{end + 1} = sprintf(['| <img src="%s" alt="%s 密度" width="280"> ' ...
                              '| <img src="%s" alt="%s 速度" width="280"> ' ...
                              '| <img src="%s" alt="%s 压强" width="280"> |'], ...
                             rho, cfg.name, u, cfg.name, p, cfg.name);
    lines{end + 1} = '';
    if isfield(figures, 'overview')
        overview = relative_to(figures.overview, base);
        lines{end + 1} = sprintf('三联图：[overview.png](%s)', overview);
    end
else
    lines{end + 1} = '> 未生成图像。';
end
lines{end + 1} = '';
section = lines;
end

function out = fluid_table(cfg)
if strcmp(cfg.medium, 'single')
    f = cfg.fluids;
    out = {'| 参数 | 值 |', '| --- | ---: |', ...
           sprintf('| gamma | %s |', fmt_num(f.gamma)), ...
           sprintf('| p_inf | %s |', fmt_num(f.p_inf))};
else
    l = cfg.fluids.left;
    r = cfg.fluids.right;
    out = {'| 参数 | 左介质 | 右介质 |', '| --- | ---: | ---: |', ...
           sprintf('| gamma | %s | %s |', fmt_num(l.gamma), fmt_num(r.gamma)), ...
           sprintf('| p_inf | %s | %s |', fmt_num(l.p_inf), fmt_num(r.p_inf))};
end
end

function out = state_table(cfg)
initial = cfg.initial;
if strcmp(initial.type, 'smooth')
    out = {'| 参数 | 值 |', '| --- | ---: |', ...
           sprintf(['| 密度 | $\\rho_0 + \\rho_a\\sin(k\\pi x)$，' ...
                    '$\\rho_0=%s$，$\\rho_a=%s$，$k=%s$ |'], ...
                   fmt_num(initial.rho0), fmt_num(initial.rho_amp), ...
                   fmt_num(initial.k)), ...
           sprintf('| 速度 | %s |', fmt_num(initial.u0)), ...
           sprintf('| 压强 | %s |', fmt_num(initial.p0))};
else
    left = initial.left;
    right = initial.right;
    out = {'| 参数 | 左状态 | 右状态 |', '| --- | ---: | ---: |', ...
           sprintf('| rho | %s | %s |', fmt_num(left.rho), fmt_num(right.rho)), ...
           sprintf('| u | %s | %s |', fmt_num(left.u), fmt_num(right.u)), ...
           sprintf('| p | %s | %s |', fmt_num(left.p), fmt_num(right.p))};
end
end

function out = choose(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

function v = get_field(s, name, default)
if isfield(s, name)
    v = s.(name);
else
    v = default;
end
end

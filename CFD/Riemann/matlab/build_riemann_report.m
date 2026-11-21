function text = build_riemann_report(results, report_path, config_paths, meta)
%BUILD_RIEMANN_REPORT 生成黎曼问题精确解的 Markdown 测试报告
%
%   对应 Python 版本 code/run_cases.py 中的 build_report。
if nargin < 3
    config_paths = {};
end
if nargin < 4 || isempty(meta)
    meta = default_meta();
end
report_dir = fileparts(absolute_path(report_path));

title = '刚性气体黎曼问题精确解测试报告';
if isfield(meta, 'title') && ~isempty(meta.title)
    title = meta.title;
end
description = '';
if isfield(meta, 'description') && ~isempty(meta.description)
    description = meta.description;
end

lines = {};
lines{end + 1} = ['# ', title];
lines{end + 1} = '';
if ~isempty(description)
    lines{end + 1} = description;
    lines{end + 1} = '';
end
lines{end + 1} = sprintf('> 由 `matlab/run_cases.m` 读取 %s 自动生成。', ...
                         format_config_paths(config_paths));
lines{end + 1} = sprintf('> 生成时间：%s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
lines{end + 1} = '';

lines{end + 1} = '## 算例汇总';
lines{end + 1} = '';
lines{end + 1} = '| # | 算例 | 类型 | p* (Pa) | u* (m/s) | 左波 | 右波 |';
lines{end + 1} = '| ---: | --- | --- | ---: | ---: | --- | --- |';
for k = 1:numel(results)
    r = results{k};
    lines{end + 1} = sprintf('| %d | %s | %s | %s | %s | %s | %s |', ...
                             k, r.name, type_label(r.type), ...
                             fmt_num_fixed(r.star.p_star), ...
                             fmt_num_fixed(r.star.u_star), ...
                             r.waves{1}.type, r.waves{end}.type); %#ok<AGROW>
end
lines{end + 1} = '';
lines{end + 1} = '> 表中 p*、u* 对爆燃算例指火焰波后的中间状态 (p1, u1)。';
lines{end + 1} = '';

lines{end + 1} = '## 求解方法';
lines{end + 1} = '';
lines{end + 1} = '控制方程为无粘一维 Euler 方程组。对于刚性气体，状态方程为';
lines{end + 1} = '';
lines{end + 1} = '$$';
lines{end + 1} = 'p = (\gamma-1)\rho e - \gamma p_\infty .';
lines{end + 1} = '$$';
lines{end + 1} = '';
lines{end + 1} = ['非反应波（激波/稀疏波）的匹配方程 $f_L(p)+f_R(p)+u_R-u_L=0$ ' ...
                  '分别采用 Newton 迭代和 Brent 方法求解；'];
lines{end + 1} = '反应流算例在匹配方程中引入爆轰/爆燃的 Rayleigh-Hugoniot 关系。';
lines{end + 1} = '所有图像由精确解在均匀网格上直接采样得到，竖直虚线标出各道波的位置。';
lines{end + 1} = ['若某个 $p^*$ 与对应端部状态的压强之差在求解精度以内，则该波为' ...
                  '零强度波（波速等于当地声速附近的特征速度），图像上不会出现间断。'];
lines{end + 1} = '';

for k = 1:numel(results)
    lines = [lines, case_lines(k, results{k}, report_dir)]; %#ok<AGROW>
    lines{end + 1} = ''; %#ok<AGROW>
end

lines{end + 1} = '## 复现方式';
lines{end + 1} = '';
lines{end + 1} = '```matlab';
lines{end + 1} = 'cd CFD/Riemann/matlab';
lines{end + 1} = 'run_cases                                          % 运行全部算例';
lines{end + 1} = 'run_cases --config cases/inert.json                % 只运行非反应算例';
lines{end + 1} = 'run_cases --config cases/detonation.json           % 只运行爆轰算例';
lines{end + 1} = 'run_cases --config cases/deflagration.json         % 只运行爆燃算例';
lines{end + 1} = 'run_cases --list                                   % 列出全部算例';
lines{end + 1} = '```';
lines{end + 1} = '';
lines{end + 1} = '新算例按类型追加到 `cases/` 下对应的 JSON 文件中，无需修改代码。';
lines{end + 1} = '';

text = strjoin(lines, sprintf('\n'));
end

% ----------------------------------------------------------------------
function meta = default_meta()
meta = struct();
meta.title = '刚性气体与反应流黎曼问题精确解测试报告';
meta.description = ['一维 Euler 方程组的精确 Riemann 求解器测试。包括非反应刚性气体、' ...
                    '爆燃波与爆轰波算例。'];
end

function label = type_label(kind)
switch kind
    case 'inert'
        label = '非反应（刚性气体）';
    case 'detonation'
        label = '爆轰';
    case 'deflagration'
        label = '爆燃';
    otherwise
        label = kind;
end
end

function text = format_config_paths(paths)
base = fileparts(riemann_matlab_root());
labels = cell(1, numel(paths));
for k = 1:numel(paths)
    labels{k} = ['`', relative_to(paths{k}, base), '`'];
end
text = strjoin(labels, '、');
end

function lines = case_lines(index, r, report_dir)
cs = r.case;
figures = r.figures;

lines = {};
lines{end + 1} = sprintf('## %d. %s', index, r.name);
lines{end + 1} = '';
if ~isempty(r.description)
    lines{end + 1} = r.description;
    lines{end + 1} = '';
end
lines{end + 1} = '**初始条件与计算参数**';
lines{end + 1} = '';
lines = [lines, state_table(cs)]; %#ok<AGROW>
lines{end + 1} = '';
lines{end + 1} = '**求解结果**';
lines{end + 1} = '';
lines = [lines, star_table(r)]; %#ok<AGROW>
lines{end + 1} = '';
lines{end + 1} = '**波系**';
lines{end + 1} = '';
lines = [lines, wave_table(r.waves)]; %#ok<AGROW>
lines{end + 1} = '';
lines{end + 1} = '**结果图像**';
lines{end + 1} = '';
lines{end + 1} = '| 密度 $\rho$ | 速度 $u$ | 压强 $p$ |';
lines{end + 1} = '| :---: | :---: | :---: |';
keys = {'rho', 'u', 'p'};
labels = {'密度', '速度', '压强'};
cells = cell(1, 3);
for k = 1:3
    link = relative_to(figures.(keys{k}), report_dir);
    cells{k} = sprintf('<img src="%s" alt="%s %s" width="280">', ...
                       link, r.name, labels{k});
end
lines{end + 1} = ['| ', strjoin(cells, ' | '), ' |'];
lines{end + 1} = '';
lines{end + 1} = sprintf('三联图：[overview.png](%s)', ...
                         relative_to(figures.overview, report_dir));
end

function lines = state_table(cs)
left = cs.left;
right = cs.right;
reactive = any(strcmp(cs.type, {'detonation', 'deflagration'}));
if reactive
    extra_label = 'q (J/kg)';
    extra_key = 'q';
else
    extra_label = 'p_inf (Pa)';
    extra_key = 'p_inf';
end
rows = {'rho (kg/m^3)', 'rho'; 'u (m/s)', 'u'; 'p (Pa)', 'p'; ...
        'gamma', 'gamma'; extra_label, extra_key};
lines = {'| 参数 | 左状态 | 右状态 |', '| --- | ---: | ---: |'};
for k = 1:size(rows, 1)
    lv = struct_value(left, rows{k, 2});
    rv = struct_value(right, rows{k, 2});
    lines{end + 1} = sprintf('| %s | %s | %s |', rows{k, 1}, ...
                             fmt_num_fixed(lv), fmt_num_fixed(rv)); %#ok<AGROW>
end
d = cs.domain;
lines{end + 1} = sprintf(['| 计算区域 | x = [%s, %s], x_c = %s | ' ...
                          't = %s, N = %d |'], ...
                         fmt_num_fixed(d.x_L), fmt_num_fixed(d.x_R), ...
                         fmt_num_fixed(d.x_c), fmt_num_fixed(d.t_end), ...
                         d.N);
num = cs.numerical;
lines{end + 1} = sprintf('| 数值参数 | %s, tol = %s, maxit = %d | |', ...
                         num.method, fmt_num_fixed(num.tol), num.maxit);
end

function v = struct_value(s, name)
if isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = 0.0;
end
end

function lines = star_table(r)
star = r.star;
switch r.type
    case 'inert'
        rows = {'中间压强 p* (Pa)', 'p_star'; ...
                '接触间断速度 u* (m/s)', 'u_star'; ...
                '左波后密度 rho*_L (kg/m^3)', 'rho_star_L'; ...
                '右波后密度 rho*_R (kg/m^3)', 'rho_star_R'};
    case 'detonation'
        rows = {'中间压强 p* (Pa)', 'p_star'; ...
                '接触间断速度 u* (m/s)', 'u_star'; ...
                '左波后密度 rho*_L (kg/m^3)', 'rho_star_L'; ...
                '右波后密度 rho*_R (kg/m^3)', 'rho_star_R'; ...
                'CJ 压强 p_CJ (Pa)', 'p_cj'; ...
                'CJ 速度 u_CJ (m/s)', 'u_cj'; ...
                'CJ 密度 rho_CJ (kg/m^3)', 'rho_cj'};
    otherwise
        rows = {'火焰波后压强 p1 (Pa)', 'p_star'; ...
                '接触间断速度 u1 (m/s)', 'u_star'; ...
                '火焰波后密度 rho1 (kg/m^3)', 'rho1'; ...
                '火焰波前压强 p0 (Pa)', 'p0'; ...
                '火焰波前速度 u0 (m/s)', 'u0'; ...
                '火焰波前密度 rho0 (kg/m^3)', 'rho0'; ...
                '火焰速度 V (m/s)', 'V'; ...
                '左波后密度 rho*_L (kg/m^3)', 'rho_star_L'};
end
lines = {'| 量 | 值 |', '| --- | ---: |'};
for k = 1:size(rows, 1)
    lines{end + 1} = sprintf('| %s | %s |', rows{k, 1}, ...
                             fmt_num_fixed(star.(rows{k, 2}))); %#ok<AGROW>
end
end

function lines = wave_table(waves)
lines = {'| 波 | 类型 | 速度 (m/s) |', '| --- | --- | --- |'};
for k = 1:numel(waves)
    w = waves{k};
    fns = fieldnames(w.speeds);
    parts = cell(1, numel(fns));
    for j = 1:numel(fns)
        parts{j} = sprintf('%s = %s', fns{j}, ...
                           fmt_num_fixed(w.speeds.(fns{j})));
    end
    lines{end + 1} = sprintf('| %s | %s | %s |', w.name, w.type, ...
                             strjoin(parts, ', ')); %#ok<AGROW>
end
end

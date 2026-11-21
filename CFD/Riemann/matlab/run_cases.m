function run_cases(varargin)
%RUN_CASES 一维刚性气体 / 反应流黎曼问题精确解的算例驱动
%
%   用法（在 CFD/Riemann/matlab 目录下）：
%       run_cases
%       run_cases --list
%       run_cases --case inert_sod
%       run_cases --case inert_sod,detonation_cj
%       run_cases --config cases/inert.json
%       run_cases --config cases/detonation.json cases/deflagration.json
%       run_cases --report ../report.md
%
%   对应 Python 版本 CFD/Riemann/code/run_cases.py。
add_riemann_paths();

opts = parse_args(varargin);
root = riemann_matlab_root();

if isfield(opts, 'config')
    config_args = opts.config;
else
    config_args = {};
end

config_paths = riemann_discover_configs(config_args);
configs = cell(1, numel(config_paths));
for k = 1:numel(config_paths)
    configs{k} = load_riemann_config(config_paths{k});
end

% 合并全部算例并检查名称全局唯一
cases = {};
by_name = struct();
for k = 1:numel(configs)
    list = configs{k}.cases;
    for j = 1:numel(list)
        cs = list{j};
        if isfield(by_name, cs.name)
            error('run_cases:dup', '算例名称重复: %s（%s 与 %s）', ...
                  cs.name, by_name.(cs.name), cs.source_file);
        end
        by_name.(cs.name) = cs.source_file;
        cases{end + 1} = cs; %#ok<AGROW>
    end
end

if isfield(opts, 'list') && opts.list
    for k = 1:numel(cases)
        cs = cases{k};
        [~, src, ext] = fileparts(cs.source_file);
        fprintf('%-24s %-14s %-20s %s\n', cs.name, cs.type, ...
                [src ext], cs.description);
    end
    return;
end

% 按名称筛选
if isfield(opts, 'case')
    raw = opts.case;
    wanted = {};
    for k = 1:numel(raw)
        parts = strsplit(raw{k}, ',');
        for j = 1:numel(parts)
            name = strtrim(parts{j});
            if ~isempty(name)
                wanted{end + 1} = name; %#ok<AGROW>
            end
        end
    end
    selected = cell(1, numel(wanted));
    for k = 1:numel(wanted)
        name = wanted{k};
        if ~isfield(by_name, name)
            error('run_cases:case', ...
                  '未知算例 %s。可选算例: %s', name, ...
                  strjoin(sort(fieldnames(by_name)), ', '));
        end
        selected{k} = lookup_case(cases, name);
    end
    cases = selected;
end

% 输出目录
if isfield(opts, 'fig_dir') && ischar(opts.fig_dir)
    fig_dir = absolute_path(opts.fig_dir);
else
    fig_dir = fullfile(root, 'fig');
end
if exist(fig_dir, 'dir') ~= 7
    mkdir(fig_dir);
end

% 报告路径
default_meta = struct();
default_meta.title = '刚性气体与反应流黎曼问题精确解测试报告';
default_meta.description = ['一维 Euler 方程组的精确 Riemann 求解器测试。包括非反应刚性气体、' ...
                            '爆燃波与爆轰波算例。'];

report_path = '';
if isfield(opts, 'report') && ischar(opts.report)
    report_path = absolute_path(opts.report);
elseif (isfield(opts, 'config') && numel(config_paths) == 1)
    [~, stem, ~] = fileparts(config_paths{1});
    report_path = fullfile(root, ['report_' stem '.md']);
else
    report_path = fullfile(root, 'report.md');
end

quiet = isfield(opts, 'quiet') && opts.quiet;
results = cell(1, numel(cases));
for k = 1:numel(cases)
    cs = cases{k};
    plot_cfg = deep_merge(default_plot_cfg(), cs.plot);
    results{k} = run_riemann_case(cs, fig_dir, plot_cfg, ~quiet);
end

% 只运行了部分算例或部分配置文件时，默认不覆盖完整报告
no_report = isfield(opts, 'no_report') && opts.no_report;
if ~no_report
    subset_configs = false;
    if isfield(opts, 'config')
        try
            default_paths = riemann_discover_configs({});
        catch
            default_paths = {};
        end
        subset_configs = numel(config_paths) ~= numel(default_paths) || ...
            ~isempty(setxor(cellfun(@absolute_path, config_paths, ...
                                    'uniformoutput', false), ...
                            cellfun(@absolute_path, default_paths, ...
                                    'uniformoutput', false)));
    end
    if isfield(opts, 'case') && ~isfield(opts, 'report')
        fprintf(['仅运行了部分算例，未生成报告（避免覆盖完整报告）。' ...
                 '如需生成请显式指定 --report <路径>。\n']);
    elseif subset_configs && numel(config_paths) > 1 && ~isfield(opts, 'report')
        fprintf(['仅加载了部分配置文件，未生成报告（避免覆盖完整报告）。' ...
                 '如需生成请显式指定 --report <路径>。\n']);
    else
        if numel(config_paths) == 1
            meta = deep_merge(default_meta, configs{1}.meta);
        else
            meta = default_meta;
        end
        out = write_riemann_report(results, report_path, config_paths, meta);
        fprintf('报告已写入: %s\n', out);
    end
end
end

% ----------------------------------------------------------------------
function cfg = default_plot_cfg()
cfg = struct('dpi', 200, 'figsize', [4.2, 3.0], 'line_color', '#1f77b4');
end

function cs = lookup_case(cases, name)
cs = [];
for k = 1:numel(cases)
    if strcmp(cases{k}.name, name)
        cs = cases{k};
        return;
    end
end
error('run_cases:case', '未知算例 %s', name);
end

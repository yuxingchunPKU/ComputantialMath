function run_euler(varargin)
%RUN_EULER 一维 Euler 方程统一求解器的命令行入口
%
%   用法（在 CFD/Euler/matlab 目录下或加入路径后）：
%       run_euler
%       run_euler --list
%       run_euler --case fv_single_sod
%       run_euler --config configs/fv_single_sod.json
%       run_euler --no-plot --no-report
%
%   对应 Python 版本 CFD/Euler/run_euler.py。
add_euler_paths();
opts = parse_args(varargin);

if isfield(opts, 'config')
    config_args = opts.config;
else
    config_args = {};
end

try
    paths = discover_config_paths(config_args);
    configs = cell(1, numel(paths));
    for k = 1:numel(paths)
        configs{k} = load_config(paths{k});
    end
catch err
    fprintf(2, 'error: %s\n', err.message);
    error('run_euler:config', '%s', err.message);
end

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
    by_name = struct();
    for k = 1:numel(configs)
        by_name.(configs{k}.name) = configs{k};
    end
    missing = {};
    selected = cell(1, numel(wanted));
    for k = 1:numel(wanted)
        name = wanted{k};
        if ~isfield(by_name, name)
            missing{end + 1} = name; %#ok<AGROW>
        else
            selected{k} = by_name.(name);
        end
    end
    if ~isempty(missing)
        fprintf(2, 'error: unknown case(s): %s\n', strjoin(missing, ', '));
        error('run_euler:case', 'unknown case(s): %s', strjoin(missing, ', '));
    end
    configs = selected;
end

if isfield(opts, 'list') && opts.list
    fprintf('%-26s %-6s %-8s %-6s %s\n', 'case', 'scheme', 'medium', ...
            'order', 'config');
    for k = 1:numel(configs)
        cfg = configs{k};
        fprintf('%-26s %-6s %-8s %-6d %s\n', cfg.name, upper(cfg.scheme), ...
                cfg.medium, cfg.order, cfg.config_path);
    end
    return;
end

quiet = isfield(opts, 'quiet') && opts.quiet;
do_plot = ~(isfield(opts, 'no_plot') && opts.no_plot);
full_run = ~isfield(opts, 'case') && ~isfield(opts, 'config');

report_path = '';
if isfield(opts, 'report') && ischar(opts.report)
    report_path = opts.report;
elseif full_run && ~(isfield(opts, 'no_report') && opts.no_report)
    report_path = fullfile(euler_matlab_root(), 'report.md');
end

items = {};
failed = {};
for index = 1:numel(configs)
    cfg = configs{index};
    if ~quiet
        fprintf('[%d/%d] running %s ...\n', index, numel(configs), cfg.name);
    end
    try
        item = run_euler_case(cfg, do_plot);
        items{end + 1} = item; %#ok<AGROW>
        if ~quiet
            fprintf('    %d steps, dt = %.3e\n', item.result.steps, ...
                    item.result.dt);
        end
    catch err
        failed{end + 1} = struct('name', cfg.name, 'message', err.message); %#ok<AGROW>
        fprintf(2, '    FAILED: %s: %s\n', err.identifier, err.message);
    end
end

if ~isempty(report_path) && ~isempty(items)
    config_paths = cell(1, numel(items));
    for k = 1:numel(items)
        config_paths{k} = items{k}.cfg.config_path;
    end
    out = write_euler_report(items, report_path, config_paths);
    if ~quiet
        fprintf('report written to %s\n', out);
    end
end

if ~isempty(failed)
    fprintf(2, '%d case(s) failed:\n', numel(failed));
    for k = 1:numel(failed)
        fprintf(2, '  - %s: %s\n', failed{k}.name, failed{k}.message);
    end
    error('run_euler:failed', '%d case(s) failed', numel(failed));
end
end

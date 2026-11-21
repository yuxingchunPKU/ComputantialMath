function cfg = load_config(path)
%LOAD_CONFIG 读取并校验一个 JSON 算例配置（对应 Python 版本的 config.py）
%
%   MATLAB 没有内置 YAML，因此算例配置使用 JSON 文件；
%   结构与 Python 版本的 YAML 完全一致。
    path = char(path);
    if exist(path, 'file') ~= 2
        error('load_config:missing', 'configuration file not found: %s', path);
    end

    text = fileread(path);
    try
        raw = jsondecode(text);
    catch err
        error('load_config:json', '%s: invalid JSON (%s)', path, err.message);
    end
    if isempty(raw)
        raw = struct();
    end
    raw = require_struct(raw, path);

    cfg = deep_merge(defaults(), raw);
    if isfield(cfg, 'name') && ~isempty(cfg.name)
        cfg.name = char(cfg.name);
    else
        [~, stem, ~] = fileparts(path);
        cfg.name = stem;
    end
    cfg.config_path = path;

    cfg.scheme = lower(char(cfg.scheme));
    cfg.medium = lower(char(cfg.medium));
    if ~any(strcmp(cfg.scheme, {'fv', 'dg'}))
        error('load_config:scheme', '%s: scheme must be one of fv, dg', path);
    end
    if ~any(strcmp(cfg.medium, {'single', 'two'}))
        error('load_config:medium', '%s: medium must be one of single, two', path);
    end

    cfg.order = as_int(cfg.order, sprintf('%s: order', path));
    if strcmp(cfg.scheme, 'fv')
        if ~any(cfg.order == [1, 2])
            error('load_config:order', ...
                  '%s: finite-volume order must be 1 or 2', path);
        end
        if strcmp(cfg.medium, 'two') && cfg.order ~= 1
            error('load_config:order', ...
                  ['%s: two-medium finite volume currently supports order 1; ' ...
                   'use scheme dg for high-order two-medium data'], path);
        end
    else
        if ~any(cfg.order == [0, 1, 2])
            error('load_config:order', '%s: DG order must be 0, 1 or 2', path);
        end
    end

    cfg.flux = lower(char(cfg.flux));
    if ~any(strcmp(cfg.flux, {'llf', 'rusanov', 'hll', 'hllc'}))
        error('load_config:flux', ...
              '%s: flux must be one of llf, rusanov, hll, hllc', path);
    end

    cfg.time = require_struct(cfg.time, sprintf('%s: time', path));
    cfg.time.t_end = as_float(cfg.time.t_end, sprintf('%s: time.t_end', path));
    cfg.time.cfl = as_float(cfg.time.cfl, sprintf('%s: time.cfl', path));
    if cfg.time.t_end <= 0.0
        error('load_config:time', '%s: time.t_end must be positive', path);
    end
    if ~(cfg.time.cfl > 0.0 && cfg.time.cfl <= 1.0)
        error('load_config:time', '%s: time.cfl must lie in (0, 1]', path);
    end

    cfg.mesh = require_struct(cfg.mesh, sprintf('%s: mesh', path));
    cfg.mesh.x_min = as_float(cfg.mesh.x_min, sprintf('%s: mesh.x_min', path));
    cfg.mesh.x_max = as_float(cfg.mesh.x_max, sprintf('%s: mesh.x_max', path));
    cfg.mesh.nx = as_int(cfg.mesh.nx, sprintf('%s: mesh.nx', path));
    if cfg.mesh.x_max <= cfg.mesh.x_min
        error('load_config:mesh', '%s: mesh.x_max must exceed mesh.x_min', path);
    end
    if cfg.mesh.nx < 2
        error('load_config:mesh', '%s: mesh.nx must be at least 2', path);
    end
    cfg.mesh.boundary = lower(char(cfg.mesh.boundary));
    if ~any(strcmp(cfg.mesh.boundary, ...
                   {'transmissive', 'periodic', 'reflective'}))
        error('load_config:mesh', ...
              '%s: boundary must be transmissive, periodic or reflective', path);
    end
    if strcmp(cfg.medium, 'two') && strcmp(cfg.mesh.boundary, 'reflective')
        error('load_config:mesh', ...
              '%s: reflective boundaries are not supported for two-medium cases', ...
              path);
    end

    cfg.fluids = require_struct(cfg.fluids, sprintf('%s: fluids', path));
    if strcmp(cfg.medium, 'single')
        gamma = as_float(cfg.fluids.gamma, sprintf('%s: fluids.gamma', path));
        p_inf = as_float(cfg.fluids.p_inf, sprintf('%s: fluids.p_inf', path));
        if gamma <= 1.0
            error('load_config:fluids', '%s: fluids.gamma must exceed 1', path);
        end
        cfg.fluids = struct('gamma', gamma, 'p_inf', p_inf);
    else
        cfg.fluids = validate_two_fluids(cfg.fluids, path);
    end

    cfg.initial = require_struct(cfg.initial, sprintf('%s: initial', path));
    cfg.initial.type = lower(char(cfg.initial.type));
    if ~any(strcmp(cfg.initial.type, {'sod', 'smooth', 'constant'}))
        error('load_config:initial', ...
              '%s: initial.type must be sod, smooth or constant', path);
    end
    if ~isfield(cfg.initial, 'x_interface') || isempty(cfg.initial.x_interface)
        cfg.initial.x_interface = 0.5 * (cfg.mesh.x_min + cfg.mesh.x_max);
    end
    cfg.initial.x_interface = as_float(cfg.initial.x_interface, ...
                                       sprintf('%s: initial.x_interface', path));
    if ~(cfg.initial.x_interface > cfg.mesh.x_min && ...
         cfg.initial.x_interface < cfg.mesh.x_max)
        error('load_config:initial', ...
              '%s: initial.x_interface must lie inside the domain', path);
    end
    if any(strcmp(cfg.initial.type, {'sod', 'constant'}))
        cfg.initial.left = validate_state(cfg.initial.left, ...
                                          sprintf('%s: initial.left', path));
        cfg.initial.right = validate_state(cfg.initial.right, ...
                                           sprintf('%s: initial.right', path));
    else
        fields = {'rho0', 'rho_amp', 'k', 'u0', 'p0'};
        values = [1.0, 0.2, 1.0, 1.0, 1.0];
        for q = 1:numel(fields)
            f = fields{q};
            if ~isfield(cfg.initial, f) || isempty(cfg.initial.(f))
                cfg.initial.(f) = values(q);
            end
            cfg.initial.(f) = as_float(cfg.initial.(f), ...
                                       sprintf('%s: initial.%s', path, f));
        end
    end

    cfg.limiter = require_struct(cfg.limiter, sprintf('%s: limiter', path));
    if ~isfield(cfg.limiter, 'tvb') || isempty(cfg.limiter.tvb)
        cfg.limiter.tvb = 0.0;
    end
    cfg.limiter.tvb = as_float(cfg.limiter.tvb, sprintf('%s: limiter.tvb', path));

    cfg.output = require_struct(cfg.output, sprintf('%s: output', path));
    if ~isfield(cfg.output, 'dir') || isempty(cfg.output.dir)
        cfg.output.dir = fullfile('figures', cfg.name);
    end
    cfg.output.dir = char(cfg.output.dir);
    cfg.output.dpi = as_int(cfg.output.dpi, sprintf('%s: output.dpi', path));
    fsz = cfg.output.figsize;
    if ~isnumeric(fsz) || numel(fsz) ~= 2
        error('load_config:output', ...
              '%s: output.figsize must be [width, height]', path);
    end
    cfg.output.figsize = double(fsz(:).');
end

function cfg = defaults()
    cfg = struct();
    cfg.scheme = 'fv';
    cfg.medium = 'single';
    cfg.order = 2;
    cfg.flux = 'llf';
    cfg.description = '';
    cfg.time = struct('t_end', 0.2, 'cfl', 0.4);
    cfg.mesh = struct('x_min', 0.0, 'x_max', 1.0, 'nx', 200, ...
                      'boundary', 'transmissive');
    cfg.fluids = struct('gamma', 1.4, 'p_inf', 0.0);
    cfg.initial = struct('type', 'sod', 'x_interface', 0.5, ...
                         'left', struct('rho', 1.0, 'u', 0.0, 'p', 1.0), ...
                         'right', struct('rho', 0.125, 'u', 0.0, 'p', 0.1));
    cfg.limiter = struct('tvb', 0.0);
    cfg.output = struct('dpi', 160, 'figsize', [4.2, 3.0]);
end

function out = validate_two_fluids(fluids, path)
    if isfield(fluids, 'left')
        left = fluids.left;
    elseif isfield(fluids, 'phase0')
        left = fluids.phase0;
    else
        error('load_config:fluids', ...
              '%s: two-medium fluids need left and right entries', path);
    end
    if isfield(fluids, 'right')
        right = fluids.right;
    elseif isfield(fluids, 'phase1')
        right = fluids.phase1;
    else
        error('load_config:fluids', ...
              '%s: two-medium fluids need left and right entries', path);
    end
    out = struct();
    out.left = validate_fluid(left, sprintf('%s: fluids.left', path));
    out.right = validate_fluid(right, sprintf('%s: fluids.right', path));
end

function out = validate_fluid(value, name)
    value = require_struct(value, name);
    gamma = 1.4;
    p_inf = 0.0;
    if isfield(value, 'gamma') && ~isempty(value.gamma)
        gamma = as_float(value.gamma, sprintf('%s.gamma', name));
    end
    if isfield(value, 'p_inf') && ~isempty(value.p_inf)
        p_inf = as_float(value.p_inf, sprintf('%s.p_inf', name));
    end
    if gamma <= 1.0
        error('load_config:fluids', '%s.gamma must exceed 1', name);
    end
    out = struct('gamma', gamma, 'p_inf', p_inf);
end

function out = validate_state(state, name)
    state = require_struct(state, name);
    keys = {'rho', 'u', 'p'};
    out = struct();
    for k = 1:numel(keys)
        f = keys{k};
        if ~isfield(state, f) || isempty(state.(f))
            error('load_config:state', '%s is missing key: %s', name, f);
        end
        out.(f) = as_float(state.(f), sprintf('%s.%s', name, f));
    end
end

function out = require_struct(value, name)
    if ~isstruct(value)
        error('load_config:type', '%s must be a mapping', name);
    end
    out = value;
end

function v = as_float(value, name)
    if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value)
        error('load_config:value', '%s must be a number', name);
    end
    v = double(value);
end

function v = as_int(value, name)
    v = as_float(value, name);
    if v ~= round(v)
        error('load_config:value', '%s must be an integer', name);
    end
end

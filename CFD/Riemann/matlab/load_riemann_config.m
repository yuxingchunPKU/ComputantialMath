function config = load_riemann_config(path)
%LOAD_RIEMANN_CONFIG 读取一个 JSON 算例文件并合并 defaults
%
%   文件结构与 Python 版本的 YAML 相同：meta / defaults / cases。
%   返回 struct，字段 meta、defaults、cases（元胞数组）、path。
path = char(path);
if exist(path, 'file') ~= 2
    error('load_riemann_config:missing', '配置文件不存在: %s', path);
end
raw = jsondecode(fileread(path));
if ~isstruct(raw)
    error('load_riemann_config:type', '配置文件 %s 的顶层必须是映射', path);
end

config = struct();
config.path = path;
config.meta = struct();
if isfield(raw, 'meta') && isstruct(raw.meta)
    config.meta = raw.meta;
end
config.defaults = struct();
if isfield(raw, 'defaults') && isstruct(raw.defaults)
    config.defaults = raw.defaults;
end

if ~isfield(raw, 'cases') || isempty(raw.cases)
    error('load_riemann_config:cases', '配置文件 %s 中没有算例', path);
end
raw_cases = raw.cases;
if isstruct(raw_cases)
    n = numel(raw_cases);
    list = cell(1, n);
    for k = 1:n
        list{k} = raw_cases(k);
    end
elseif iscell(raw_cases)
    list = raw_cases;
else
    error('load_riemann_config:cases', ...
          '配置文件 %s 的 cases 必须是列表', path);
end

cases = cell(1, numel(list));
names = {};
for k = 1:numel(list)
    cs = deep_merge(config.defaults, list{k});
    cs = validate_case(cs, path);
    if any(strcmp(names, cs.name))
        error('load_riemann_config:dup', ...
              '算例名称重复: %s（%s）', cs.name, path);
    end
    names{end + 1} = cs.name; %#ok<AGROW>
    cs.source_file = path;
    cases{k} = cs;
end
config.cases = cases;
end

function cs = validate_case(cs, path)
if ~isfield(cs, 'name') || isempty(cs.name)
    error('load_riemann_config:name', '%s: 每个算例都必须提供 name', path);
end
cs.name = char(cs.name);

if ~isfield(cs, 'type') || isempty(cs.type)
    error('load_riemann_config:type', '算例 %s 缺少 type', cs.name);
end
cs.type = lower(char(cs.type));
if ~any(strcmp(cs.type, {'inert', 'detonation', 'deflagration'}))
    error('load_riemann_config:type', ...
          '算例 %s 的 type 必须是 inert、detonation 或 deflagration', cs.name);
end

for key = {'left', 'right', 'domain'}
    if ~isfield(cs, key{1}) || isempty(cs.(key{1}))
        error('load_riemann_config:field', '算例 %s 缺少 %s', ...
              cs.name, key{1});
    end
end
for key = {'x_L', 'x_R', 'x_c', 't_end', 'N'}
    if ~isfield(cs.domain, key{1}) || isempty(cs.domain.(key{1}))
        error('load_riemann_config:domain', ...
              '算例 %s 的 domain 缺少 %s', cs.name, key{1});
    end
end
cs.domain.N = round(double(cs.domain.N));

if strcmp(cs.type, 'deflagration')
    if ~isfield(cs, 'deflagration')
        error('load_riemann_config:deflagration', ...
              '爆燃算例 %s 缺少 deflagration 参数 (K, Q)', cs.name);
    end
    if ~isfield(cs.deflagration, 'K')
        error('load_riemann_config:deflagration', ...
              '爆燃算例 %s 缺少 K', cs.name);
    end
    if ~isfield(cs.deflagration, 'Q') || isempty(cs.deflagration.Q)
        cs.deflagration.Q = 2;
    end
    cs.deflagration.Q = round(double(cs.deflagration.Q));
end

if isfield(cs, 'numerical')
    if isfield(cs.numerical, 'method') && ~isempty(cs.numerical.method)
        cs.numerical.method = lower(char(cs.numerical.method));
    else
        cs.numerical.method = 'newton';
    end
    if ~isfield(cs.numerical, 'tol') || isempty(cs.numerical.tol)
        cs.numerical.tol = 1.0e-12;
    end
    if ~isfield(cs.numerical, 'maxit') || isempty(cs.numerical.maxit)
        cs.numerical.maxit = 1000;
    end
    cs.numerical.maxit = round(double(cs.numerical.maxit));
else
    cs.numerical = struct('method', 'newton', 'tol', 1.0e-12, ...
                          'maxit', 1000);
end

if ~isfield(cs, 'plot')
    cs.plot = struct();
end
if ~isfield(cs, 'description')
    cs.description = '';
end
end

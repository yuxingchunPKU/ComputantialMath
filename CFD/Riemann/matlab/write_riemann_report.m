function path = write_riemann_report(results, report_path, config_paths, meta)
%WRITE_RIEMANN_REPORT 生成并写出 Markdown 报告（UTF-8）
if nargin < 3
    config_paths = {};
end
if nargin < 4
    meta = struct();
end
report_path = char(report_path);
parent = fileparts(absolute_path(report_path));
if ~isempty(parent) && exist(parent, 'dir') ~= 7
    mkdir(parent);
end

text = build_riemann_report(results, report_path, config_paths, meta);

fid = -1;
try
    fid = fopen(report_path, 'w', 'n', 'UTF-8');
catch
    fid = -1;
end
if fid < 0
    fid = fopen(report_path, 'w');
end
if fid < 0
    error('write_riemann_report:open', '无法写入报告: %s', report_path);
end
fprintf(fid, '%s', text);
fclose(fid);
path = report_path;
end

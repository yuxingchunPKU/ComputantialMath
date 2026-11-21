function path = write_euler_report(items, report_path, config_paths)
%WRITE_EULER_REPORT 生成并写出 Markdown 报告（UTF-8）
if nargin < 2 || isempty(report_path)
    report_path = fullfile(euler_matlab_root(), 'report.md');
end
if nargin < 3
    config_paths = {};
end
report_path = char(report_path);

parent = fileparts(absolute_path(report_path));
if ~isempty(parent) && exist(parent, 'dir') ~= 7
    mkdir(parent);
end

text = build_euler_report(items, report_path, config_paths);

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
    error('write_euler_report:open', 'cannot write report: %s', report_path);
end
fprintf(fid, '%s', text);
fclose(fid);
path = report_path;
end

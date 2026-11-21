function out = euler_output_dir(cfg)
%EULER_OUTPUT_DIR 返回算例图像输出目录（相对路径以 matlab/ 为基准）
%
%   与 Python 版本不同，输出目录以 CFD/Euler/matlab 为基准，
%   避免覆盖 Python 版本已生成的 figures/ 与 report.md。
d = char(cfg.output.dir);
is_absolute = (~isempty(d) && d(1) == '/') || ...
              (numel(d) > 1 && d(2) == ':');
if ~is_absolute
    d = fullfile(euler_matlab_root(), d);
end
out = d;
end

%MAIN 兼容入口：运行经典 Sod 激波管算例
%
%   对应 Python 版本 code/main.py。
%   算例参数位于 cases/ 目录下的 JSON 文件中，本脚本仅调用统一的算例驱动程序。
%   若需要生成完整报告，请运行：
%
%       run_cases
%
%   若需要运行其它算例，请使用：
%
%       run_cases --case <算例名>
add_riemann_paths();
run_cases('--case', 'inert_sod', '--no-report');

%MAIN4REACTCJDETONATION 兼容入口：运行 CJ 爆轰算例
%
%   对应 Python 版本 code/main4ReactCJDetonation.py。
%   参数位于 cases/detonation.json。完整测试报告请运行：
%
%       run_cases
add_riemann_paths();
run_cases('--case', 'detonation_cj', '--no-report');

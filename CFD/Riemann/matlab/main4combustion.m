%MAIN4COMBUSTION 兼容入口：运行爆轰波算例（强爆轰与 CJ 爆轰）
%
%   对应 Python 版本 code/main4combustion.py。
%   参数位于 cases/detonation.json。完整测试报告请运行：
%
%       run_cases
add_riemann_paths();
run_cases('--case', 'detonation_strong,detonation_cj', '--no-report');

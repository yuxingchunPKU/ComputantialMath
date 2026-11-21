%MAIN4DEFLAGRATIONS 兼容入口：运行爆燃波算例（u_L = 0）
%
%   对应 Python 版本 code/main4deflagrations.py。
%   参数位于 cases/deflagration.json。完整测试报告请运行：
%
%       run_cases
add_riemann_paths();
run_cases('--case', 'deflagration_uL0', '--no-report');

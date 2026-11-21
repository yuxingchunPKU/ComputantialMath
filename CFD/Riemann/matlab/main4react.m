%MAIN4REACT 兼容入口：运行反应流算例（强爆轰与爆燃）
%
%   对应 Python 版本 code/main4react.py。
%   参数位于 cases/detonation.json 与 cases/deflagration.json。
%   完整测试报告请运行：
%
%       run_cases
add_riemann_paths();
run_cases('--case', 'detonation_strong,deflagration', '--no-report');

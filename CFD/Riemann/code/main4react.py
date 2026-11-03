#!/usr/bin/env python3
'''
兼容入口：运行反应流算例（强爆轰与爆燃）。

参数位于 cases/detonation.yaml 与 cases/deflagration.yaml。完整测试报告请运行：

    python run_cases.py
'''
from run_cases import main


if __name__ == "__main__":
    main(["--case", "detonation_strong,deflagration", "--no-report"])

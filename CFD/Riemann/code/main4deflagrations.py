#!/usr/bin/env python3
'''
兼容入口：运行爆燃波算例（u_L = 0）。

参数位于 cases/deflagration.yaml。完整测试报告请运行：

    python run_cases.py
'''
from run_cases import main


if __name__ == "__main__":
    main(["--case", "deflagration_uL0", "--no-report"])

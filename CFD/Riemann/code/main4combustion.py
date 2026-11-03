#!/usr/bin/env python3
'''
兼容入口：运行爆轰波算例（强爆轰与 CJ 爆轰）。

参数位于 cases/detonation.yaml。完整测试报告请运行：

    python run_cases.py
'''
from run_cases import main


if __name__ == "__main__":
    main(["--case", "detonation_strong,detonation_cj", "--no-report"])

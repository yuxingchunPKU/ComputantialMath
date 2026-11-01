#!/usr/bin/env python3
'''
兼容入口：运行 CJ 爆轰算例。

参数位于 cases.yaml。完整测试报告请运行：

    python run_cases.py
'''
from run_cases import main


if __name__ == "__main__":
    main(["--case", "detonation_cj", "--no-report"])

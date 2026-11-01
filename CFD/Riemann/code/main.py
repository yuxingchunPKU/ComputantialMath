#!/usr/bin/env python3
'''
兼容入口：运行经典 Sod 激波管算例。

算例参数已移至 cases.yaml，本脚本仅调用统一的算例驱动程序。
若需要生成完整报告，请运行：

    python run_cases.py

若需要运行其它算例，请使用：

    python run_cases.py --case <算例名>
'''
from run_cases import main


if __name__ == "__main__":
    main(["--case", "inert_sod", "--no-report"])

# 一维多介质黎曼问题求解器的精确解

控制方程是Euler方程组，气体状态方程是刚性气体状态方程
$$
p = (\gamma-1)\rho e - \gamma p_{\infty}
$$
求解的数值方法是牛顿方法或 Brent 方法。

数值方法的实现流程

Cutforth, M. C. (2019). *Sharp interface schemes for multi-material computational fluid dynamics*. https://doi.org/10.17863/CAM.44907

## 运行测试

全部算例参数写在 [`code/cases.yaml`](./code/cases.yaml) 中，程序读取 YAML 文件后
自动计算并输出图像与 Markdown 报告：

```bash
cd code
pip install -r requirements.txt
python run_cases.py              # 运行全部算例并生成 ../report.md
python run_cases.py --list       # 列出全部算例
python run_cases.py --case inert_sod
```

完整的测试结果（密度、速度、压强分布以及波系参数）见
[`report.md`](./report.md)。

## 例子

初始值
$$
(\rho, u ,p, \gamma,p_\infty) =
\begin{cases} 
(1,0,1,1.4,0) \\
(0.125,0,0.1,1.667,0)
\end{cases}
$$
计算的终止时间是$t=0.2$.

## 完全解

| 密度                      | 速度                  | 压强                  |
| ------------------------- | --------------------- | --------------------- |
| ![rho](./fig/inert_sod/rho.png) | ![u](./fig/inert_sod/u.png) | ![p](./fig/inert_sod/p.png) |


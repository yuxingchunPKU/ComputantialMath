# 一维黎曼问题精确解求解器（MATLAB / Octave 版本）

本目录是 [`CFD/Riemann/code`](../code/) Python 程序的 MATLAB 移植版，求解
刚性气体与反应流一维 Euler 方程的 Riemann 问题精确解。控制方程为 Euler
方程组，状态方程为刚性气体状态方程

$$
p = (\gamma-1)\rho e - \gamma p_\infty ,
$$

数值方法为牛顿迭代（PVRS 初值）或 Brent 方法，用于求解匹配方程
$f_L(p)+f_R(p)+u_R-u_L=0$；反应流算例在匹配方程中引入爆轰/爆燃的
Rayleigh-Hugoniot 关系。

MATLAB 没有内置 YAML 解析，因此算例配置改为 JSON（用 `jsondecode` 读取）。

## 快速开始

在 MATLAB 中（或 Octave 中）执行：

```matlab
cd CFD/Riemann/matlab
run_cases                                   % 运行全部算例并生成 report.md
run_cases --config cases/inert.json         % 只运行非反应算例
run_cases --config cases/detonation.json    % 只运行爆轰算例
run_cases --config cases/deflagration.json  % 只运行爆燃算例
run_cases --list                            % 列出全部算例
run_cases --case inert_sod
run_cases --case inert_sod,detonation_cj    % 逗号分隔或多次 --case
selftest_riemann                            % 快速自检
```

也可以在任意目录下调用，只要先把本目录加入路径：

```matlab
addpath('/path/to/CFD/Riemann/matlab');
run_cases
```

### 兼容入口

与 Python 版本同名，便于对照：

| 脚本 | 运行的算例 |
| --- | --- |
| `main.m` | `inert_sod` |
| `main4combustion.m` | `detonation_strong`、`detonation_cj` |
| `main4deflagrations.m` | `deflagration_uL0` |
| `main4ReactCJDetonation.m` | `detonation_cj` |
| `main4react.m` | `detonation_strong`、`deflagration` |

这些入口默认加 `--no-report`，不覆盖完整报告。

### Octave 说明

本目录在 GNU Octave 10.3 下测试通过。若使用 conda 安装的 Octave，
需要显式指定 `OCTAVE_HOME`（该构建内置的路径是无效的占位值）：

```bash
OCTAVE_HOME=/path/to/miniconda3/envs/octave \
  /path/to/miniconda3/envs/octave/bin/octave -q --eval "run_cases"
```

## 目录结构

```
CFD/Riemann/matlab/
├── run_cases.m                 % 命令行入口
├── run_riemann_case.m          % 单算例调度、检查与出图
├── cases/*.json                % 算例配置
│   ├── inert.json              % 非反应刚性气体
│   ├── detonation.json         % 爆轰
│   └── deflagration.json       % 爆燃
├── Fluid.m, FluidReact.m       % 状态与波关系
├── ExactRiemannSolver.m        % 非反应精确黎曼求解（Newton + Brent）
├── ExactRiemannSolverDetonation.m
├── ExactRiemannSolverDeflagration.m
├── RiemannCompleteSolve.m      % 完全解采样（非反应）
├── RiemannCompleteSolDet.m     % 完全解采样（爆轰）
├── RiemannCompleteSolDef.m     % 完全解采样（爆燃）
├── nonreactive_left_wave.m     % 稀疏波/激波分支
├── nonreactive_right_wave.m
├── compute_inert.m             % 按类型组装算例
├── compute_detonation.m
├── compute_deflagration.m
├── plot_riemann_case.m         % rho/u/p 与三联图
├── build_riemann_report.m      % 报告组装
├── selftest_riemann.m          % 自检脚本
├── main*.m                     % 兼容入口
├── fig/<case>/                 % 每个算例的图像
└── report.md                   % 自动生成的测试报告
```

## JSON 配置

```json
{
  "meta": {
    "title": "爆轰波黎曼问题精确解测试报告",
    "description": "反应流中爆轰波 Riemann 问题的精确解测试。"
  },
  "defaults": {
    "numerical": {"method": "newton", "tol": 1.0e-12, "maxit": 1000},
    "plot": {"dpi": 200, "figsize": [4.2, 3.0], "line_color": "#1f77b4"}
  },
  "cases": [
    {
      "name": "detonation_strong",
      "description": "强爆轰算例。",
      "type": "detonation",
      "left": {"rho": 1.57861, "u": 2799.82, "p": 7.70752e6, "gamma": 1.27, "q": 0.0},
      "right": {"rho": 0.601, "u": 0.0, "p": 1.0e5, "gamma": 1.27, "q": 1.3444444444444444e7},
      "domain": {"x_L": 0.0, "x_R": 8.0, "x_c": 4.0, "t_end": 5.0e-4, "N": 1200}
    }
  ]
}
```

`type` 取 `inert`（非反应）、`detonation`（爆轰）或 `deflagration`（爆燃）；
`numerical.method` 取 `newton` 或 `brent`。`defaults` 中的参数会与每个
`cases` 条目做深层合并，单算例可覆盖。

## 算例

| 算例 | 类型 | 内容 |
| --- | --- | --- |
| `inert_sod` | 非反应 | 经典 Sod 激波管，左右 `gamma` 不同 |
| `inert_stiffened_gas` | 非反应 | Sod 的刚性气体推广（非零 `p_inf`） |
| `detonation_strong` | 爆轰 | 强爆轰波 |
| `detonation_cj` | 爆轰 | CJ 爆轰波 + 稀疏波 |
| `deflagration` | 爆燃 | 爆燃波（`u_L ≠ 0` 的一般情形） |
| `deflagration_uL0` | 爆燃 | 爆燃波（`u_L = 0`） |

新增算例只需在 `cases/` 下对应的 JSON 文件中追加条目，不必改代码。

## 输出

- `fig/<case>/rho.png`、`u.png`、`p.png`、`overview.png`；
  竖直虚线标出各道波的位置；
- `report.md`：算例汇总表、求解方法、每个算例的初值/中间状态/波系表、
  结果图像与复现命令。

图像与报告默认写入本目录（`matlab/fig/`、`matlab/report.md`），不会覆盖
Python 版本已生成的 `CFD/Riemann/fig/` 与 `CFD/Riemann/report.md`。
只运行单个配置文件时，报告默认写到 `report_<文件名>.md`；只跑部分算例时
默认不生成报告（避免覆盖完整报告），可用 `--report <path>` 显式指定。

## 与 Python 版本的一致性

全部 6 个算例的中间状态与 Python 版本一致：

| 算例 | p* | u* |
| --- | ---: | ---: |
| `inert_sod` | 0.3143966584 | 0.9013775087 |
| `inert_stiffened_gas` | 0.2938748334 | 0.6295798596 |
| `detonation_strong` | 7.70752512e6 | 2.79981870e3 |
| `detonation_cj` | 4.04266012e6 | 1.58604994e3 |
| `deflagration` | 9.45695111e4 | -1.81018081e2 |
| `deflagration_uL0` | 1.15181901e5 | -1.37909786e2 |

`selftest_riemann` 检查经典 Sod 精确解（`p* = 0.30313017805`，
`u* = 0.92745262005`）、Newton/Brent 一致性、匹配方程残差、全部算例的
`p*`/`u*`、解的正性与波速有限性。

## 备注

- 与 Python 版本的主要差异：配置为 JSON；MATLAB 数组从 1 开始计数；
  输出目录以 `matlab/` 为基准。
- 原始 Python 脚本（`code/*.py`）未作改动，保留作为参考实现。
- 参考实现：Cutforth, M. C. (2019). *Sharp interface schemes for
  multi-material computational fluid dynamics*.
  https://doi.org/10.17863/CAM.44907

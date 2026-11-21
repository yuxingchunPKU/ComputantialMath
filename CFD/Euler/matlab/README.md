# 一维 Euler 方程统一求解器（MATLAB / Octave 版本）

本目录是 [`CFD/Euler`](../) Python 程序（`euler_unified/` + `run_euler.py`）的
MATLAB 移植版，功能与数值方法一一对应：

- 有限体积（FV）求解器；
- 间断有限元（DG）求解器；
- 单介质与双介质（单个移动界面）两种物理模型；
- 格式参数全部写在 `configs/*.json` 中，由 `run_euler.m` 统一读取运行并生成
  Markdown 报告。

MATLAB 没有内置 YAML 解析，因此配置格式改为 JSON（用 `jsondecode` 读取）。
函数名、类名与选项含义与 Python 版本保持一致。

## 快速开始

在 MATLAB 中（或 Octave 中）执行：

```matlab
cd CFD/Euler/matlab
run_euler                    % 运行 configs/ 下全部算例，生成 report.md
run_euler --list             % 列出算例
run_euler --case fv_single_sod
run_euler --config configs/dg_two_sod.json
run_euler --no-plot          % 只计算不画图
run_euler --no-report        % 不写报告
selftest_euler               % 快速自检
```

也可以在任意目录下调用，只要先把本目录加入路径：

```matlab
addpath('/path/to/CFD/Euler/matlab');
run_euler
```

### Octave 说明

本目录在 GNU Octave 10.3 下测试通过。若使用 conda 安装的 Octave，
脚本需要显式指定 `OCTAVE_HOME`（该构建内置的路径是无效的占位值）：

```bash
OCTAVE_HOME=/path/to/miniconda3/envs/octave \
  /path/to/miniconda3/envs/octave/bin/octave -q --eval "run_euler"
```

## 目录结构

```
CFD/Euler/matlab/
├── run_euler.m             % 命令行入口
├── run_euler_case.m        % 单个算例调度
├── configs/*.json          % 算例配置
├── FVSolver.m              % 单介质 FV 求解器
├── DGSolver.m              % 单介质 DG 求解器（P0/P1/P2）
├── TwoMediumSolver.m       % 双介质切割单元 DG / FV 求解器
├── StiffenedGas.m          % 刚性气体状态方程
├── numerical_flux.m        % LLF / HLL / HLLC 通量
├── minmod2.m, minmod_tvb.m % 限制器
├── gauss_legendre.m        % Gauss-Legendre 求积
├── exact_riemann.m         % 精确黎曼解采样
├── plot_euler_case.m       % 图像输出
├── build_euler_report.m    % 报告组装
├── selftest_euler.m        % 自检脚本
├── figures/<case>/         % 每个算例的 rho/u/p/overview 图
└── report.md               % 自动生成的测试报告
```

## JSON 配置

```json
{
  "name": "dg_single_sod",
  "description": "单介质 DG P2 格式求解 Sod 激波管问题。",
  "scheme": "dg",
  "medium": "single",
  "order": 2,
  "flux": "llf",
  "time": {"t_end": 0.15, "cfl": 0.1},
  "mesh": {"x_min": 0.0, "x_max": 1.0, "nx": 200, "boundary": "transmissive"},
  "fluids": {"gamma": 1.4, "p_inf": 0.0},
  "initial": {
    "type": "sod",
    "x_interface": 0.5,
    "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
    "right": {"rho": 0.125, "u": 0.0, "p": 0.1}
  },
  "limiter": {"tvb": 0.0},
  "output": {"dir": "figures/dg_single_sod", "dpi": 160, "figsize": [4.2, 3.0]}
}
```

双介质算例把 `fluids` 换成左右两组参数：

```json
"fluids": {
  "left":  {"gamma": 1.4,   "p_inf": 0.0},
  "right": {"gamma": 1.667, "p_inf": 0.0}
}
```

`initial.type` 还可取 `smooth`（`rho0`、`rho_amp`、`k`、`u0`、`p0`）或 `constant`。
未写出的键使用默认值，未知的键会被忽略；非法取值（如 `scheme: fv` 配
`order: 3`、双介质配 `reflective`）会在读取时直接报错。

## 数值方法

| 组合 | 实现 |
| --- | --- |
| `fv` + `single` | 一阶 LLF/HLL/HLLC；二阶 MUSCL + minmod（守恒量）；SSP-RK2 |
| `dg` + `single` | P0/P1/P2 Legendre 基；P1 minmod、P2 特征变量 TVB 限制器；SSP-RK2（P0/P1）、SSP-RK3（P2） |
| `dg` + `two` | 界面单元用两相切割单元；界面速度与界面通量由刚性气体精确黎曼解给出；相内数据为 P0/P1/P2 多项式 |
| `fv` + `two` | 即 `dg` + `two` 的 P0 特例（`order: 1`） |

刚性气体状态方程 `p = (gamma-1)(E - rho u^2/2) - gamma p_inf`，双介质左右两侧
可取不同的 `gamma`、`p_inf`。

## 算例

| 配置 | 内容 |
| --- | --- |
| `fv_single_sod` | 单介质 FV 二阶 Sod |
| `fv_single_smooth` | 单介质 FV 二阶光滑波周期输运 |
| `dg_single_sod` | 单介质 DG P2 Sod |
| `dg_single_smooth` | 单介质 DG P2 光滑波周期输运 |
| `fv_two_sod` | 双介质切割单元 P0 Sod |
| `dg_two_sod` | 双介质 DG P2 Sod |
| `dg_two_smooth` | 双介质 DG P2 光滑波（界面只作介质标记） |
| `dg_two_gamma_sod` | 左右 `gamma` 不同的双介质 DG P2 Sod |

新增算例只需在 `configs/` 增加 JSON 文件，不必改代码。想临时尝试其他通量，
把 `flux` 改成 `hll` 或 `hllc` 即可。

## 输出

- `figures/<case>/rho.png`、`u.png`、`p.png`、`overview.png`；
- `report.md`：算例汇总表、每个算例的参数/流体/初值表、L1/L∞ 误差、
  结果图像与复现命令。

图像与报告默认写入本目录（`matlab/figures/`、`matlab/report.md`），
不会覆盖 Python 版本已生成的 `CFD/Euler/figures/` 与 `CFD/Euler/report.md`。
全量运行（不带 `--case` / `--config`）时默认写 `report.md`；只跑部分算例时
默认不写报告，可用 `--report <path>` 显式指定。

## 与 Python 版本的一致性

移植后与 Python 版本逐算例对比（相同配置、相同机器），步数、时间步长、
界面位置与误差指标均一致：

| 算例 | 对比结果 |
| --- | --- |
| `fv_single_sod` | 327 步；L1(rho) ≈ 3.06e-3 |
| `dg_single_sod`（P2） | 650 步；L1(rho) ≈ 3.04e-3 |
| `fv_two_sod` | 325 步，dt = 5.711084e-5，界面 0.6392748，L1 1.218975e-2 |
| `dg_two_sod`（P1） | 317 步，dt = 1.879094e-4，界面 0.6391777，L1 8.456384e-3 |
| `dg_two_sod`（P2） | 317 步，dt = 2.243533e-4，界面 0.6393553，L1 7.211567e-3 |

`selftest_euler` 对求积、限制器、状态方程、精确解与上述典型算例做快速检查。

## 备注

- 与 Python 版本的主要差异：配置为 JSON；MATLAB 数组从 1 开始计数；双介质
  状态用 `1×2` 元胞保存左右相数据；输出目录以 `matlab/` 为基准。
- 精确黎曼解复用本目录的 `exact_riemann.m`（底层为 `Fluid` /
  `ExactRiemannSolver` / `RiemannCompleteSolve`，与 `CFD/Riemann/matlab`
  中的实现相同）。
- 原始 Python 脚本（`Euler.py`、`EulerMUSCL.py`、`DG/` 等）未作改动，
  保留作为参考实现。

# 一维 Euler 方程统一求解器

把 `CFD/Euler` 下原来的两套程序统一为一个包 `euler_unified`：

- 有限体积（FV）求解器；
- 间断有限元（DG）求解器；
- 单介质与双介质（单个移动界面）两种物理模型。

所有算例参数（流体参数、数值方法、网格、时间、初边值条件、输出）都写在
`configs/*.yaml` 中，由 `run_euler.py` 统一读取运行并生成 Markdown 报告。

## 快速开始

```bash
cd CFD/Euler
source /home/yxc2025/code/py311/venv311/bin/activate   # 或使用该 venv 的 python

python run_euler.py                 # 运行 configs/ 下全部算例，生成 report.md
python run_euler.py --list          # 列出算例
python run_euler.py --case dg_single_sod
python run_euler.py --config configs/fv_single_sod.yaml
python run_euler.py --no-plot       # 只计算不画图
python run_euler.py --no-report     # 不写报告
```

依赖见 `requirements.txt`（numpy、scipy、matplotlib、PyYAML）。

## 目录结构

```
CFD/Euler/
├── run_euler.py            # 命令行入口
├── configs/*.yaml          # 算例配置
├── euler_unified/
│   ├── config.py           # YAML 读取、默认值与校验
│   ├── eos.py              # 刚性气体状态方程
│   ├── numerics.py         # minmod、LLF/HLL/HLLC 通量、精确黎曼采样
│   ├── initial.py          # 初始条件与精确解
│   ├── single.py           # 单介质 FV / DG 求解器
│   ├── two_medium.py       # 双介质切割单元 DG / FV 求解器
│   ├── plotting.py         # 图像输出
│   └── run.py              # 算例调度与 Markdown 报告
├── figures/<case>/         # 每个算例的 rho/u/p/overview 图
└── report.md               # 自动生成的测试报告
```

## YAML 配置

```yaml
name: dg_single_sod          # 算例名，默认取文件名
description: 说明文字        # 可选，写入报告
scheme: dg                   # fv | dg
medium: single               # single | two
order: 2                     # FV: 1/2；DG: 0/1/2；双介质 FV 仅 1
flux: llf                    # llf/rusanov | hll | hllc

time:
  t_end: 0.15
  cfl: 0.1

mesh:
  x_min: 0.0
  x_max: 1.0
  nx: 200
  boundary: transmissive     # transmissive | periodic | reflective
                             # 双介质不支持 reflective

# 单介质
fluids:
  gamma: 1.4
  p_inf: 0.0

# 双介质
# fluids:
#   left:  {gamma: 1.4,   p_inf: 0.0}
#   right: {gamma: 1.667, p_inf: 0.0}

initial:
  type: sod                  # sod | smooth | constant
  x_interface: 0.5           # 间断位置（双介质时即初始界面位置）
  left:  {rho: 1.0,   u: 0.0, p: 1.0}
  right: {rho: 0.125, u: 0.0, p: 0.1}

# type: smooth 时改用：
#   rho0: 1.0    # rho = rho0 + rho_amp*sin(k*pi*(x - u0*t))
#   rho_amp: 0.2
#   k: 1
#   u0: 1.0
#   p0: 1.0

limiter:
  tvb: 0.0                   # DG 限制器的 TVB 参数

output:
  dir: figures/dg_single_sod # 相对 CFD/Euler
  dpi: 160
  figsize: [4.2, 3.0]
```

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

新增算例只需在 `configs/` 增加 YAML，不必改代码。想临时尝试其他通量，把
`flux` 改成 `hll` 或 `hllc` 即可。

## 输出

- `figures/<case>/rho.png`、`u.png`、`p.png`、`overview.png`；
- `report.md`：算例汇总表、每个算例的参数/流体/初值表、L1/L∞ 误差、
  结果图像与复现命令。

全量运行（不带 `--case` / `--config`）时默认覆盖写 `report.md`；只跑部分
算例时默认不写报告，可用 `--report <path>` 显式指定。

## 与参考程序的对应关系

- `DG/DGEuler1D_order.py` 的 P2 光滑波算例（`Nx=160, t=2`）在本程序中对应
  `dg_single_smooth`，终了时刻的 DG 系数与参考实现逐点最大差 `5.5e-12`。
- `DG/DGEuler2medium*.py` 的双介质三单元模板、界面黎曼推进与 RK 阶段索引
  约定沿用参考实现，界面位置与参考结果一致（P2 Sod 约 `0.6394`）。

## 备注

- `Euler.py`、`EulerMUSCL.py`、`DG/` 下的原始脚本保留未改动，作为参考实现。
- 精确黎曼解复用 `Riemann/code` 中的刚性气体求解器；单介质 Sod 与双介质
  算例的误差都以它为参照。

"""Case dispatcher and Markdown report generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .config import DEFAULT_REPORT
from .plotting import plot_case
from .single import solve_single
from .two_medium import solve_two_medium


def run_case(cfg: dict, make_plots: bool = True) -> dict:
    if cfg["medium"] == "single":
        result = solve_single(cfg)
    elif cfg["medium"] == "two":
        result = solve_two_medium(cfg)
    else:  # pragma: no cover - validated by config
        raise ValueError(cfg["medium"])
    figures = plot_case(cfg, result) if make_plots else {}
    return {"cfg": cfg, "result": result, "figures": figures}


def _fmt(value):
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if value == 0.0:
            return "0"
        if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
            return f"{value:.3e}"
        return f"{value:.6g}"
    return str(value)


def _error_metrics(result):
    metrics = {}
    for key in ("rho", "u", "p"):
        exact = result.get(f"{key}_exact")
        if exact is None:
            metrics[key] = (None, None)
            continue
        diff = np.asarray(result[key]) - np.asarray(exact)
        metrics[key] = (float(np.mean(np.abs(diff))), float(np.max(np.abs(diff))))
    return metrics


def _relative_image(report_path: Path, image_path: Path) -> str:
    try:
        rel = image_path.resolve().relative_to(report_path.resolve().parent)
    except ValueError:
        import os
        rel = Path(os.path.relpath(image_path.resolve(), report_path.resolve().parent))
    return rel.as_posix()


def _state_table(cfg):
    initial = cfg["initial"]
    if initial["type"] == "smooth":
        return (
            "| 参数 | 值 |\n| --- | ---: |\n"
            f"| 密度 | $\\rho_0 + \\rho_a\\sin(k\\pi x)$，"
            f"$\\rho_0={_fmt(initial['rho0'])}$，"
            f"$\\rho_a={_fmt(initial['rho_amp'])}$，$k={_fmt(initial['k'])}$ |\n"
            f"| 速度 | {_fmt(initial['u0'])} |\n"
            f"| 压强 | {_fmt(initial['p0'])} |\n"
        )
    left = initial["left"]
    right = initial["right"]
    return (
        "| 参数 | 左状态 | 右状态 |\n| --- | ---: | ---: |\n"
        f"| rho | {_fmt(left['rho'])} | {_fmt(right['rho'])} |\n"
        f"| u | {_fmt(left['u'])} | {_fmt(right['u'])} |\n"
        f"| p | {_fmt(left['p'])} | {_fmt(right['p'])} |\n"
    )


def _fluid_table(cfg):
    if cfg["medium"] == "single":
        f = cfg["fluids"]
        return (
            "| 参数 | 值 |\n| --- | ---: |\n"
            f"| gamma | {_fmt(f['gamma'])} |\n"
            f"| p_inf | {_fmt(f['p_inf'])} |\n"
        )
    l = cfg["fluids"]["left"]
    r = cfg["fluids"]["right"]
    return (
        "| 参数 | 左介质 | 右介质 |\n| --- | ---: | ---: |\n"
        f"| gamma | {_fmt(l['gamma'])} | {_fmt(r['gamma'])} |\n"
        f"| p_inf | {_fmt(l['p_inf'])} | {_fmt(r['p_inf'])} |\n"
    )


def _summary_row(index, item):
    cfg = item["cfg"]
    res = item["result"]
    two = cfg["medium"] == "two"
    interface = _fmt(res.get("interface")) if two else "—"
    err = _error_metrics(res)
    rho_l1 = _fmt(err["rho"][0])
    rho_linf = _fmt(err["rho"][1])
    return (
        f"| {index} | {cfg['name']} | {cfg['scheme'].upper()} | "
        f"{'多介质' if two else '单介质'} | {cfg['order']} | "
        f"{cfg['flux'].upper()} | {cfg['mesh']['nx']} | "
        f"{_fmt(cfg['time']['t_end'])} | {res['steps']} | {interface} | "
        f"{rho_l1} | {rho_linf} |"
    )


def _case_section(index, item, report_path):
    cfg = item["cfg"]
    res = item["result"]
    figures = item["figures"]
    two = cfg["medium"] == "two"
    err = _error_metrics(res)

    lines = [f"## {index}. {cfg['name']}", ""]
    if cfg.get("description"):
        lines += [cfg["description"], ""]

    lines += ["**计算参数**", "", "| 参数 | 值 |", "| --- | ---: |",
              f"| 格式 | {cfg['scheme'].upper()}，{cfg['order']} 阶 |",
              f"| 介质 | {'双介质（单个移动界面）' if two else '单介质'} |",
              f"| 数值通量 | {cfg['flux'].upper()} |",
              f"| 计算区域 | $x\\in[{_fmt(cfg['mesh']['x_min'])},"
              f"{_fmt(cfg['mesh']['x_max'])}]$，{cfg['mesh']['nx']} 个单元 |",
              f"| 边界条件 | {cfg['mesh']['boundary']} |",
              f"| 终止时间 / CFL | {_fmt(cfg['time']['t_end'])} / "
              f"{_fmt(cfg['time']['cfl'])} |",
              f"| 时间步数 | {res['steps']} |"]
    if two:
        lines.append(f"| 界面终了位置 | {_fmt(res.get('interface'))} |")
    lines += ["", "**流体参数**", "", _fluid_table(cfg),
              "**初始条件**", "", _state_table(cfg),
              "**误差（相对精确解）**", "",
              "| 量 | L1 | L∞ |", "| --- | ---: | ---: |"]
    for key, label in (("rho", "密度"), ("u", "速度"), ("p", "压强")):
        lines.append(f"| {label} | {_fmt(err[key][0])} | {_fmt(err[key][1])} |")
    if res.get("exact_error"):
        lines += ["", f"> 精确解未能计算：`{res['exact_error']}`"]
    lines += ["", "**结果图像**", ""]

    if all(key in figures for key in ("rho", "u", "p")):
        rho = _relative_image(report_path, figures["rho"])
        u = _relative_image(report_path, figures["u"])
        p = _relative_image(report_path, figures["p"])
        lines += [
            "| 密度 $\\rho$ | 速度 $u$ | 压强 $p$ |",
            "| :---: | :---: | :---: |",
            f'| <img src="{rho}" alt="{cfg["name"]} 密度" width="280"> '
            f'| <img src="{u}" alt="{cfg["name"]} 速度" width="280"> '
            f'| <img src="{p}" alt="{cfg["name"]} 压强" width="280"> |',
            "",
        ]
        if "overview" in figures:
            overview = _relative_image(report_path, figures["overview"])
            lines.append(f"三联图：[overview.png]({overview})")
    else:
        lines.append("> 未生成图像。")
    lines.append("")
    return lines


def build_report(items, report_path: Path, config_paths) -> str:
    report_path = Path(report_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_list = "、".join(f"`{Path(p).name}`" for p in config_paths)
    lines = [
        "# 一维 Euler 方程统一求解器测试报告",
        "",
        "本报告由统一的 FV/DG 求解器自动生成。程序支持单介质与双介质（单个移动界面）"
        "一维 Euler 方程组，数值格式、网格、流体参数与初始条件均由 YAML 配置。",
        "",
        f"> 由 `run_euler.py` 读取 {config_list} 自动生成。",
        f"> 生成时间：{now}",
        "",
        "## 算例汇总",
        "",
        "| # | 算例 | 格式 | 介质 | 阶数 | 通量 | 单元数 | 终止时间 | 步数 | 界面位置 | rho L1 | rho L∞ |",
        "| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(items, start=1):
        lines.append(_summary_row(index, item))
    lines += [
        "",
        "数值解以蓝色实线表示，精确解以橙色虚线表示。双介质算例的界面位置由"
        "界面黎曼问题推进，并在表中给出终了位置。",
        "",
        "## 数值方法",
        "",
        "- 单介质有限体积：一阶 LLF/HLL/HLLC，或 MUSCL + minmod 二阶重构；"
        "时间推进采用 SSP-RK2。",
        "- 单介质 DG：P0/P1/P2 Legendre 基，LLF/HLL/HLLC 数值通量，"
        "特征变量 TVB/minmod 限制器，SSP-RK2/RK3。",
        "- 双介质：界面所在单元用两相切割单元表示，界面速度由刚性气体精确黎曼求解器给出，"
        "相内数据采用 P0（等价于一阶 FV）或 P1/P2 DG 多项式。",
        "",
        "## 算例结果",
        "",
    ]
    for index, item in enumerate(items, start=1):
        lines += _case_section(index, item, report_path)
        lines.append("")

    lines += [
        "## 复现方式",
        "",
        "```bash",
        "cd CFD/Euler",
        "python run_euler.py                    # 运行 configs/ 下全部算例并生成 report.md",
        "python run_euler.py --list             # 列出全部算例",
        "python run_euler.py --case fv_single_sod",
        "python run_euler.py --config configs/fv_single_sod.yaml",
        "```",
        "",
        "新增算例只需在 `configs/` 下添加 YAML 文件，无需修改求解器代码。",
        "",
    ]
    return "\n".join(lines)


def write_report(items, report_path=None, config_paths=()):
    report_path = Path(report_path) if report_path is not None else DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_report(items, report_path, config_paths)
    report_path.write_text(text, encoding="utf-8")
    return report_path

#!/usr/bin/env python3
'''
一维刚性气体 / 反应流黎曼问题精确求解器的算例驱动与报告生成程序

读取 YAML 配置文件中的算例参数，调用精确 Riemann 求解器，输出
密度、速度、压强的一维精确解图像，并生成可在 GitHub 上直接渲染的
Markdown 测试报告。

用法：
    python run_cases.py                     # 读取 cases/*.yaml，运行全部算例并生成报告
    python run_cases.py --case inert_sod
    python run_cases.py --case inert_sod,detonation_cj
    python run_cases.py --list
    python run_cases.py --config cases/inert.yaml
    python run_cases.py --config cases/detonation.yaml cases/deflagration.yaml
    python run_cases.py --report ../report.md
'''
import argparse
import copy
import datetime
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Fluid as Fd
import FluidReact as FR
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS
import Exact_Riemann_Combustion as ERC

CASE_TYPES = ("inert", "detonation", "deflagration")
CASE_FILE_ORDER = {name: idx for idx, name in enumerate(CASE_TYPES)}
TYPE_LABELS = {
    "inert": "非反应（刚性气体）",
    "detonation": "爆轰",
    "deflagration": "爆燃",
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = SCRIPT_DIR / "cases"
DEFAULT_FIG_DIR = SCRIPT_DIR.parent / "fig"
DEFAULT_REPORT = SCRIPT_DIR.parent / "report.md"

DEFAULT_PLOT_CFG = {
    "dpi": 200,
    "figsize": [4.2, 3.0],
    "line_color": "#1f77b4",
}
DEFAULT_META = {
    "title": "刚性气体与反应流黎曼问题精确解测试报告",
    "description": "一维 Euler 方程组的精确 Riemann 求解器测试。包括非反应刚性气体、"
                   "爆燃波与爆轰波算例。",
}


# ----------------------------------------------------------------------
# 配置读取
# ----------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件 {path} 的顶层必须是映射")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    '''递归合并字典，override 优先'''
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _config_sort_key(path: Path):
    '''按 inert → detonation → deflagration 的顺序排列配置文件'''
    return (CASE_FILE_ORDER.get(path.stem, len(CASE_FILE_ORDER)), path.name)


def yaml_files_in(directory: Path) -> list:
    '''返回目录下按算例类型顺序排列的全部 YAML 文件'''
    files = [path for path in directory.iterdir()
             if path.is_file() and path.suffix in (".yaml", ".yml")]
    return [path.resolve() for path in sorted(files, key=_config_sort_key)]


def resolve_cli_path(value: str) -> Path:
    '''命令行中给出的路径：先按当前目录解析，不存在时按脚本目录解析'''
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = (Path.cwd() / path)
    if candidate.exists():
        return candidate.resolve()
    return (SCRIPT_DIR / path).resolve()


def discover_config_paths(config_args=None) -> list:
    '''
    确定要读取的 YAML 配置文件。

    :param config_args: 命令行 --config 的参数列表；为 None 时读取 cases/ 全部 YAML
    :return: 去重后的配置文件路径列表
    '''
    if not config_args:
        paths = yaml_files_in(DEFAULT_CASE_DIR)
        if not paths:
            raise FileNotFoundError(f"在 {DEFAULT_CASE_DIR} 中未找到 YAML 配置文件")
        return paths

    paths = []
    for value in config_args:
        path = resolve_cli_path(value)
        if path.is_dir():
            found = yaml_files_in(path)
            if not found:
                raise FileNotFoundError(f"目录 {path} 中没有 YAML 配置文件")
            paths.extend(found)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"配置文件不存在: {path}")

    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def load_configs(paths: list) -> list:
    '''读取多个 YAML 文件，并把文件路径记录在 _path 中'''
    configs = []
    for path in paths:
        config = load_yaml(path)
        config["_path"] = path
        configs.append(config)
    return configs


def prepare_all_cases(configs: list) -> list:
    '''合并多个配置文件中的算例，并检查算例名全局唯一'''
    cases = []
    by_name = {}
    for config in configs:
        path = Path(config["_path"])
        for case in prepare_cases(config):
            name = case["name"]
            if name in by_name:
                raise ValueError(
                    f"算例名称重复: {name}（{by_name[name]} 与 {path}）")
            by_name[name] = path
            case["_source_file"] = path
            cases.append(case)
    return cases


def report_meta(configs: list) -> dict:
    '''单个配置文件时使用其 meta，多个文件时使用汇总报告的 meta'''
    if len(configs) == 1:
        return deep_merge(DEFAULT_META, configs[0].get("meta") or {})
    return copy.deepcopy(DEFAULT_META)


def format_config_paths(paths: list) -> str:
    '''把配置文件路径格式化为报告中的相对路径说明'''
    labels = []
    base = SCRIPT_DIR.parent
    for path in paths:
        path = Path(path)
        try:
            shown = path.relative_to(base)
        except ValueError:
            shown = path
        labels.append(f"`{shown.as_posix()}`")
    return "、".join(labels)


def prepare_cases(config: dict) -> list:
    defaults = config.get("defaults", {}) or {}
    cases = config.get("cases", []) or []
    prepared = []
    names = set()
    for raw in cases:
        if "name" not in raw:
            raise ValueError("每个算例都必须提供 name")
        if raw["name"] in names:
            raise ValueError(f"算例名称重复: {raw['name']}")
        names.add(raw["name"])
        case = deep_merge(defaults, raw)
        if case.get("type") not in CASE_TYPES:
            raise ValueError(
                f"算例 {case['name']} 的 type 必须是 {CASE_TYPES} 之一，"
                f"当前为 {case.get('type')!r}")
        for key in ("left", "right", "domain"):
            if key not in case:
                raise ValueError(f"算例 {case['name']} 缺少 {key}")
        for key in ("x_L", "x_R", "x_c", "t_end", "N"):
            if key not in case["domain"]:
                raise ValueError(f"算例 {case['name']} 的 domain 缺少 {key}")
        if case["type"] == "deflagration" and "deflagration" not in case:
            raise ValueError(f"爆燃算例 {case['name']} 缺少 deflagration 参数 (K, Q)")
        prepared.append(case)
    return prepared


def select_cases(cases: list, names) -> list:
    if not names:
        return cases
    selected = []
    available = {c["name"] for c in cases}
    for name in names:
        if name not in available:
            raise ValueError(f"未知算例 {name!r}。可选算例: {sorted(available)}")
        selected.append(next(c for c in cases if c["name"] == name))
    return selected


# ----------------------------------------------------------------------
# 状态与波系信息
# ----------------------------------------------------------------------
def make_fluid(spec: dict, reactive: bool):
    if reactive:
        return FR.FluidReact(
            rho=float(spec["rho"]), u=float(spec["u"]), p=float(spec["p"]),
            gamma=float(spec.get("gamma", 1.4)), q=float(spec.get("q", 0.0)))
    return Fd.Fluid(
        rho=float(spec["rho"]), u=float(spec["u"]), p=float(spec["p"]),
        gamma=float(spec.get("gamma", 1.4)), p_inf=float(spec.get("p_inf", 0.0)))


def _p_inf(W) -> float:
    return float(getattr(W, "p_inf", 0.0))


def nonreactive_left_wave(W, p_star: float, u_star: float) -> dict:
    '''WL 到中间状态的非反应波（激波或稀疏波）'''
    gamma = W.gamma
    rho0, u0, p0 = W.rho, W.u, W.p
    pinf = _p_inf(W)
    mu2 = (gamma - 1) / (gamma + 1)
    if p_star >= p0:
        A = 2.0 / ((gamma + 1) * rho0)
        B = (gamma - 1) * (p0 + pinf) / (gamma + 1)
        speed = u0 - (1.0 / rho0) * math.sqrt((B + pinf + p_star) / A)
        rho_star = rho0 * ((gamma + 1) * (p_star + pinf) + (gamma - 1) * (p0 + pinf)) / \
                          ((gamma - 1) * (p_star + pinf) + (gamma + 1) * (p0 + pinf))
        return {"name": "左波", "type": "激波",
                "speeds": {"S_L": speed}, "rho_star": rho_star}
    a0 = W.a()
    a_star = a0 * ((p_star + pinf) / (p0 + pinf)) ** ((gamma - 1) / (2 * gamma))
    rho_star = rho0 * ((p_star + pinf) / (p0 + pinf)) ** (1.0 / gamma)
    return {"name": "左波", "type": "稀疏波",
            "speeds": {"S_head": u0 - a0, "S_tail": u_star - a_star},
            "rho_star": rho_star}


def nonreactive_right_wave(W, p_star: float, u_star: float) -> dict:
    '''中间状态到 WR 的非反应波（激波或稀疏波）'''
    gamma = W.gamma
    rho0, u0, p0 = W.rho, W.u, W.p
    pinf = _p_inf(W)
    if p_star >= p0:
        A = 2.0 / ((gamma + 1) * rho0)
        B = (gamma - 1) * (p0 + pinf) / (gamma + 1)
        speed = u0 + (1.0 / rho0) * math.sqrt((B + pinf + p_star) / A)
        rho_star = rho0 * ((gamma + 1) * (p_star + pinf) + (gamma - 1) * (p0 + pinf)) / \
                          ((gamma - 1) * (p_star + pinf) + (gamma + 1) * (p0 + pinf))
        return {"name": "右波", "type": "激波",
                "speeds": {"S_R": speed}, "rho_star": rho_star}
    a0 = W.a()
    a_star = a0 * ((p_star + pinf) / (p0 + pinf)) ** ((gamma - 1) / (2 * gamma))
    rho_star = rho0 * ((p_star + pinf) / (p0 + pinf)) ** (1.0 / gamma)
    return {"name": "右波", "type": "稀疏波",
            "speeds": {"S_tail": u_star + a_star, "S_head": u0 + a0},
            "rho_star": rho_star}


# ----------------------------------------------------------------------
# 算例计算
# ----------------------------------------------------------------------
def compute_inert(case: dict):
    d = case["domain"]
    tol = float(case["numerical"]["tol"])
    maxit = int(case["numerical"]["maxit"])
    method = str(case["numerical"].get("method", "newton")).lower()

    WL = make_fluid(case["left"], reactive=False)
    WR = make_fluid(case["right"], reactive=False)
    solver = ERS.ExactRiemannSolver(WL, WR, tol, maxit)
    if method == "brent":
        p_star, u_star = solver.solver_brent()
    else:
        p_star, u_star = solver.solver()

    complete = ERCS.Riemann_Complete_solve(
        p_star, u_star, WL, WR,
        float(d["t_end"]), float(d["x_L"]), float(d["x_R"]), int(d["N"]))
    x, rho, u, p = complete.solve()

    left = nonreactive_left_wave(WL, p_star, u_star)
    right = nonreactive_right_wave(WR, p_star, u_star)
    waves = [left, {"name": "接触间断", "type": "-", "speeds": {"S": u_star}}, right]
    star = {
        "p_star": p_star,
        "u_star": u_star,
        "rho_star_L": left["rho_star"],
        "rho_star_R": right["rho_star"],
    }
    return x, rho, u, p, star, waves, {"method": method}


def compute_detonation(case: dict):
    d = case["domain"]
    tol = float(case["numerical"]["tol"])
    maxit = int(case["numerical"]["maxit"])
    method = str(case["numerical"].get("method", "newton")).lower()

    WL = make_fluid(case["left"], reactive=True)
    WR = make_fluid(case["right"], reactive=True)
    solver = ERC.ExactRiemannSolverDetonation(WL, WR, tol, maxit, True)
    if method == "brent":
        p_star, u_star = solver.solver_brent()
    else:
        p_star, u_star = solver.solver()
    p_cj, u_cj, rho_cj = solver.p_cj, solver.u_cj, solver.rho_cj

    complete = ERC.RiemannCompleteSolDet(
        p_star, u_star, WL, WR,
        float(d["t_end"]), float(d["x_L"]), float(d["x_R"]),
        float(d["x_c"]), int(d["N"]), True)
    x, rho, u, p = complete.complete_solution()

    left = nonreactive_left_wave(WL, p_star, u_star)
    gamma = WR.gamma
    if p_star >= p_cj:
        # 强爆轰：单道激波
        m = (p_star - WR.p) / (u_star - WR.u)
        S_det = WR.u + m / WR.rho
        rho_star_R = WR.rho * (S_det - WR.u) / (S_det - u_star)
        right = {"name": "右波", "type": "强爆轰波",
                 "speeds": {"S_det": S_det}, "rho_star": rho_star_R}
        wave_type = "强爆轰"
    else:
        # CJ 爆轰 + 稀疏波复合波
        a_cj = math.sqrt(gamma * p_cj / rho_cj)
        a_star_R = a_cj * (p_star / p_cj) ** ((gamma - 1) / (2 * gamma))
        S_det = u_cj + a_cj
        S_tail = u_star + a_star_R
        rho_star_R = rho_cj * (p_star / p_cj) ** (1.0 / gamma)
        right = {"name": "右波", "type": "CJ 爆轰 + 稀疏波",
                 "speeds": {"S_det": S_det, "S_tail": S_tail},
                 "rho_star": rho_star_R}
        wave_type = "CJ 爆轰"

    waves = [left, {"name": "接触间断", "type": "-", "speeds": {"S": u_star}}, right]
    star = {
        "p_star": p_star,
        "u_star": u_star,
        "rho_star_L": left["rho_star"],
        "rho_star_R": right["rho_star"],
        "p_cj": p_cj,
        "u_cj": u_cj,
        "rho_cj": rho_cj,
        "right_wave_type": wave_type,
    }
    return x, rho, u, p, star, waves, {"method": method}


def compute_deflagration(case: dict):
    d = case["domain"]
    num = case["numerical"]
    tol = float(num["tol"])
    maxit = int(num["maxit"])
    K = float(case["deflagration"]["K"])
    Q = int(case["deflagration"].get("Q", 2))

    WL = make_fluid(case["left"], reactive=True)
    WR = make_fluid(case["right"], reactive=True)
    solver = ERC.ExactRiemannSolverDeflagration(WL, WR, K, Q, tol, maxit, True)
    p_star, u_star, p0, rho0, V = solver.solver_brent()

    complete = ERC.RiemannCompleteSolDef(
        p_star, u_star, p0, rho0, WL, WR,
        float(d["t_end"]), float(d["x_L"]), float(d["x_R"]),
        float(d["x_c"]), int(d["N"]), True,
        V=V, rho1=solver.rho1, u0=solver.u0)
    x, rho, u, p = complete.complete_solution()

    left = nonreactive_left_wave(WL, p_star, u_star)
    right = nonreactive_right_wave(WR, p0, solver.u0)
    flame = {"name": "爆燃波", "type": "火焰", "speeds": {"V": V}}
    waves = [left, {"name": "接触间断", "type": "-", "speeds": {"S": u_star}}, flame, right]
    star = {
        "p_star": p_star,
        "u_star": u_star,
        "rho_star_L": left["rho_star"],
        "rho1": solver.rho1,
        "p0": p0,
        "u0": solver.u0,
        "rho0": rho0,
        "V": V,
        "K": K,
        "Q": Q,
    }
    return x, rho, u, p, star, waves, {"method": "brent"}


def run_case(case: dict, fig_dir: Path, plot_cfg: dict, verbose: bool = True) -> dict:
    kind = case["type"]
    if kind == "inert":
        x, rho, u, p, star, waves, info = compute_inert(case)
    elif kind == "detonation":
        x, rho, u, p, star, waves, info = compute_detonation(case)
    elif kind == "deflagration":
        x, rho, u, p, star, waves, info = compute_deflagration(case)
    else:  # pragma: no cover
        raise ValueError(kind)

    if not (np.all(np.isfinite(rho)) and np.all(np.isfinite(u)) and np.all(np.isfinite(p))):
        raise RuntimeError(f"算例 {case['name']} 的计算结果包含非有限值")
    if np.any(rho <= 0.0):
        raise RuntimeError(f"算例 {case['name']} 出现非正密度")
    if np.any(p <= 0.0):
        raise RuntimeError(f"算例 {case['name']} 出现非正压强")

    case_fig_dir = fig_dir / case["name"]
    case_fig_dir.mkdir(parents=True, exist_ok=True)
    figures = plot_case(case, x, rho, u, p, waves, case_fig_dir, plot_cfg)

    if verbose:
        print(f"[{case['name']}] p* = {star['p_star']:.6e}, u* = {star['u_star']:.6e}")
        for wave in waves:
            speed_text = ", ".join(f"{k}={v:.6e}" for k, v in wave["speeds"].items())
            print(f"    {wave['name']} ({wave['type']}): {speed_text}")
        print(f"    图像已保存: {case_fig_dir}")

    return {
        "name": case["name"],
        "description": case.get("description", ""),
        "type": kind,
        "case": case,
        "star": star,
        "waves": waves,
        "info": info,
        "figures": figures,
        "solution_range": {
            "rho": (float(rho.min()), float(rho.max())),
            "u": (float(u.min()), float(u.max())),
            "p": (float(p.min()), float(p.max())),
            "N": int(len(x)),
        },
    }


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def _wave_positions(case: dict, waves: list):
    d = case["domain"]
    x_c = float(d["x_c"])
    t_end = float(d["t_end"])
    x_L = float(d["x_L"])
    x_R = float(d["x_R"])
    positions = []
    for wave in waves:
        for _, speed in wave["speeds"].items():
            xp = x_c + speed * t_end
            if x_L < xp < x_R:
                positions.append(xp)
    return positions


def plot_case(case: dict, x, rho, u, p, waves, out_dir: Path, plot_cfg: dict) -> dict:
    dpi = int(plot_cfg.get("dpi", 200))
    figsize = tuple(plot_cfg.get("figsize", [4.2, 3.0]))
    color = plot_cfg.get("line_color", "#1f77b4")
    x_L, x_R = float(case["domain"]["x_L"]), float(case["domain"]["x_R"])
    positions = _wave_positions(case, waves)

    quantities = [
        ("rho", rho, r"$\rho$", "density"),
        ("u", u, r"$u$", "velocity"),
        ("p", p, r"$p$", "pressure"),
    ]
    figures = {}

    for key, values, ylabel, title in quantities:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.plot(x, values, color=color, linewidth=1.8, label="Exact")
        for xp in positions:
            ax.axvline(xp, color="0.55", linestyle="--", linewidth=0.8)
        ax.set_xlim(x_L, x_R)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{case['name']}: {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        path = out_dir / f"{key}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        figures[key] = path

    # 三联图
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.2), sharex=True, constrained_layout=True)
    for ax, (key, values, ylabel, title) in zip(axes, quantities):
        ax.plot(x, values, color=color, linewidth=1.8)
        for xp in positions:
            ax.axvline(xp, color="0.55", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(r"$x$")
    axes[-1].set_xlim(x_L, x_R)
    fig.suptitle(f"{case['name']} — exact solution at $t={case['domain']['t_end']:g}$")
    overview = out_dir / "overview.png"
    fig.savefig(overview, dpi=dpi)
    plt.close(fig)
    figures["overview"] = overview
    return figures


# ----------------------------------------------------------------------
# Markdown 报告
# ----------------------------------------------------------------------
def fnum(value) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        return f"{value:.6e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _rel_link(path: Path, report_dir: Path) -> str:
    return os.path.relpath(path, report_dir).replace(os.sep, "/")


def _state_table(case: dict) -> str:
    left, right = case["left"], case["right"]
    reactive = case["type"] in ("detonation", "deflagration")
    if reactive:
        extra = "q (J/kg)"
    else:
        extra = "p_inf (Pa)"
    rows = [
        ("rho (kg/m^3)", "rho", left, right),
        ("u (m/s)", "u", left, right),
        ("p (Pa)", "p", left, right),
        ("gamma", "gamma", left, right),
        (extra, "q" if reactive else "p_inf", left, right),
    ]
    lines = ["| 参数 | 左状态 | 右状态 |", "| --- | ---: | ---: |"]
    for label, key, L, R in rows:
        lines.append(f"| {label} | {fnum(L.get(key, 0.0))} | {fnum(R.get(key, 0.0))} |")
    d = case["domain"]
    lines.append(f"| 计算区域 | x = [{fnum(d['x_L'])}, {fnum(d['x_R'])}], "
                 f"x_c = {fnum(d['x_c'])} | t = {fnum(d['t_end'])}, N = {d['N']} |")
    num = case["numerical"]
    lines.append(f"| 数值参数 | {num.get('method', 'newton')}, "
                 f"tol = {fnum(num['tol'])}, maxit = {num['maxit']} | |")
    return "\n".join(lines)


def _wave_table(waves: list) -> str:
    lines = ["| 波 | 类型 | 速度 (m/s) |", "| --- | --- | --- |"]
    for wave in waves:
        speeds = ", ".join(f"{k} = {fnum(v)}" for k, v in wave["speeds"].items())
        lines.append(f"| {wave['name']} | {wave['type']} | {speeds} |")
    return "\n".join(lines)


def _star_table(result: dict) -> str:
    star = result["star"]
    kind = result["type"]
    rows = []
    if kind == "inert":
        rows = [
            ("中间压强 p* (Pa)", star["p_star"]),
            ("接触间断速度 u* (m/s)", star["u_star"]),
            ("左波后密度 rho*_L (kg/m^3)", star["rho_star_L"]),
            ("右波后密度 rho*_R (kg/m^3)", star["rho_star_R"]),
        ]
    elif kind == "detonation":
        rows = [
            ("中间压强 p* (Pa)", star["p_star"]),
            ("接触间断速度 u* (m/s)", star["u_star"]),
            ("左波后密度 rho*_L (kg/m^3)", star["rho_star_L"]),
            ("右波后密度 rho*_R (kg/m^3)", star["rho_star_R"]),
            ("CJ 压强 p_CJ (Pa)", star["p_cj"]),
            ("CJ 速度 u_CJ (m/s)", star["u_cj"]),
            ("CJ 密度 rho_CJ (kg/m^3)", star["rho_cj"]),
        ]
    else:
        rows = [
            ("火焰波后压强 p1 (Pa)", star["p_star"]),
            ("接触间断速度 u1 (m/s)", star["u_star"]),
            ("火焰波后密度 rho1 (kg/m^3)", star["rho1"]),
            ("火焰波前压强 p0 (Pa)", star["p0"]),
            ("火焰波前速度 u0 (m/s)", star["u0"]),
            ("火焰波前密度 rho0 (kg/m^3)", star["rho0"]),
            ("火焰速度 V (m/s)", star["V"]),
            ("左波后密度 rho*_L (kg/m^3)", star["rho_star_L"]),
        ]
    lines = ["| 量 | 值 |", "| --- | ---: |"]
    lines += [f"| {label} | {fnum(value)} |" for label, value in rows]
    return "\n".join(lines)


def build_report(results: list, report_path: Path, config_paths: list, meta: dict) -> str:
    report_dir = report_path.parent
    title = meta.get("title", "刚性气体黎曼问题精确解测试报告")
    description = meta.get("description", "")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    if description:
        lines.append(description)
        lines.append("")
    sources = format_config_paths(config_paths)
    lines.append(f"> 由 `code/run_cases.py` 读取 {sources} 自动生成。")
    lines.append(f"> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 汇总表
    lines.append("## 算例汇总")
    lines.append("")
    lines.append("| # | 算例 | 类型 | p* (Pa) | u* (m/s) | 左波 | 右波 |")
    lines.append("| ---: | --- | --- | ---: | ---: | --- | --- |")
    for idx, result in enumerate(results, start=1):
        star = result["star"]
        left_wave = result["waves"][0]
        right_wave = result["waves"][-1]
        lines.append(
            f"| {idx} | {result['name']} | {TYPE_LABELS.get(result['type'], result['type'])} | "
            f"{fnum(star['p_star'])} | {fnum(star['u_star'])} | "
            f"{left_wave['type']} | {right_wave['type']} |")
    lines.append("")
    lines.append("> 表中 p*、u* 对爆燃算例指火焰波后的中间状态 (p1, u1)。")
    lines.append("")

    # 方法说明
    lines.append("## 求解方法")
    lines.append("")
    lines.append("控制方程为无粘一维 Euler 方程组。对于刚性气体，状态方程为")
    lines.append("")
    lines.append("$$")
    lines.append("p = (\\gamma-1)\\rho e - \\gamma p_\\infty .")
    lines.append("$$")
    lines.append("")
    lines.append("非反应波（激波/稀疏波）的匹配方程 $f_L(p)+f_R(p)+u_R-u_L=0$ "
                 "分别采用 Newton 迭代和 Brent 方法求解；")
    lines.append("反应流算例在匹配方程中引入爆轰/爆燃的 Rayleigh-Hugoniot 关系。")
    lines.append("所有图像由精确解在均匀网格上直接采样得到，竖直虚线标出各道波的位置。")
    lines.append("若某个 $p^*$ 与对应端部状态的压强之差在求解精度以内，则该波为"
                 "零强度波（波速等于当地声速附近的特征速度），图像上不会出现间断。")
    lines.append("")

    # 每个算例
    for idx, result in enumerate(results, start=1):
        case = result["case"]
        figures = result["figures"]
        lines.append(f"## {idx}. {result['name']}")
        lines.append("")
        if result["description"]:
            lines.append(result["description"])
            lines.append("")
        lines.append("**初始条件与计算参数**")
        lines.append("")
        lines.append(_state_table(case))
        lines.append("")
        lines.append("**求解结果**")
        lines.append("")
        lines.append(_star_table(result))
        lines.append("")
        lines.append("**波系**")
        lines.append("")
        lines.append(_wave_table(result["waves"]))
        lines.append("")
        lines.append("**结果图像**")
        lines.append("")
        lines.append("| 密度 $\\rho$ | 速度 $u$ | 压强 $p$ |")
        lines.append("| :---: | :---: | :---: |")
        cells = []
        for key, label in (("rho", "密度"), ("u", "速度"), ("p", "压强")):
            link = _rel_link(figures[key], report_dir)
            alt = f"{result['name']} {label}"
            cells.append(f'<img src="{link}" alt="{alt}" width="280">')
        lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"三联图：[overview.png]({_rel_link(figures['overview'], report_dir)})")
        lines.append("")

    lines.append("## 复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append("cd CFD/Riemann/code")
    lines.append("pip install -r requirements.txt")
    lines.append("python run_cases.py                                  # 运行全部算例")
    lines.append("python run_cases.py --config cases/inert.yaml        # 只运行非反应算例")
    lines.append("python run_cases.py --config cases/detonation.yaml   # 只运行爆轰算例")
    lines.append("python run_cases.py --config cases/deflagration.yaml # 只运行爆燃算例")
    lines.append("python run_cases.py --list                           # 列出全部算例")
    lines.append("```")
    lines.append("")
    lines.append("新算例按类型追加到 `cases/` 下对应的 YAML 文件中，无需修改代码。")
    lines.append("")
    return "\n".join(lines)


def write_report(results: list, report_path: Path, config_paths: list, meta: dict):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = build_report(results, report_path, config_paths, meta)
    report_path.write_text(content, encoding="utf-8")
    print(f"报告已写入: {report_path}")


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="读取 YAML 配置，运行一维黎曼问题精确解算例并生成 Markdown 报告")
    parser.add_argument("--config", nargs="+", default=None,
                        help="一个或多个 YAML 配置文件或目录；默认读取 cases/*.yaml")
    parser.add_argument("--case", default=None,
                        help="要运行的算例名称，多个名称用逗号分隔")
    parser.add_argument("--list", action="store_true", help="列出全部算例后退出")
    parser.add_argument("--fig-dir", default=None,
                        help=f"图像输出目录（默认 {DEFAULT_FIG_DIR}）")
    parser.add_argument("--report", default=None,
                        help="报告输出路径（默认 ../report.md；只加载单个配置文件时为 "
                             "../report_<文件名>.md）")
    parser.add_argument("--no-report", action="store_true", help="不生成 Markdown 报告")
    parser.add_argument("--quiet", action="store_true", help="不打印每个算例的求解结果")
    args = parser.parse_args(argv)

    config_paths = discover_config_paths(args.config)
    configs = load_configs(config_paths)
    cases = prepare_all_cases(configs)

    if args.list:
        for case in cases:
            source = Path(case["_source_file"]).name
            print(f"{case['name']:24s} {case['type']:14s} {source:22s} "
                  f"{case.get('description', '')}")
        return 0

    names = None
    if args.case:
        names = [name.strip() for name in args.case.split(",") if name.strip()]
    selected = select_cases(cases, names)

    if args.fig_dir:
        fig_dir = Path(args.fig_dir).expanduser().resolve()
    else:
        fig_dir = DEFAULT_FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
    elif args.config is not None and len(config_paths) == 1:
        report_path = DEFAULT_REPORT.parent / f"report_{config_paths[0].stem}.md"
    else:
        report_path = DEFAULT_REPORT

    results = []
    for case in selected:
        plot_cfg = deep_merge(DEFAULT_PLOT_CFG, case.get("plot") or {})
        results.append(run_case(case, fig_dir, plot_cfg, verbose=not args.quiet))

    subset_configs = False
    if args.config is not None:
        try:
            default_config_paths = discover_config_paths(None)
        except FileNotFoundError:
            default_config_paths = []
        subset_configs = set(config_paths) != set(default_config_paths)

    if not args.no_report:
        # 只运行部分算例时，默认不覆盖完整报告，避免丢失其它算例的结果。
        if args.case and not args.report:
            print("仅运行了部分算例，未生成报告（避免覆盖完整报告）。"
                  "如需生成请显式指定 --report <路径>。")
        elif subset_configs and len(config_paths) > 1 and not args.report:
            print("仅加载了部分配置文件，未生成报告（避免覆盖完整报告）。"
                  "如需生成请显式指定 --report <路径>。")
        else:
            write_report(results, report_path, config_paths, report_meta(configs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

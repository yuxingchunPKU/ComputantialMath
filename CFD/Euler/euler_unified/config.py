"""YAML configuration loading and validation for the unified Euler solver."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


EULER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = EULER_ROOT / "configs"
DEFAULT_FIG_DIR = EULER_ROOT / "figures"
DEFAULT_REPORT = EULER_ROOT / "report.md"

_SCHEMES = {"fv", "dg"}
_MEDIA = {"single", "two"}
_FLUXES = {"llf", "rusanov", "hll", "hllc"}
_BOUNDARIES = {"transmissive", "periodic", "reflective"}
_INITIAL_TYPES = {"sod", "smooth", "constant"}

_DEFAULTS: dict[str, Any] = {
    "scheme": "fv",
    "medium": "single",
    "order": 2,
    "flux": "llf",
    "description": "",
    "time": {"t_end": 0.2, "cfl": 0.4},
    "mesh": {"x_min": 0.0, "x_max": 1.0, "nx": 200, "boundary": "transmissive"},
    "fluids": {"gamma": 1.4, "p_inf": 0.0},
    "initial": {
        "type": "sod",
        "x_interface": 0.5,
        "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
        "right": {"rho": 0.125, "u": 0.0, "p": 0.1},
    },
    "limiter": {"tvb": 0.0},
    "output": {"dpi": 160, "figsize": [4.2, 3.0]},
}


def _deep_update(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _as_float(value, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _as_int(value, name: str) -> int:
    ivalue = int(value)
    if ivalue != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return ivalue


def _validate_state(state, name: str) -> dict:
    state = _require_mapping(state, name)
    missing = {"rho", "u", "p"} - set(state)
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    return {key: _as_float(state[key], f"{name}.{key}") for key in ("rho", "u", "p")}


def load_config(path: str | Path) -> dict:
    """Load one YAML file and return a validated case dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    raw = _require_mapping(raw, str(path))
    cfg = _deep_update(_DEFAULTS, raw)

    cfg["name"] = str(cfg.get("name") or path.stem)
    cfg["config_path"] = str(path)

    scheme = str(cfg["scheme"]).lower()
    medium = str(cfg["medium"]).lower()
    if scheme not in _SCHEMES:
        raise ValueError(f"{path}: scheme must be one of {sorted(_SCHEMES)}")
    if medium not in _MEDIA:
        raise ValueError(f"{path}: medium must be one of {sorted(_MEDIA)}")
    cfg["scheme"] = scheme
    cfg["medium"] = medium

    order = _as_int(cfg["order"], f"{path}: order")
    if scheme == "fv":
        if order not in (1, 2):
            raise ValueError(f"{path}: finite-volume order must be 1 or 2")
        if medium == "two" and order != 1:
            raise ValueError(
                f"{path}: two-medium finite volume currently supports order: 1; "
                "use scheme: dg for high-order two-medium data"
            )
    else:
        if order not in (0, 1, 2):
            raise ValueError(f"{path}: DG order must be 0, 1 or 2")
    cfg["order"] = order

    flux = str(cfg["flux"]).lower()
    if flux not in _FLUXES:
        raise ValueError(f"{path}: flux must be one of {sorted(_FLUXES)}")
    cfg["flux"] = flux

    time = _require_mapping(cfg["time"], f"{path}: time")
    time["t_end"] = _as_float(time.get("t_end", 0.2), f"{path}: time.t_end")
    time["cfl"] = _as_float(time.get("cfl", 0.4), f"{path}: time.cfl")
    if time["t_end"] <= 0.0:
        raise ValueError(f"{path}: time.t_end must be positive")
    if not 0.0 < time["cfl"] <= 1.0:
        raise ValueError(f"{path}: time.cfl must lie in (0, 1]")

    mesh = _require_mapping(cfg["mesh"], f"{path}: mesh")
    mesh["x_min"] = _as_float(mesh.get("x_min", 0.0), f"{path}: mesh.x_min")
    mesh["x_max"] = _as_float(mesh.get("x_max", 1.0), f"{path}: mesh.x_max")
    mesh["nx"] = _as_int(mesh.get("nx", 200), f"{path}: mesh.nx")
    if mesh["x_max"] <= mesh["x_min"]:
        raise ValueError(f"{path}: mesh.x_max must exceed mesh.x_min")
    if mesh["nx"] < 2:
        raise ValueError(f"{path}: mesh.nx must be at least 2")
    boundary = str(mesh.get("boundary", "transmissive")).lower()
    if boundary not in _BOUNDARIES:
        raise ValueError(f"{path}: boundary must be one of {sorted(_BOUNDARIES)}")
    if medium == "two" and boundary == "reflective":
        raise ValueError(f"{path}: reflective boundaries are not supported for two-medium cases")
    mesh["boundary"] = boundary

    fluids = _require_mapping(cfg["fluids"], f"{path}: fluids")
    if medium == "single":
        gamma = _as_float(fluids.get("gamma", 1.4), f"{path}: fluids.gamma")
        p_inf = _as_float(fluids.get("p_inf", 0.0), f"{path}: fluids.p_inf")
        if gamma <= 1.0:
            raise ValueError(f"{path}: fluids.gamma must exceed 1")
        cfg["fluids"] = {"gamma": gamma, "p_inf": p_inf}
    else:
        left = fluids.get("left", fluids.get("phase0"))
        right = fluids.get("right", fluids.get("phase1"))
        if left is None or right is None:
            raise ValueError(f"{path}: two-medium fluids need left and right entries")
        norm = {}
        for key, value in (("left", left), ("right", right)):
            value = _require_mapping(value, f"{path}: fluids.{key}")
            gamma = _as_float(value.get("gamma", 1.4), f"{path}: fluids.{key}.gamma")
            p_inf = _as_float(value.get("p_inf", 0.0), f"{path}: fluids.{key}.p_inf")
            if gamma <= 1.0:
                raise ValueError(f"{path}: fluids.{key}.gamma must exceed 1")
            norm[key] = {"gamma": gamma, "p_inf": p_inf}
        cfg["fluids"] = norm

    initial = _require_mapping(cfg["initial"], f"{path}: initial")
    itype = str(initial.get("type", "sod")).lower()
    if itype not in _INITIAL_TYPES:
        raise ValueError(f"{path}: initial.type must be one of {sorted(_INITIAL_TYPES)}")
    initial["type"] = itype
    initial["x_interface"] = _as_float(
        initial.get("x_interface", 0.5 * (mesh["x_min"] + mesh["x_max"])),
        f"{path}: initial.x_interface",
    )
    if not mesh["x_min"] < initial["x_interface"] < mesh["x_max"]:
        raise ValueError(f"{path}: initial.x_interface must lie inside the domain")
    if itype in ("sod", "constant"):
        initial["left"] = _validate_state(initial.get("left"), f"{path}: initial.left")
        initial["right"] = _validate_state(initial.get("right"), f"{path}: initial.right")
    elif itype == "smooth":
        initial["rho0"] = _as_float(initial.get("rho0", 1.0), f"{path}: initial.rho0")
        initial["rho_amp"] = _as_float(initial.get("rho_amp", 0.2), f"{path}: initial.rho_amp")
        initial["k"] = _as_float(initial.get("k", 1.0), f"{path}: initial.k")
        initial["u0"] = _as_float(initial.get("u0", 1.0), f"{path}: initial.u0")
        initial["p0"] = _as_float(initial.get("p0", 1.0), f"{path}: initial.p0")
    cfg["initial"] = initial

    limiter = _require_mapping(cfg.get("limiter", {}), f"{path}: limiter")
    limiter["tvb"] = _as_float(limiter.get("tvb", 0.0), f"{path}: limiter.tvb")
    cfg["limiter"] = limiter

    output = _require_mapping(cfg.get("output", {}), f"{path}: output")
    out_dir = output.get("dir") or f"figures/{cfg['name']}"
    output["dir"] = str(out_dir)
    output["dpi"] = _as_int(output.get("dpi", 160), f"{path}: output.dpi")
    figsize = output.get("figsize", [4.2, 3.0])
    if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
        raise ValueError(f"{path}: output.figsize must be [width, height]")
    output["figsize"] = [float(figsize[0]), float(figsize[1])]
    cfg["output"] = output

    return cfg


def discover_config_paths(config_args=None) -> list[Path]:
    """Return the YAML files to run.

    ``config_args`` may contain files, directories, or be ``None`` (all YAML
    files in the default ``configs/`` directory).
    """
    if config_args is None:
        if not DEFAULT_CONFIG_DIR.is_dir():
            raise FileNotFoundError(f"default config directory not found: {DEFAULT_CONFIG_DIR}")
        paths = sorted(DEFAULT_CONFIG_DIR.glob("*.yaml"))
        if not paths:
            raise FileNotFoundError(f"no YAML files found in {DEFAULT_CONFIG_DIR}")
        return paths

    paths: list[Path] = []
    for item in config_args:
        path = Path(item).expanduser()
        if path.is_dir():
            found = sorted(path.glob("*.yaml"))
            if not found:
                raise FileNotFoundError(f"no YAML files found in {path}")
            paths.extend(found)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"configuration not found: {path}")
    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for path in paths:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def output_dir(cfg: dict) -> Path:
    path = Path(cfg["output"]["dir"])
    if not path.is_absolute():
        path = EULER_ROOT / path
    return path

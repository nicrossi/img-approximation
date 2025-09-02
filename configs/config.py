from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import datetime as _dt
import re
import copy
import yaml


__all__ = ["load_config"]


# ==== YAML helpers ====

def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Top-level YAML must be a mapping (dict). File: {path}")
        return data
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {path}: {e}") from e


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge `updates` into `base` (dicts only). Returns a NEW dict."""
    out = copy.deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_update(out[k], v)  # type: ignore[index]
        else:
            out[k] = copy.deepcopy(v)
    return out


# ==== Overrides ====

def _coerce_scalar(text: str) -> Any:
    """Parse scalar/list/dict from a string override.
    Usa YAML loader para soportar: ints, floats, bools, null, listas, dicts.
    """
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def _parse_dot_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    """Convert --set a.b.c=42 --set renderer.backend=opencv → nested dict."""
    tree: Dict[str, Any] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise ValueError(f"Invalid override (missing '='): {raw!r}")
        path, value = raw.split("=", 1)
        keys = [k for k in path.strip().split(".") if k]
        if not keys:
            raise ValueError(f"Invalid override path: {raw!r}")
        node = tree
        for k in keys[:-1]:
            node = node.setdefault(k, {})  # type: ignore[assignment]
            if not isinstance(node, dict):
                raise ValueError(f"Override path conflicts with non-dict at {k} in {raw!r}")
        node[keys[-1]] = _coerce_scalar(value.strip())
    return tree


# ==== Substitutions ====

_SUB_RE = re.compile(r"\$\{([^}]+)\}")

def _lookup_path(cfg: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _expand_substitutions_in_str(cfg: Mapping[str, Any], s: str) -> str:
    """Replace ${a.b.c} inside a string using values from the config mapping."""
    def _repl(m: re.Match[str]) -> str:
        path = m.group(1).strip()
        try:
            val = _lookup_path(cfg, path)
        except KeyError:
            return m.group(0)
        return str(val)
    return _SUB_RE.sub(_repl, s)


def _expand_substitutions(cfg: Any, root: Mapping[str, Any]) -> Any:
    """Recursively expand ${...} in all string leaves of cfg, using root for lookups."""
    if isinstance(cfg, str):
        return _expand_substitutions_in_str(root, cfg)
    if isinstance(cfg, list):
        return [_expand_substitutions(v, root) for v in cfg]
    if isinstance(cfg, tuple):
        return tuple(_expand_substitutions(v, root) for v in cfg)
    if isinstance(cfg, dict):
        return {k: _expand_substitutions(v, root) for k, v in cfg.items()}
    return cfg


# ==== Public API ====

def load_config(config_path: str, profile: str | None = None, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    """
    Carga una config YAML, aplica un perfil opcional y overrides por CLI.

    Args:
        config_path: ruta al YAML base (e.g., 'configs/base.yaml').
        profile: nombre del perfil (e.g., 'fast') → se busca en 'profiles/<profile>.yaml'
                 relativo al YAML base. También acepta ruta absoluta a un YAML.
        overrides: lista de strings en notación 'a.b.c=valor'. El valor se parsea con YAML.

    Returns:
        dict con la configuración final. Aplica:
          - merge (base <- profile <- overrides)
          - expansión de placeholders ${...} (p.ej. ${experiment.name})
          - timestamp YYYYMMDD-HHMMSS añadido a experiment.output_dir (si existe)
    """
    base_path = Path(config_path).expanduser().resolve()
    base_cfg = _read_yaml(base_path)

    merged = base_cfg
    if profile:
        prof_path = Path(profile)
        if not prof_path.is_file():
            # buscar en 'profiles/<profile>.yaml' al lado del base
            prof_path = base_path.with_name("profiles").joinpath(f"{profile}.yaml")
        prof_cfg = _read_yaml(prof_path)
        merged = _deep_update(merged, prof_cfg)

    if overrides:
        ov = _parse_dot_overrides(overrides)
        merged = _deep_update(merged, ov)

    merged = _expand_substitutions(merged, merged)

    # 4) Agregar timestamp al output_dir si existe experiment.output_dir
    try:
        exp = merged.get("experiment", {})
        out_dir = exp.get("output_dir")
        if isinstance(out_dir, str) and out_dir:
            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            # No crear directorio acá; solo devolvemos el path con timestamp
            merged["experiment"]["output_dir"] = str(Path(out_dir) / ts)
    except Exception:
        # Si faltan claves o no es string, lo ignoramos silenciosamente
        pass

    return merged


from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:

    if isinstance(value, str):
        def _replace(match: "re.Match[str]") -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            return default if default is not None else ""

        return _ENV_PLACEHOLDER.sub(_replace, value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Union[str, Path]) -> Dict:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)


class ConfigLoader:

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self._base_config: Optional[Dict] = None
        self._servers_config: Optional[Dict] = None
        self._datasets_config: Optional[Dict] = None
        self._sweeps_config: Optional[Dict] = None

    def _load_base(self) -> Dict:
        if self._base_config is None:
            base_path = self.config_dir / "base.yaml"
            if base_path.exists():
                self._base_config = load_yaml(base_path)
            else:
                self._base_config = {}
        return self._base_config.copy()

    def _load_servers(self) -> Dict:
        if self._servers_config is None:
            servers_path = self.config_dir / "servers.yaml"
            if servers_path.exists():
                self._servers_config = load_yaml(servers_path)
            else:
                self._servers_config = {"servers": {}, "default": "local"}
        return self._servers_config

    def _load_datasets(self) -> Dict:
        if self._datasets_config is None:
            datasets_path = self.config_dir / "datasets.yaml"
            if datasets_path.exists():
                self._datasets_config = load_yaml(datasets_path)
            else:
                self._datasets_config = {"base_dir": "", "benign": {}, "aliases": {}}
        return self._datasets_config

    def _load_sweeps(self) -> Dict:
        if self._sweeps_config is None:
            sweeps_path = self.config_dir / "sweeps.yaml"
            if sweeps_path.exists():
                self._sweeps_config = load_yaml(sweeps_path)
            else:
                self._sweeps_config = {"sweeps": {}, "default": "benchmark"}
        return self._sweeps_config

    def _resolve_server_ref(self, server_ref: str) -> Dict:
        servers_cfg = self._load_servers()
        servers = servers_cfg.get("servers", {})

        if server_ref not in servers:
            raise ValueError(
                f"Server configuration not found: {server_ref}. "
                f"Available servers: {list(servers.keys())}"
            )

        return servers[server_ref].copy()

    def _resolve_dataset_ref(self, dataset_ref: str) -> Dict:
        datasets_cfg = self._load_datasets()
        base_dir = datasets_cfg.get("base_dir", "")

        if "/" in dataset_ref or dataset_ref.startswith(("/", "~", ".")):
            return {
                "file": dataset_ref,
                "template": "plain",
            }

        parts = dataset_ref.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid dataset reference format: {dataset_ref}, expected 'category.name'")

        category, name = parts[0], parts[1]

        if category == "aliases":
            alias_target = datasets_cfg.get("aliases", {}).get(name)
            if alias_target:
                return self._resolve_dataset_ref(alias_target)
            raise ValueError(f"Unknown alias: {name}")

        category_cfg = datasets_cfg.get(category)
        if not category_cfg or not isinstance(category_cfg, dict):
            raise ValueError(f"Dataset category not found: {category}")

        dataset_cfg = category_cfg.get(name)
        if not dataset_cfg:
            available = list(category_cfg.keys()) if isinstance(category_cfg, dict) else []
            raise ValueError(
                f"Dataset not found: {category}.{name}. "
                f"Available datasets in this category: {available}"
            )

        if isinstance(dataset_cfg, list):
            return [
                self._resolve_dataset_ref(ref) if isinstance(ref, str) else ref
                for ref in dataset_cfg
            ]

        path = dataset_cfg.get("path", "")
        if base_dir and not path.startswith(("/", "~")):
            path = str(Path(base_dir) / path)

        return {
            "file": path,
            "template": dataset_cfg.get("template", "plain"),
            "description": dataset_cfg.get("description", ""),
            "tier": dataset_cfg.get("tier"),
            "max_prompts": dataset_cfg.get("max_prompts"),
        }

    def _resolve_extends(self, config: Dict, config_path: Path) -> Dict:
        extends = config.pop("extends", None)
        if not extends:
            return config

        parent_path = Path(extends).expanduser()
        if not parent_path.is_absolute():
            parent_path = (config_path.parent / extends).resolve()
            if not parent_path.exists() and self.config_dir:
                alt_path = (self.config_dir / extends).resolve()
                if alt_path.exists():
                    parent_path = alt_path
        if not parent_path.exists():
            raise FileNotFoundError(f"Extended config not found: {extends}")

        parent = load_yaml(parent_path)
        parent = self._resolve_extends(parent, parent_path)
        return deep_merge(parent, config)

    def _expand_config(self, config: Dict, config_path: Path) -> Dict:
        result = config.copy()

        server_ref = result.get("server")
        if server_ref and isinstance(server_ref, str):
            result["server"] = self._resolve_server_ref(server_ref)

        datasets = result.get("datasets", {})
        if datasets and isinstance(datasets, dict):
            for key, value in datasets.items():
                if isinstance(value, str):
                    result["datasets"][key] = self._resolve_dataset_ref(value)
                elif isinstance(value, list):
                    result["datasets"][key] = [
                        self._resolve_dataset_ref(v) if isinstance(v, str) else v
                        for v in value
                    ]
                elif isinstance(value, dict) and any(
                    tier in value for tier in ("high", "mid", "low")
                ):
                    resolved_tiers = value.copy()
                    for tier in ("high", "mid", "low"):
                        tier_value = value.get(tier)
                        if tier_value is None:
                            continue
                        if isinstance(tier_value, list):
                            resolved_tiers[tier] = [
                                self._resolve_dataset_ref(v) if isinstance(v, str) else v
                                for v in tier_value
                            ]
                        elif isinstance(tier_value, str):
                            resolved_tiers[tier] = [self._resolve_dataset_ref(tier_value)]
                        else:
                            resolved_tiers[tier] = tier_value
                    result["datasets"][key] = resolved_tiers
                elif isinstance(value, dict) and "path" in value:
                    result["datasets"][key] = value

        return result

    def _resolve_path(self, path: str) -> str:
        if not path:
            return path
        if path.startswith(("/", "~", ".")):
            return path
        datasets_cfg = self._load_datasets()
        base_dir = datasets_cfg.get("base_dir", "")
        return str(Path(base_dir) / path) if base_dir else path

    def load(self, config_name: str) -> Dict:
        config_path = (self.config_dir / config_name).with_suffix(".yaml")
        if not config_path.exists():
            config_path = (self.config_dir / config_name).with_suffix(".yml")

        if not config_path.exists():
            config_path = (self.config_dir / "experiments" / config_name).with_suffix(".yaml")
        if not config_path.exists():
            config_path = (self.config_dir / "experiments" / config_name).with_suffix(".yml")

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_name}")

        base = self._load_base()
        config = load_yaml(config_path)
        config = self._resolve_extends(config, config_path)
        config = deep_merge(base, config)
        config = self._expand_config(config, config_path)

        config["_meta"] = {
            "config_path": str(config_path),
            "config_dir": str(self.config_dir),
            "config_name": config_path.stem,
        }

        return config

    def load_sweep(self, sweep_name: Optional[str] = None) -> Dict:
        sweeps_cfg = self._load_sweeps()
        default_name = sweeps_cfg.get("default", "benchmark")
        name = sweep_name or default_name

        sweeps = sweeps_cfg.get("sweeps", {})
        if name not in sweeps:
            raise ValueError(
                f"Sweep configuration not found: {name}. "
                f"Available Sweeps: {list(sweeps.keys())}"
            )

        sweep = sweeps[name].copy()
        sweep["name"] = name

        if "payloads" in sweep:
            datasets_cfg = self._load_datasets()
            base_dir = datasets_cfg.get("base_dir", "")
            for payload in sweep["payloads"]:
                if "file" in payload and not payload["file"].startswith("/"):
                    payload["file"] = str(Path(base_dir) / payload["file"])

        return sweep

    def list_sweeps(self) -> Dict[str, Dict]:
        sweeps_cfg = self._load_sweeps()
        sweeps = {}
        for name, config in sweeps_cfg.get("sweeps", {}).items():
            sweeps[name] = {
                "enabled": config.get("enabled", False),
                "description": config.get("description", ""),
            }
        return sweeps


_loader: Optional[ConfigLoader] = None


def get_loader() -> ConfigLoader:
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader


def load_config(name: str) -> Dict:
    return get_loader().load(name)


def load_sweep(name: Optional[str] = None) -> Dict:
    return get_loader().load_sweep(name)


def list_sweeps() -> Dict[str, Dict]:
    return get_loader().list_sweeps()


def get_server(cfg: Dict) -> Dict:
    server = cfg.get("server")
    if isinstance(server, str):
        return get_loader()._resolve_server_ref(server)
    return server or {}


def get_experiment(cfg: Dict) -> Dict:
    return cfg.get("experiment", {})


def get_datasets(cfg: Dict) -> Dict:
    return cfg.get("datasets", {})


__all__ = [
    "ConfigLoader",
    "load_config",
    "load_sweep",
    "list_sweeps",
    "get_loader",
    "get_server",
    "get_experiment",
    "get_datasets",
    "deep_merge",
    "load_yaml",
]

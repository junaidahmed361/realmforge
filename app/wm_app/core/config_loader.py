from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_domain_config(domain_config_path: str | Path) -> dict:
    domain_cfg = load_yaml(domain_config_path)
    inherits = domain_cfg.get("inherits")
    if not inherits:
        return domain_cfg
    base_path = (Path(domain_config_path).parent / inherits).resolve()
    base_cfg = load_yaml(base_path)
    merged = {**base_cfg, **domain_cfg}
    if "backbone" in base_cfg and "backbone" in domain_cfg:
        merged["backbone"] = {**base_cfg["backbone"], **domain_cfg["backbone"]}
    return merged

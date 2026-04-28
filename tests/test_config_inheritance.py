from wm_app.core.config_loader import load_domain_config


def test_domain_inheritance_loads_backbone_fields():
    cfg = load_domain_config("domains/clinical_hf/configs/domain.yaml")
    assert "backbone" in cfg
    assert cfg["domain"] == "clinical_hf"

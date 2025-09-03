import yaml

def test_base_yaml_presets():
    with open("Dev/AlignTune/configs/base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["preset"] in {"fast", "balanced", "thorough"}
    for key in ["unsloth", "sft", "dpo"]:
        assert key in cfg
    assert isinstance(cfg["sft"]["max_seq_len"], dict)
    assert set(cfg["sft"]["max_seq_len"]).issuperset({"fast", "balanced", "thorough"})

def test_readme_mentions_train_pref():
    with open("Dev/AlignTune/README.md", "r", encoding="utf-8") as f:
        t = f.read()
    assert "scripts.train_pref" in t

from pathlib import Path

from caslr.config import load_config


def test_load_config_expands_sample_paths_and_experts():
    cfg = load_config(Path("configs/default.yaml"), sample=True)

    assert cfg.project_title == "Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models"
    assert cfg.method_name == "CASLR"
    assert cfg.data_files == [Path("data/sample/sample_router_data.csv")]
    assert "Qwen2.5-7B-Instruct" in cfg.experts

"""Archive #70-73/#140-151: freeze the production comparison workload."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pd32k_workload_is_frozen_against_same_parameter_native():
    acceptance = json.loads((ROOT / "configs/acceptance.json").read_text())
    workload = acceptance["production_workload"]
    assert workload == {
        "dataset_requests": 127,
        "mean_input_tokens": 32768,
        "batch_size": 32,
        "deployment_mode": "pd_fused",
        "comparison": "same_native_parameters",
        "required_oscar_route_hit_ratio": 1.0,
    }

from pathlib import Path

from tools import diag_ascendc_env


def test_distribution_version_does_not_import_package(monkeypatch):
    seen = []

    def version(name):
        seen.append(name)
        return "1.2.3"

    monkeypatch.setattr(diag_ascendc_env.metadata, "version", version)
    assert diag_ascendc_env._distribution_version("vllm-ascend") == "1.2.3"
    assert seen == ["vllm-ascend"]


def test_missing_distribution_is_reported(monkeypatch):
    def missing(_name):
        raise diag_ascendc_env.metadata.PackageNotFoundError

    monkeypatch.setattr(diag_ascendc_env.metadata, "version", missing)
    assert diag_ascendc_env._distribution_version("vllm-ascend") == "NOT INSTALLED"

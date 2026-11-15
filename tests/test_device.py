import subprocess
from types import SimpleNamespace

from era_agent.retrieval import embeddings


def smi(monkeypatch, out=None, error=None):
    def run(*_, **__):
        if error:
            raise error
        return SimpleNamespace(stdout=out)

    monkeypatch.setattr(embeddings.subprocess, "run", run)


def test_explicit_device_wins(monkeypatch):
    smi(monkeypatch, error=AssertionError("not called"))
    assert embeddings.pick_device("cpu") == "cpu"
    assert embeddings.pick_device("cuda") == "cuda"


def test_gpu_only_when_free(monkeypatch):
    smi(monkeypatch, "45000\n")  # vLLM stopped
    assert embeddings.pick_device(None) == "cuda"
    smi(monkeypatch, "660\n")  # vLLM at 97%
    assert embeddings.pick_device(None) == "cpu"


def test_no_gpu_tool_means_cpu(monkeypatch):
    smi(monkeypatch, error=FileNotFoundError())
    assert embeddings.pick_device(None) == "cpu"
    smi(monkeypatch, error=subprocess.TimeoutExpired("nvidia-smi", 10))
    assert embeddings.pick_device(None) == "cpu"


def test_threshold_from_config(monkeypatch):
    smi(monkeypatch, "8000\n")
    assert embeddings.pick_device(None, 6144) == "cuda"
    assert embeddings.pick_device(None, 10000) == "cpu"

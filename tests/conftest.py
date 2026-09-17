"""Bootstrap published inputs before collection; keep the offline suite on CPU."""
import os

# Set before test modules import tensor/model libraries, including when randomly
# changes collection order. Tests use tiny synthetic CPU models only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_sessionstart(session):
    from bootstrap_results import main

    if main() != 0:
        raise RuntimeError("Cannot bootstrap the published data_mirror/ inputs")

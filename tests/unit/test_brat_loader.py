import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.brat_loader import BRATLoader


@pytest.mark.skipif(
    not Path("/home/amin/Dropbox/code/SDOH/Track_2_SHAC/SHAC").exists(),
    reason="SHAC raw data directory not available"
)
def test_brat_loader_instantiation_and_scan_does_not_crash():
    data_root = Path("/home/amin/Dropbox/code/SDOH/Track_2_SHAC/SHAC")
    loader = BRATLoader(target_event="Drug")

    # Only scan a tiny subset if present to keep the test light
    splits = [s for s in ["train", "dev", "test"] if (data_root / s).exists()][:1] or ["train"]
    sources = [s for s in ["mimic", "uw"] if (data_root / splits[0] / s).exists()][:1] or ["mimic"]

    # The loader should run without throwing; content assertions are left to integration tests
    _ = loader.load_from_directory(data_root, sources=sources, splits=splits)

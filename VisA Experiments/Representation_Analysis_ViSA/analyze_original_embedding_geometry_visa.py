#!/usr/bin/env python3
"""VisA entry point for the shared corrected original-embedding analysis."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "protocols" / "visa"))
from visa_protocol import add_manifest_protocol_arguments, manifest_index, verify_test_episodes

HERE = Path(__file__).resolve().parent
engines = sorted(
    path for path in HERE.parents[1].glob(
        "*/Representation_Analysis_MVTec/analyze_original_embedding_geometry_mvtec.py"
    ) if path.is_file()
)
if len(engines) != 1:
    raise ImportError(f"Expected one sibling MVTec geometry engine, found {len(engines)}.")
MVTEC_ENGINE = engines[0]
VISA_ROOT = HERE.parent
VISA_BAYES_CODE = VISA_ROOT / "Bayesian_ProtoMAML"
VISA_PROTO_CODE = VISA_ROOT / "Classical_ProtoMAML"

spec = importlib.util.spec_from_file_location("shared_embedding_geometry_engine", MVTEC_ENGINE)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load shared analysis engine: {MVTEC_ENGINE}")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

# VisA run and source paths.
engine.HERE = HERE
engine.PROJECT = VISA_ROOT
engine.BAYES_CODE = VISA_BAYES_CODE
engine.PROTO_CODE = VISA_PROTO_CODE
engine.BAYES_ROOT = (VISA_BAYES_CODE / "runs" /
    "baypranometa_visa_three_seed_fixed_split42_25ep_30epochs_100test" /
    "revised_clean_official_2cls_highshot")
engine.PROTO_ROOT = (VISA_PROTO_CODE / "runs" /
    "protomaml_visa_corrected_avgpool_three_seed" /
    "revised_clean_official_2cls_highshot")
engine.OUTPUT_ROOT = HERE / "results"
engine.EXPECTED_CATEGORY_COUNT = 12
engine.SMOKE_CATEGORY = "candle"

# Bind the VisA posterior, log density, and evaluation transform.
for module_name in list(sys.modules):
    if module_name == "utils" or module_name.startswith("utils."):
        del sys.modules[module_name]
sys.path.insert(0, str(VISA_BAYES_CODE))
from utils.utils_niw import log_student_t, niw_posterior
from utils.utils_data_visa_leave1out import transform_eval

engine.log_student_t = log_student_t
engine.niw_posterior = niw_posterior
engine.TRANSFORM = transform_eval


def configure_run(args, parser):
    protocol_index = manifest_index(args, parser)
    load_manifest = engine.load_manifest

    def load_verified_manifest(seed, category):
        episodes = load_manifest(seed, category)
        verify_test_episodes(episodes, category, protocol_index)
        return episodes

    engine.load_manifest = load_verified_manifest


if __name__ == "__main__":
    engine.main(configure_parser=add_manifest_protocol_arguments, configure_run=configure_run)

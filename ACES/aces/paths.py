from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ACES_ROOT = REPO_ROOT / "ACES"

VALID_ROOT = REPO_ROOT / "Methods" / "VALID-v1.0" / "FDU-donglab-VALID-34f5637"
SN2N_ROOT = Path("/data2/wjb/SN2N-main")
FAST_ROOT = REPO_ROOT / "Methods" / "FAST-main"

ORIGINAL_WORKFLOW_ROOT = (
    FAST_ROOT
    / "reproduction"
    / "valid_comparison"
    / "experiments"
    / "valid_paper_3p_neuron"
    / "original_workflow_paper_params_repeat01_seed3407"
)
REPEATS_ROOT = ORIGINAL_WORKFLOW_ROOT / "data" / "all_repeats_275z"
MEAN9_PROXY = ORIGINAL_WORKFLOW_ROOT / "data" / "gt_proxy" / "3P_neuron_mean9_gt_proxy_275z.tif"

DEFAULT_RUN_DIR = ACES_ROOT / "runs" / "stage01_3p_repeat_pairability_seed3407"
THREEPM_PYTHON = Path("/data2/wjb/anaconda3/envs/threePM/bin/python")

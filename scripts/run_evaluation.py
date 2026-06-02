"""
Script entry point for evaluation in the unified reservoir operation framework.

This script runs:
1) DP evaluation,
2) SDP evaluation,
3) ISO-SURF evaluation, and
4) DQN evaluation.

The purpose of this script is to keep executable evaluation workflows separate
from reusable library modules.
"""

from reservoir_benchmark.evaluation.ddp.ddp_eval import main_ddp_eval
from reservoir_benchmark.evaluation.sdp.sdp_eval import main_sdp_eval
from reservoir_benchmark.evaluation.iso_surf.iso_surf_eval import main_iso_surf_eval
from reservoir_benchmark.evaluation.dqn.dqn_eval import main_dqn_eval


def main() -> None:
    try:
        print("=" * 80)
        print("Running DDP evaluation...")
        print("=" * 80)
        main_ddp_eval()

        print("\n" + "=" * 80)
        print("Running SDP evaluation...")
        print("=" * 80)
        main_sdp_eval()

        print("\n" + "=" * 80)
        print("Running ISO-SURF evaluation...")
        print("=" * 80)
        main_iso_surf_eval()

        print("\n" + "=" * 80)
        print("Running DQN evaluation...")
        print("=" * 80)
        main_dqn_eval()

        print("\n" + "=" * 80)
        print("All evaluation tasks completed.")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap dependencies for the unified reservoir operation framework in the
CURRENT Conda environment.

This script does NOT create a .venv by default. It installs/checks dependencies
against the Python interpreter that is used to run this script. Therefore, users
should first activate the desired Conda environment and then run this script.

Recommended usage:
    conda create -n reservoir_benchmark python=3.11 -y
    conda activate reservoir_benchmark
    python bootstrap_conda_environment_setup.py

Dry run:
    python bootstrap_conda_environment_setup.py --dry-run

If you intentionally want to use base or a non-Conda Python, pass:
    --allow-base
    --allow-system-python
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def shlex_join(cmd: list[str]) -> str:
    try:
        import shlex
        return shlex.join(cmd)
    except Exception:
        return " ".join(cmd)


def run(cmd: list[str], *, check: bool = True) -> int:
    print(f"+ {shlex_join(cmd)}")
    process = subprocess.run(cmd)
    if check and process.returncode != 0:
        raise SystemExit(process.returncode)
    return process.returncode


def detect_os() -> str:
    return platform.system()


def ensure_python_version(min_major: int, min_minor: int) -> None:
    if sys.version_info < (min_major, min_minor):
        raise SystemExit(
            f"Python version is too old: current "
            f"{sys.version_info.major}.{sys.version_info.minor}, "
            f"required >= {min_major}.{min_minor}.\n"
            f"Please create a Conda environment with Python {min_major}.{min_minor}+."
        )


def get_nvidia_smi_output() -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        return subprocess.check_output(
            [executable],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return None


def parse_cuda_version_from_nvidia_smi(text: str) -> tuple[int, int] | None:
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_environment_profile() -> tuple[str, tuple[int, int] | None]:
    os_name = detect_os()
    if os_name == "Windows":
        nvidia_smi_output = get_nvidia_smi_output()
        cuda_version = parse_cuda_version_from_nvidia_smi(nvidia_smi_output or "")
        if nvidia_smi_output and cuda_version:
            return "windows_nvidia", cuda_version
        return "windows_cpu", None
    if os_name == "Darwin":
        return "macos", None
    return "linux_or_other", None


def pick_torch_index_for_cuda(cuda_version: tuple[int, int]) -> str | None:
    """
    Select a PyTorch wheel index URL based on the NVIDIA driver CUDA version.
    The index is selected conservatively from commonly available PyTorch wheels.
    """
    major, minor = cuda_version
    if (major, minor) >= (12, 4):
        return "https://download.pytorch.org/whl/cu124"
    if (major, minor) >= (12, 1):
        return "https://download.pytorch.org/whl/cu121"
    if (major, minor) >= (11, 8):
        return "https://download.pytorch.org/whl/cu118"
    return None


def detect_conda_environment() -> tuple[bool, str | None, Path | None]:
    """
    Detect whether the current Python interpreter belongs to a Conda environment.

    Returns:
        is_conda: True if a Conda environment is detected.
        env_name: Conda environment name if available.
        env_prefix: Conda environment prefix if available.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")

    if conda_prefix:
        return True, conda_env_name, Path(conda_prefix)

    # This also works when the script is launched directly with
    # D:\\...\\envs\\name\\python.exe without activating the environment.
    prefix = Path(sys.prefix)
    if (prefix / "conda-meta").exists():
        return True, prefix.name, prefix

    return False, None, None


def package_is_installed(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def maybe_run(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] {shlex_join(cmd)}")
    else:
        run(cmd)


def install_if_needed(
    pip_cmd: list[str],
    package_name: str,
    import_name: str | None = None,
    *,
    upgrade: bool = False,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    import_name = import_name or package_name.split("[")[0].replace("-", "_")
    extra_args = extra_args or []

    if package_is_installed(import_name) and not upgrade:
        print(f"Requirement already available: {package_name} ({import_name}). Skipping.")
        return

    cmd = pip_cmd + ["install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package_name)
    cmd += extra_args
    maybe_run(cmd, dry_run)


def build_torch_install_args(
    environment_profile: str,
    cuda_version: tuple[int, int] | None,
    disable_torch_gpu_auto: bool,
) -> list[str]:
    if (
        environment_profile == "windows_nvidia"
        and cuda_version is not None
        and not disable_torch_gpu_auto
    ):
        index_url = pick_torch_index_for_cuda(cuda_version)
        if index_url is not None:
            print(
                f"Detected Windows + NVIDIA GPU. CUDA version reported by driver: "
                f"{cuda_version[0]}.{cuda_version[1]}. Using PyTorch index: {index_url}"
            )
            return ["--index-url", index_url]

        print(
            f"Detected Windows + NVIDIA GPU with CUDA {cuda_version[0]}.{cuda_version[1]}, "
            "but no matching PyTorch CUDA wheel channel was selected. "
            "Falling back to default torch."
        )

    elif environment_profile == "windows_nvidia" and disable_torch_gpu_auto:
        print("Detected Windows + NVIDIA GPU, but automatic CUDA torch selection is disabled.")
    elif environment_profile == "windows_cpu":
        print("Detected Windows without available NVIDIA/CUDA information. Installing default torch if needed.")
    elif environment_profile == "macos":
        print("Detected macOS. Installing default torch if needed. Apple Silicon may use the MPS backend.")
    else:
        print("Detected Linux or another platform. Installing default torch if needed.")

    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install/check project dependencies in the current Conda environment."
    )
    parser.add_argument(
        "--gym-extras",
        default="classic-control",
        help="Gymnasium extra to install. Default: classic-control. Use empty string for plain gymnasium.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade packages even if they are already installed. Default: only install missing packages.",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip PyTorch installation/check.",
    )
    parser.add_argument(
        "--no-torch-gpu-auto",
        action="store_true",
        help="Disable automatic CUDA wheel selection for Windows + NVIDIA.",
    )
    parser.add_argument(
        "--no-project-install",
        action="store_true",
        help="Do not install the current project in editable mode.",
    )
    parser.add_argument(
        "--allow-base",
        action="store_true",
        help="Allow installation into Conda base environment. Not recommended.",
    )
    parser.add_argument(
        "--allow-system-python",
        action="store_true",
        help="Allow installation when the current Python is not detected as a Conda environment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    ensure_python_version(3, 10)

    is_conda, conda_env_name, conda_prefix = detect_conda_environment()
    os_name = detect_os()
    environment_profile, cuda_version = detect_environment_profile()

    print("=" * 80)
    print("Conda environment bootstrap started")
    print("=" * 80)
    print(f"Python executable: {sys.executable}")
    print(f"Python prefix: {sys.prefix}")
    print(f"Detected OS: {os_name}")
    print(f"Detected environment profile: {environment_profile}")
    print(f"Detected Conda environment: {conda_env_name or 'N/A'}")
    print(f"Detected Conda prefix: {conda_prefix or 'N/A'}")
    if cuda_version is not None:
        print(f"Detected CUDA version from nvidia-smi: {cuda_version[0]}.{cuda_version[1]}")
    print("=" * 80)

    if not is_conda and not args.allow_system_python:
        raise SystemExit(
            "No Conda environment was detected for the current Python interpreter.\n"
            "Please create and activate a dedicated Conda environment first, for example:\n\n"
            "    conda create -n reservoir_benchmark python=3.11 -y\n"
            "    conda activate reservoir_benchmark\n"
            "    python bootstrap_conda_environment_setup.py\n\n"
            "If you really want to install into the current non-Conda Python, rerun with --allow-system-python."
        )

    if (conda_env_name == "base" or (conda_prefix and conda_prefix.name == "base")) and not args.allow_base:
        raise SystemExit(
            "The current environment appears to be Conda base. Installing project dependencies into base is not recommended.\n"
            "Create a dedicated environment instead:\n\n"
            "    conda create -n reservoir_benchmark python=3.11 -y\n"
            "    conda activate reservoir_benchmark\n"
            "    python bootstrap_conda_environment_setup.py\n\n"
            "If you intentionally want to modify base, rerun with --allow-base."
        )

    pip_cmd = [sys.executable, "-m", "pip"]

    # Keep packaging tools healthy. This is usually safe, but still controlled by --upgrade.
    install_if_needed(pip_cmd, "pip", "pip", upgrade=args.upgrade, dry_run=args.dry_run)
    install_if_needed(pip_cmd, "setuptools", "setuptools", upgrade=args.upgrade, dry_run=args.dry_run)
    install_if_needed(pip_cmd, "wheel", "wheel", upgrade=args.upgrade, dry_run=args.dry_run)

    dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("openpyxl", "openpyxl"),
        ("pyyaml", "yaml"),
    ]
    for package_name, import_name in dependencies:
        install_if_needed(
            pip_cmd,
            package_name,
            import_name,
            upgrade=args.upgrade,
            dry_run=args.dry_run,
        )

    gym_pkg = f"gymnasium[{args.gym_extras}]" if args.gym_extras else "gymnasium"
    install_if_needed(
        pip_cmd,
        gym_pkg,
        "gymnasium",
        upgrade=args.upgrade,
        dry_run=args.dry_run,
    )

    if not args.skip_torch:
        torch_extra_args = build_torch_install_args(
            environment_profile=environment_profile,
            cuda_version=cuda_version,
            disable_torch_gpu_auto=args.no_torch_gpu_auto,
        )
        install_if_needed(
            pip_cmd,
            "torch",
            "torch",
            upgrade=args.upgrade,
            dry_run=args.dry_run,
            extra_args=torch_extra_args,
        )
    else:
        print("PyTorch installation/check skipped by --skip-torch.")

    if not args.no_project_install:
        project_root = Path.cwd()
        if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
            maybe_run(pip_cmd + ["install", "-e", "."], args.dry_run)
        else:
            print("No pyproject.toml or setup.py found in current directory. Skipping editable project install.")
    else:
        print("Project editable installation skipped by --no-project-install.")

    print("\n" + "=" * 80)
    print("Conda environment bootstrap completed")
    print("=" * 80)
    print("Quick check:")
    print(
        "python -c \"import sys; print(sys.executable); "
        "import torch; print('torch', torch.__version__); "
        "print('cuda', torch.cuda.is_available())\""
    )


if __name__ == "__main__":
    main()

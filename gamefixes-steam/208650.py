"""Game fix for Batman: Arkham Knight — enable PhysX GPU via ZLUDA"""

import os
from pathlib import Path
from protonfixes import util


def load_file_overrides() -> dict[str, str]:
    """Read optional per-launch investigation overrides from 208650.env."""
    path = Path(__file__).with_suffix('.env')
    if not path.exists():
        return {}
    overrides = {}
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        key = key.strip()
        if not sep or not key:
            raise ValueError(f'Invalid override line {line_no} in {path}: {raw_line!r}')
        overrides[key] = value
    return overrides


FILE_OVERRIDES = load_file_overrides()


def set_default_environment(envvar: str, value: str) -> None:
    """Allow shell-provided investigation overrides while keeping defaults."""
    util.set_environment(envvar, FILE_OVERRIDES.get(envvar, os.environ.get(envvar, value)))


def main() -> None:
    """Enable ZLUDA CUDA-on-HIP translation for PhysX GPU effects"""
    # Force NVAPI so PhysX detects a CUDA-capable GPU through dxvk-nvapi
    util.set_environment('PROTON_FORCE_NVAPI', '1')
    util.set_environment('PROTON_ENABLE_ZLUDA', '1')
    util.set_environment('DXVK_NVAPI_ALLOW_OTHER_DRIVERS', '1')
    util.set_environment('DXVK_ENABLE_NVAPI', '1')
    # AMD Van Gogh (Steam Deck) → override to gfx1030 for ROCm compat
    util.set_environment('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
    # Disable SDMA on Steam Deck / gfx10.3: ROCm module+copy probes and live
    # Batman runs can hang after SDMA queue exhaustion, while HSA_ENABLE_SDMA=0
    # restores stable blit-kernel copies.
    util.set_environment('HSA_ENABLE_SDMA', '0')
    # Use Wine's builtin nvcuda.dll stub → nvcuda.dll.so relay → ZLUDA libcuda.so
    util.winedll_override('nvcuda', 'b')

    # Debug logging — always on during investigation
    util.set_environment('ZLUDA_DEBUG', '1')
    # Investigate the corrected dedicated launch mutex path in the live Proton runtime.
    set_default_environment('ZLUDA_LAUNCH_MUTEX', '2')
    if True:
        set_default_environment('ZLUDA_LOG_ALL', '0')
        set_default_environment('ZLUDA_LOG_LAUNCHES', '1')
        set_default_environment('ZLUDA_VALIDATE_SORT', '0')
        # Keep the default live Batman branch on the lower-noise later-family path.
        # When CreateGrid validation/tracing is the subject of a specific A/B, use
        # the sibling 208650.env override file instead of keeping these on by default.
        set_default_environment('ZLUDA_VALIDATE_SORT_SAFE', '0')
        set_default_environment('ZLUDA_TRACE_SORT_COPY', '0')
        set_default_environment('ZLUDA_CREATEGRID_TOUCH_LOG', '0')
        set_default_environment('ZLUDA_CREATEGRID_SNAPSHOT_LOG', '0')
        set_default_environment('ZLUDA_CREATECELL_SNAPSHOT_LOG', '0')
        set_default_environment('ZLUDA_SORT_REPLAY_CAPTURE_BIT', '0')
        set_default_environment('ZLUDA_MS4_CAPTURE', '1')
        set_default_environment('ZLUDA_TFS_CAPTURE', '1')
        set_default_environment('ZLUDA_IOFX_CAPTURE', '0')
        set_default_environment('ZLUDA_IOFX_BBOX_CAPTURE', '0')
        set_default_environment('ZLUDA_DOWNSWEEP_CAPTURE', '1')
        # Phase 5C diagnostic: comma-separated kernel-name substrings to SKIP
        # (launch returns OK without actually dispatching). Used for fault-
        # attribution bisection. Default empty = no skipping.
        set_default_environment('ZLUDA_SKIP_KERNELS', '')
        set_default_environment('ZLUDA_COUNTER_SNAPSHOT', '1')
        set_default_environment('ZLUDA_SORT_ZERO_INPUT_SLACK', '1')
        # Investigation gate: strict NULL-stream semantics removed the older
        # n=8827 sort failure and exposed a later n=4688 CreateGrid family.
        set_default_environment('ZLUDA_NULL_STREAM_STRICT', '1')
        # Opt-in: extended WAR barriers for PhysX sort kernels that reshuffle
        # data entirely within LDS (IOFX_newRadixSortStepKernel and friends).
        # Default 0; flip to 1 via 208650.env to test.
        set_default_environment('ZLUDA_PHYSX_WAR_BARRIERS', '0')
        set_default_environment('PROTON_LOG', '0')
        set_default_environment('DXVK_LOG_LEVEL', 'warn')
        set_default_environment('WINEDEBUG', '-all')
        # HIP runtime logging OFF — AMD_LOG_LEVEL>0 triggers ProfilingSignal race
        set_default_environment('AMD_LOG_LEVEL', '0')

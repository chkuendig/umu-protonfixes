"""Game fix for Batman: Arkham Knight — enable PhysX GPU via ZLUDA"""

import os
from protonfixes import util


def main() -> None:
    """Enable ZLUDA CUDA-on-HIP translation for PhysX GPU effects"""
    # Force NVAPI so PhysX detects a CUDA-capable GPU through dxvk-nvapi
    util.set_environment('PROTON_FORCE_NVAPI', '1')
    util.set_environment('PROTON_ENABLE_ZLUDA', '1')
    util.set_environment('DXVK_NVAPI_ALLOW_OTHER_DRIVERS', '1')
    util.set_environment('DXVK_ENABLE_NVAPI', '1')
    # AMD Van Gogh (Steam Deck) → override to gfx1030 for ROCm compat
    util.set_environment('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
    # Disable SDMA to prevent DMA engine conflicts during VRAM eviction
    util.set_environment('HSA_ENABLE_SDMA', '0')
    # Use Wine's builtin nvcuda.dll stub → nvcuda.dll.so relay → ZLUDA libcuda.so
    util.winedll_override('nvcuda', 'b')

    # Debug logging — always on during investigation
    util.set_environment('ZLUDA_DEBUG', '1')
    # Investigate the corrected dedicated launch mutex path in the live Proton runtime.
    util.set_environment('ZLUDA_LAUNCH_MUTEX', '2')
    if True:
        util.set_environment('ZLUDA_LOG_ALL', '0')
        util.set_environment('ZLUDA_LOG_LAUNCHES', '1')
        util.set_environment('ZLUDA_VALIDATE_SORT', '0')
        util.set_environment('ZLUDA_VALIDATE_SORT_SAFE', '1')
        util.set_environment('ZLUDA_TRACE_SORT_COPY', '1')
        util.set_environment('ZLUDA_MS4_CAPTURE', '1')
        util.set_environment('ZLUDA_SORT_ZERO_INPUT_SLACK', '1')
        util.set_environment('PROTON_LOG', '0')
        util.set_environment('DXVK_LOG_LEVEL', 'warn')
        util.set_environment('WINEDEBUG', '-all')
        # HIP runtime logging OFF — AMD_LOG_LEVEL>0 triggers ProfilingSignal race
        util.set_environment('AMD_LOG_LEVEL', '0')

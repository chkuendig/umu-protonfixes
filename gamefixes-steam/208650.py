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
    # Use Wine's builtin nvcuda.dll stub → nvcuda.dll.so relay → ZLUDA libcuda.so
    util.winedll_override('nvcuda', 'b')

    # Optional debug logging (set ZLUDA_DEBUG=1 in Steam launch options to enable)
    if os.environ.get('ZLUDA_DEBUG'):
        util.set_environment('PROTON_LOG', '1')
        util.set_environment('DXVK_LOG_LEVEL', 'info')
        util.set_environment('WINEDEBUG', '+loaddll,+nvcuda')

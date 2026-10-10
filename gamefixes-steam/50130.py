"""Experimental Mafia II Classic setup for the separate WoW64 tool."""
from pathlib import Path
import os
from protonfixes import util


def cpu_control():
    return os.environ.get('ZLUDA_WOW64_CPU_CONTROL') == '1'


def early():
    # GE11 derives WINEARCH before the normal game hook runs.
    util.set_environment('PROTON_USE_WOW64', '1')
    if cpu_control():
        # The fork's ZLUDA launcher tests presence, not the value "0".
        util.del_environment('PROTON_ENABLE_ZLUDA')
        util.del_environment('PROTON_FORCE_NVAPI')
    else:
        util.set_environment('PROTON_ENABLE_ZLUDA', '1')
        util.set_environment('PROTON_FORCE_NVAPI', '1')
    util.set_environment('DXVK_NVAPI_ALLOW_OTHER_DRIVERS', '1')
    util.set_environment('DXVK_ENABLE_NVAPI', '0' if cpu_control() else '1')


def main():
    tool = Path(__file__).resolve().parents[2]
    util.set_environment('ZLUDA_WOW64_LIBRARY', str(tool / 'files/lib64/libcuda.so'))
    util.set_environment('WINEDLLOVERRIDES', 'nvcuda=b;nvapi,dxgi,d3d9,d3d11=n,b;PhysXLoader=n,b')
    env_file = Path(__file__).with_suffix('.env')  # absent in a normal install
    preserve = os.environ.get('ZLUDA_WOW64_ENV_PRESERVE') == '1'
    for raw in env_file.read_text().splitlines() if env_file.exists() else ():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.split('=', 1)
        # Diagnostic runs may pre-set a switch; keep it when asked.
        if preserve and key in os.environ:
            continue
        util.set_environment(key, value)
    if cpu_control():
        # Loading diagnostic: same WoW64 setup with CUDA and NVAPI unavailable,
        # so PhysX falls back to the CPU.
        for key in ('PROTON_ENABLE_ZLUDA', 'PROTON_FORCE_NVAPI'):
            util.del_environment(key)
        for key in ('PROTON_ENABLE_NVAPI', 'DXVK_ENABLE_NVAPI'):
            util.set_environment(key, '0')
        util.set_environment('WINEDLLOVERRIDES',
                             'nvcuda=;nvapi,nvapi64=;dxgi,d3d9,d3d11=n,b;PhysXLoader=n,b')
    # The shipped executable imports d3dx9_42; use the complete D3DX9 verb.
    # Keep PhysX version management in the existing per-game helper.
    for verb in ('d3dx9', 'physx'):
        if not util.checkinstalled(verb) and not util.protontricks(verb):
            raise RuntimeError(f'{verb} installation failed; inspect Protonfixes log')

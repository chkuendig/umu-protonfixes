"""Game fix for Batman: Arkham Knight — enable PhysX GPU via ZLUDA"""

import os
from pathlib import Path
from protonfixes import util


def load_file_overrides() -> dict[str, str]:
    """Read optional per-launch investigation overrides from 208650.env.

    deploy_batman.sh writes this file from its shell environment so we can A/B
    ZLUDA env vars during debugging without editing this fix. It is absent in a
    normal install, so shipped behaviour is just the defaults below.
    """
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
    """Set envvar to the .env override, else the shell value, else the default."""
    util.set_environment(envvar, FILE_OVERRIDES.get(envvar, os.environ.get(envvar, value)))


def main() -> None:
    """Enable ZLUDA CUDA-on-HIP translation for PhysX GPU effects.

    PROTON_ENABLE_ZLUDA=1 triggers Proton's central ZLUDA block, which routes
    nvcuda.dll -> ZLUDA, sets the NVAPI trio (so PhysX detects an NVIDIA GPU),
    the ZLUDA lib dir + wine stack fix, and the AMD device env (the HSA gfx
    override is auto-detected per GPU). Kernel cache and trace logging use lib
    defaults (cache -> steamapps/shadercache/<appid>/ZLUDA, logging off).
    """
    util.set_environment('PROTON_ENABLE_ZLUDA', '1')

    # Investigation trace log + PTX dump: OFF by default. deploy_batman's .env
    # sets ZLUDA_LOG=1 for debug runs (the run-assessment tooling reads tex.log).
    # The lib defaults the log path to /tmp/zluda_tex.log (dxvk/wine convention);
    # no ZLUDA_LOG_PATH/ZLUDA_LOG_DIR override needed here.
    set_default_environment('ZLUDA_LOG', '0')
    set_default_environment('ZLUDA_LOG_ALL', '0')
    set_default_environment('ZLUDA_LOG_LAUNCHES', '0')
    set_default_environment('ZLUDA_DTOH_LOG', '0')
    set_default_environment('ZLUDA_DEBUG', '0')
    set_default_environment('ZLUDA_DUMP_PTX', '0')

    # PhysX workarounds not (yet) baked into lib defaults; overridable via .env
    # for A/B. The primary crash fixes (cross-kernel system-scope coherence,
    # bar.sync fence, VGPR-zero prolog, dirty-flag null-stream sync, texobj sync,
    # HDP flush) are all lib-default-on and need nothing here.
    set_default_environment('ZLUDA_BROAD_MEMBAR', '1')
    set_default_environment('ZLUDA_IOFX_ACTOR_MEMBAR', '1')
    set_default_environment('ZLUDA_LAUNCH_MUTEX', '2')
    # NULL-stream strict: closes a CUDA-spec gap (HIP's null-stream doesn't
    # serialise like CUDA's). Source default OFF; previous protonfix had it
    # at 1 and that was the validated state on the 9-26 baseline.
    set_default_environment('ZLUDA_NULL_STREAM_STRICT', '1')

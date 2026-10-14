"""Game fix for Mirror's Edge: GPU PhysX through ZLUDA"""

import shutil
from pathlib import Path

from protonfixes import util
from protonfixes.logger import log

# The game ships PhysX 2.8.0.7 next to its executable. That PhysXCore has
# no GPU code (PTX), so PhysX initialises CUDA and silently falls back to
# the CPU. Moved aside, the game loads the PhysX System Software in the
# prefix, whose PhysXCore 2.8.0.46 has it.
BUNDLED_PHYSX = ('PhysXCore.dll', 'PhysXDevice.dll', 'PhysXLocal')


def early() -> None:
    # 32-bit: CUDA reaches ZLUDA through nvcuda's WoW64 bridge, which needs
    # Wine's new WoW64 mode. Proton reads this before the prefix is set up.
    util.set_environment('PROTON_USE_WOW64', '1')


def main() -> None:
    util.set_environment('PROTON_ENABLE_ZLUDA', '1')
    for verb in ('d3dx9', 'd3dcompiler_43', 'physx'):
        if not util.checkinstalled(verb) and not util.protontricks(verb):
            raise RuntimeError(f'{verb} installation failed; inspect Protonfixes log')
    # Only once the system runtime is installed. Steam's file verification
    # restores the bundled copies; the next launch moves them aside again.
    binaries = Path(util.get_game_install_path()) / 'Binaries'
    for name in BUNDLED_PHYSX:
        bundled, moved = binaries / name, binaries / f'{name}.bak'
        if not bundled.exists():
            continue
        if moved.is_dir():
            shutil.rmtree(moved)
        elif moved.exists():
            moved.unlink()
        bundled.rename(moved)
        log.info(f'Moved bundled {name} aside so the system PhysX runtime loads')

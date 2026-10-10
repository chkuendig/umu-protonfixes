"""Game fix for Batman: Arkham Asylum GOTY: GPU PhysX through ZLUDA"""

from protonfixes import util


def early() -> None:
    # 32-bit: CUDA reaches ZLUDA through nvcuda's WoW64 bridge, which needs
    # Wine's new WoW64 mode. Proton reads this before the prefix is set up.
    util.set_environment('PROTON_USE_WOW64', '1')


def main() -> None:
    util.set_environment('PROTON_ENABLE_ZLUDA', '1')
    # Unreal Engine 3 needs native D3DX9 and its shader compiler; PhysX 2.8
    # loads from the PhysX System Software.
    for verb in ('d3dx9', 'd3dcompiler_43', 'physx'):
        if not util.checkinstalled(verb) and not util.protontricks(verb):
            raise RuntimeError(f'{verb} installation failed; inspect Protonfixes log')

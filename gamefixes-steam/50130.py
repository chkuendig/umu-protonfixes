"""Game fix for Mafia II (Classic): GPU PhysX through ZLUDA"""

from protonfixes import util


def early() -> None:
    # 32-bit: CUDA reaches ZLUDA through nvcuda's WoW64 bridge, which needs
    # Wine's new WoW64 mode. Proton reads this before the prefix is set up.
    # Wine's own WINESTEAMNOEXEC=1 for this CEG title stays: with 0, Wine
    # writes jumps into the game's steamclient.dll code instead.
    util.set_environment('PROTON_USE_WOW64', '1')


def main() -> None:
    util.set_environment('PROTON_ENABLE_ZLUDA', '1')
    # The executable imports d3dx9_42. Steam installs the PhysX System
    # Software: the game's 3rd/PhysX_9.10.0513 and the Steamworks common
    # redistributable 9.14.0702, which wins.
    if not util.checkinstalled('d3dx9') and not util.protontricks('d3dx9'):
        raise RuntimeError('d3dx9 installation failed; inspect Protonfixes log')

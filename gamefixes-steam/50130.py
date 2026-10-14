"""Game fix for Mafia II (Classic): GPU PhysX through ZLUDA"""

import os

from protonfixes import util


def early() -> None:
    # 32-bit: CUDA reaches ZLUDA through nvcuda's WoW64 bridge, which needs
    # Wine's new WoW64 mode. Proton reads this before the prefix is set up.
    util.set_environment('PROTON_USE_WOW64', '1')
    # Wine sets WINESTEAMNOEXEC=1 for this game (CEG). Its steamclient fault
    # redirect then returns without restoring the 32-bit FS selector, which
    # crashes under new WoW64. Wine only sets it when unset, so this wins.
    if 'WINESTEAMNOEXEC' not in os.environ:
        util.set_environment('WINESTEAMNOEXEC', '0')


def main() -> None:
    util.set_environment('PROTON_ENABLE_ZLUDA', '1')
    # The executable imports d3dx9_42. Steam's installscript installs the
    # game's own PhysX System Software (3rd/PhysX_9.10.0513).
    if not util.checkinstalled('d3dx9') and not util.protontricks('d3dx9'):
        raise RuntimeError('d3dx9 installation failed; inspect Protonfixes log')

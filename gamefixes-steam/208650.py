"""Game fix for Batman Arkham Knight"""

from protonfixes import util


def main() -> None:
    """NVIDIA PhysX support."""
    # Enables NVIDIA PhysX in Batman Arkham Knight.
    util.protontricks('physx')
    # GPU PhysX through ZLUDA: Proton's ZLUDA block routes nvcuda to ZLUDA
    # and makes NVAPI report a CUDA-capable GPU, so PhysX picks its GPU path.
    util.set_environment('PROTON_ENABLE_ZLUDA', '1')

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
        # Phase H-step 2: fence-around-Downsweep experiment. When set to '1',
        # wraps every RadixSortDownsweepKernel launch in pre- and post-launch
        # hipDeviceSynchronize + hipPeekAtLastError so we can decide whether
        # the reported Downsweep fault is actually upstream (pre-launch peek
        # returns non-OK) or localized to the Downsweep kernel itself.
        set_default_environment('ZLUDA_DOWNSWEEP_FENCE', '0')
        # Phase H-step 1: canary memory guard. When '1', allocates 8 × 1MB
        # device canary regions filled with 0xDEADBEEF at first Downsweep
        # launch; scans them pre-Downsweep for wild-write detection.
        set_default_environment('ZLUDA_DOWNSWEEP_CANARY', '0')
        # LDS-zero prelude (2026-04-24): patches every RadixSortDownsweepKernel
        # PTX at load time to emit a cooperative LDS-zero loop + bar.sync at
        # kernel entry. Tests whether residue-driven uninit-LDS reads produce
        # the observed TCP-write fault at values_a - <varying negative offset>.
        # Default 0; flip to 1 via 208650.env or host shell.
        set_default_environment('ZLUDA_DOWNSWEEP_LDS_ZERO', '0')
        # Force HW initialization of v1/v2 (workitem_id_y/z) VGPRs at
        # Downsweep kernel entry. Block=(128,1,1) means y=z=0 always, but
        # the compiler DCE's the tid.y/z reads → kernel descriptor's
        # enable_vgpr_workitem_id=0 → HW does NOT zero v1,v2 → Downsweep
        # prolog spills garbage residue to scratch → reloaded into a
        # scatter address → wild TCP-write at values_a - <negative>.
        # This patch emits volatile tid.y/z reads so the backend flips
        # the enable bits. Default 0.
        set_default_environment('ZLUDA_FORCE_WORKITEM_ID_YZ', '0')
        # Downsweep scatter-store instrumentation (2026-04-24). Allocates a
        # pinned host-mapped witness buffer and patches every `st.global.u32
        # [%rdN], %rM;` scatter inside RadixSortDownsweepKernel to write
        # (addr_lo, addr_hi, data, slot_idx) to its per-lane slot BEFORE the
        # original store. On SIGABRT the buffer survives (pinned host mem);
        # inspect via ZLUDA's crash dump or DtoH on the host side. Default 0.
        set_default_environment('ZLUDA_DOWNSWEEP_INSTRUMENT', '0')
        # Instrumentation window: skip the first N matching stores, then cap
        # the next M. Default M=128, N=0 covers sites [0,128). To window the
        # LATER stores, export SKIP=80 MAX_SITES=128 → sites [80, 208).
        set_default_environment('ZLUDA_DOWNSWEEP_INSTRUMENT_MAX_SITES', '128')
        set_default_environment('ZLUDA_DOWNSWEEP_INSTRUMENT_SKIP', '0')
        # Diagnostic: force wave64 at codegen. Sets target-feature
        # -wavefrontsize32,+wavefrontsize64 on each kernel. Causes LLVM
        # SelectionDAG abort on Downsweep as of 2026-04-24, so currently
        # useful only for probing simpler kernels. Default 0.
        set_default_environment('ZLUDA_WAVE64', '0')
        # Number of extra VGPRs to force-zero at Downsweep kernel entry via
        # the yz-vgpr-init patch. The patch allocates N u32 PTX registers,
        # inits them to 0, OR-chains them into an accumulator, and volatile-
        # stores the accumulator. Force_yz ALSO needs to be enabled for this
        # patch to run. Probe values 16/32/64. Default 0 = off.
        set_default_environment('ZLUDA_FORCE_VGPR_ZERO_COUNT', '0')
        # LLVM-IR-level injection of AMDGPU inline asm `v_mov_b32 vN, 0`
        # for v3..v63 at each Downsweep kernel's entry. Clobber list forces
        # LLVM to treat those VGPRs as written. Together with the PTX-level
        # force-yz patch (which covers v1/v2), this should zero-init the
        # full VGPR set before the spill prolog. Default 0.
        set_default_environment('ZLUDA_DS_INLINE_VGPR_ZERO', '0')
        # Range [START, END) of VGPRs to zero via inline asm. Default v3..v64
        # (full coverage after FORCE_YZ covers v1/v2). Shrink if LLVM crashes.
        set_default_environment('ZLUDA_DS_INLINE_VGPR_ZERO_START', '3')
        set_default_environment('ZLUDA_DS_INLINE_VGPR_ZERO_END', '64')
        # HSA runtime env vars forwarded from host for scratch-policy probes.
        # HSA_NO_SCRATCH_RECLAIM=1 disables scratch-pool reclamation. Tested
        # 2026-04-24: shifts wild-fault offset but doesn't eliminate bug.
        set_default_environment('HSA_NO_SCRATCH_RECLAIM', '0')
        set_default_environment('HSA_ENABLE_SCRATCH_ONCE', '0')
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

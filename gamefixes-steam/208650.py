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
    # NOTE 2026-06-03: ZLUDA_IOFX_ACTOR_MEMBAR dropped — patch_iofx_actor_range_membar
    # deleted (redundant with patch_broad_membar_before_bar_sync which covers ALL
    # kernels including IOFX_actorRangeKernel).
    set_default_environment('ZLUDA_LAUNCH_MUTEX', '2')
    # NULL-stream strict: closes a CUDA-spec gap (HIP's null-stream doesn't
    # serialise like CUDA's). Source default OFF; previous protonfix had it
    # at 1 and that was the validated state on the 9-26 baseline.
    set_default_environment('ZLUDA_NULL_STREAM_STRICT', '1')
    # Defensive zeroing of *(rd5+22176) between MS-chain cycles. Validated
    # 2026-05-23 (project_ms1_rd5_reset_fixes_ms4_cascade_2026_05_22.md):
    # 839k clean launches N=5 vs ~2k baseline (415x). Restored to function.rs
    # 2026-06-01 after being stripped by 1aa7ba3a's over-aggressive "clean up
    # investigation probes" pass.
    set_default_environment('ZLUDA_MS1_RD5_RESET', '1')
    # 2026-06-03 A/B switch: ZLUDA_MS1_RD5_RESET_SYNC=1 reverts to the
    # pre-8b53ca00 synchronous hipMemsetD32-in-launch_kernel behavior; =0
    # (default) uses the host-thread drain queue. iter26 vs iter27 A/B in
    # progress to determine if the drain-queue refactor caused the 839k→160k
    # regression.
    set_default_environment('ZLUDA_MS1_RD5_RESET_SYNC', '0')
    set_default_environment('ZLUDA_INTERCEPT_SIMPLE_MEMSET', '0')
    set_default_environment('ZLUDA_ACTORRANGE_DUMP', '0')
    # 2026-06-03: cuEventSynchronize HDP flush. cuStreamSynchronize already does
    # hipDeviceSynchronize via ZLUDA_HDP_FLUSH_ON_SYNC (default-ON); the
    # symmetric event path did not. Ghidra-decoded Class B host crash at
    # PhysX3_x64.dll RVA 0x20ced9 is a vtable-iteration first-deref UAF
    # consistent with PhysX reading a snapshot that still holds stale
    # pointers because GPU writes are stuck in the HDP write buffer.
    # See project_class_b_physx_vtable_crash_decoded_2026_06_03.md. Default
    # OFF — iter30 A/B will determine if it's a net positive.
    # 2026-06-05 Phase 2 shipped (n=3 default-ON): mean 521k vs baseline 294k
    # (+77%), 0/3 Class B hits vs 3/5 in baseline. Class B effectively
    # eliminated; residual is Class A TCP wild-VA.
    set_default_environment('ZLUDA_EVENT_HDP_FLUSH', '1')
    # cuStreamQuery(NULL) startup mask controls. Default behavior is the refined
    # "NotReady before first real launch, real after" mode in stream.rs; these
    # toggles expose the existing A/B overrides through deploy_batman's .env.
    set_default_environment('ZLUDA_STREAM_QUERY_REAL', '0')
    set_default_environment('ZLUDA_STREAM_QUERY_LEGACY', '0')
    # 2026-06-04 Class B null-deref investigation: per-call timeline of
    # cu{Stream,Event,Ctx}Synchronize calls written to
    # /tmp/zluda_sync_timeline.log when set to 1. Off by default.
    set_default_environment('ZLUDA_SYNC_TIMELINE', '0')
    # 2026-06-04 Class B coredump enrichment: set /proc/self/coredump_filter
    # to 0xff so the next coredump captures file-backed mappings (Wine PE
    # heap). Default 0x33 excludes them and `gdb` returns "Cannot access
    # memory" at the failing iterator's array pointer. Off by default.
    set_default_environment('ZLUDA_FULL_COREDUMP', '0')
    # 2026-06-05 Phase 1 Class B host-coherent A/B: when 1, every
    # hipHostMalloc is OR'd with hipHostMallocCoherent (0x40000000),
    # forcing fine-grained pool. Bypasses HDP write-combining so GPU
    # writes to mapped buffers are immediately host-visible. Default 0
    # (legacy coarse-grained behavior).
    set_default_environment('ZLUDA_HOST_COHERENT', '0')
    # 2026-06-05 Class A experiment: hipDeviceSynchronize before every
    # hipTexRefSetAddress/Array to match CUDA's per-launch binding
    # snapshot semantics. PhysX changes texref bindings while in-flight
    # IOFX kernels read them. Default OFF — A/B candidate.
    set_default_environment('ZLUDA_TEXREF_SYNC', '0')
    # 2026-06-02 iter1: ZLUDA_MS4_POST_SYNC=1 → 580 launches, MS_4 GPU fault.
    # No clear improvement vs without it (639 / 283k). Disabled.
    # 2026-06-02 iter2: 8 probe env vars (TEXOBJ_SYNC=0 etc.) → 1028 launches but
    # crashed on compactKernel during PhysX init (same as pre-LD_PRELOAD class).
    # Reverted - the env vars DISABLE sync workarounds and re-introduce races.

    # 2026-06-02 env-var iteration log (iter5..iter8, all rejected):
    # iter5: HSA_ENABLE_SCRATCH_ONCE=0 + HSA_NO_SCRATCH_RECLAIM=0 → no change (552, early host crash)
    # iter6: GPU_MAX_HW_QUEUES=1 → no change (552-62k)
    # iter7: HIP_LAUNCH_BLOCKING=1 → 147k CUB scan crash (different class)
    # iter8: combined LAUNCH_BLOCKING + HW_QUEUES=1 + HSA_ENABLE_INTERRUPT=0
    #        → game HUNG at 63 launches (HIP_LAUNCH_BLOCKING too slow under combo)
    # Distribution of crash classes (ParticleIOS host / IOFX_actorRange GPU /
    # TFS / CUB scan) is essentially identical across all configs. The
    # residual is the Wine-10.0-amplified ROCm 6.2.4 ProfilingSignal pool
    # growth race + the documented IOFX_actorRange cross-stream texref race;
    # neither is fixable via protonfix env vars. Reverted to iter4 baseline.

    # D3D11 interop knobs (no defaults — investigation only):
    #   ZLUDA_D3D11_SUBRESOURCE=0     -- get_mapped_array returns NOT_SUPPORTED
    #   ZLUDA_D3D11_HSA_SKIP_IMAGE=1  -- register_imported skips HSA image create
    # 2026-05-31 A/B: a single run got 392 kernels (5.6x further) with both
    # enabled, but follow-up runs reverted to ~70 kernels — variance, not a
    # real workaround. Crash is in PhysX CUB ParticleIOS_compactKernel regardless
    # of D3D11 config. End users can still .env-override to A/B locally.

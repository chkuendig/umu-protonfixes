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
    set_default_environment('ZLUDA_AR_PTR_TRACE', '0')
    set_default_environment('ZLUDA_IOFX_CAPTURE_SAMPLE', '64')
    # 2026-06-10 A1 writer hunt: per-kernel D2D snapshot of the latched
    # MS_4 counter buffer ([COUNTER CHANGED] timeline). Was deploy-
    # whitelisted only — two-whitelist rule. Default OFF (heavy: D2D+sync
    # per launch).
    set_default_environment('ZLUDA_COUNTER_SNAPSHOT', '0')
    # 2026-06-10 IOFX writer hunt: in-game NRSS final-pass output sortedness
    # check (NRSS_VALIDATE log lines). Default OFF (D2H per final pass).
    set_default_environment('ZLUDA_NRSS_VALIDATE', '0')
    # 2026-06-11 scoped scatter-offset producer-store volatility (root-cause
    # fix for the NRSS phase-3 LDS race). Default OFF pending A/B.
    set_default_environment('ZLUDA_LDS_PRODUCER_VOLATILE', '0')
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
    # IOFX kernels read them.
    # 2026-06-07 SHIPPED default-ON: n=3 with ZLUDA_LOG=1 confirmed
    # IOFX_actorRangeKernel was the kernel responsible for the
    # single-store wild-VA Class A signature (TCP+RW=1+PERM=0x5, last
    # kernel before SIGABRT in 17:21 bench). With TEXREF_SYNC=1 the
    # TCP wild-write disappears in 2 of 3 runs; only display-side CB/DB
    # faults remain. Run distribution: 2411 / 110,374 / 346,963 launches
    # (mean 153k vs baseline ~6k). Best run 17.86 min runtime, 347k
    # launches — vastly exceeds 14.1-min baseline. Class B (eliminated
    # by EVENT_RELEASE_SYSTEM) did not re-emerge. The 2026-06-05 n=2
    # rejection was masked by then-unfixed Class B; with that fixed, the
    # texref sync no longer regresses.
    set_default_environment('ZLUDA_TEXREF_SYNC', '1')
    # 2026-06-09 Class A bf000 candidate fix. Post-sync after cuTexRefSet*
    # (in addition to pre-sync). Hypothesis: AMDGPU TCP/SRD cache carries
    # read-only texref flavor for pages of a bound buffer; subsequent
    # direct st.global writes to byte 0 of that buffer fault with
    # PERMISSION_FAULTS=0x5. Default OFF (A/B candidate). See
    # project_bf000_buffer_is_flex_position_2026_06_09.md.
    set_default_environment('ZLUDA_TEXREF_POST_SYNC', '0')
    # 2026-06-09 Class A counter-coherency investigation. Log every
    # cuMemcpyHtoD* / cuMemsetD* with size <= 256 bytes (= likely
    # per-frame counter / control-struct writes). Hypothesis under
    # test: the per-actor counter that MS_4's atom.global.add reads
    # is written by host but the write doesn't get HDP-flushed, so
    # GPU's first atom.add of a new frame reads a stale value →
    # underflow → wild scatter. Default OFF (diagnostic). See plan
    # ~/.claude/plans/counter-coherency-bf000-2026-06-09.md.
    set_default_environment('ZLUDA_COUNTER_TRACE', '0')
    # 2026-06-09 Class A counter-coherency fix candidate. Force a
    # hipDeviceSynchronize after every small cuMemsetD32Async (n<=64).
    # n9 trace caught 3 different streams clearing the same counter
    # alloc → cross-stream race possible. Default OFF; flip to 1 to A/B.
    set_default_environment('ZLUDA_COUNTER_CLEAR_SYNC', '0')
    # 2026-06-09 MS_4 PTX underflow guard — the surgical fix for the Class A
    # wild-VA scatter bug at its root. Inserts `setp.lt.s32 r10, 0; @ bra
    # ret;` before MS_4's `mul.wide.u32 r10*4` to skip the wild store when
    # the counter has underflowed. The bug also fires on NVIDIA driver
    # >=461.09 (Steam Community Guide 2433732129 advises downgrade to
    # 456.55). Default ON; set to 0 to disable. See
    # project_class_a_is_genuine_physx_bug_2026_06_09.md.
    set_default_environment('ZLUDA_MS4_UNDERFLOW_GUARD', '1')
    # 2026-06-10 MS_4 guard reinvestigation: pre-launch capture ring +
    # counter-vs-real-count audit (MS4_SAFE / MS4_COUNTER_UNDERFLOW log
    # lines from summarize_ms4_capture). Was whitelisted in
    # deploy_batman.sh but missing here — two-whitelist rule.
    # Default OFF (diagnostic, 2 D2D + sync per MS_4 launch).
    set_default_environment('ZLUDA_MS4_CAPTURE', '0')
    # 2026-06-10 Class A FleX investigation: per-pass FleX sort output
    # validation (readback after every NULL-stream RadixSortDownsweep,
    # logs PASS_CORRUPT with spine_max/spine_sum on bit31 corruption).
    # May-2026 infra; needed here because .env overrides only apply to
    # vars with a set_default_environment call (two-whitelist rule).
    # Default OFF (diagnostic).
    set_default_environment('ZLUDA_VALIDATE_SORT', '0')
    # 2026-06-10 prefix-scan root cause: cross-engine (SDMA<->compute)
    # ordering is unreliable on Van Gogh + ROCm 6.2.4 + gfx1033 override.
    # Lab A/B (cub_sort_multiblock_*_amdgpu): default SDMA = 3/3 hangs
    # (lost completion signal, GPU idle, runtime spinning); SDMA=0 =
    # 1500/1500 game-shaped sort iterations clean in 7s. The in-game
    # manifestation is the May-11 spine "doubling ladder" (first corrupt
    # entry = exactly 2n): the spine MEMSET (SDMA) doesn't land before
    # Upsweep (compute) adds counts -> cross-frame accumulation -> wild
    # Downsweep scatter -> Class A GPU fault. '1' = stock behavior;
    # bench A/B passes 0 via .env.
    # 2026-06-12: flipped default to '0' — in-game validated across multiple
    # SDMA=0 benches reaching 35k+ launches with NRSS 61/61 sorts clean (0
    # spine-doubling Class A faults). The residual perf-test wild-VA is a
    # separate StepKernel grid=4 scatter bug, unrelated to SDMA.
    set_default_environment('HSA_ENABLE_SDMA', '0')
    # 2026-06-10 NRSS lab-driver capture: log all 13 typed args + dims for
    # the first 600 IOFX_newRadixSortStepKernel launches. Feeds the
    # isolated game-shape reproducer. Default OFF (diagnostic).
    set_default_environment('ZLUDA_NRSS_ARG_TRACE', '0')
    # 2026-06-05 Class A experiment: drain in-flight kernels BEFORE MS_4
    # launches. MS_4 has atom.global.add[ptr], -1 — if counter is 0 when
    # MS_4 reads, the atomic wraps to 0xFFFFFFFF → out-of-bounds store.
    # The counter is written by MS_2_Batched (BB11_9). Force the producer
    # to drain via:
    #   '0' = off (default)
    #   '1' = hipDeviceSynchronize
    #   '2' = hipStreamSynchronize (this stream only)
    set_default_environment('ZLUDA_MS4_PRE_SYNC', '0')
    # 2026-06-08 Class B at depth: MS_3 has `atom.global.add count` then
    # 4 st.global.u16 slot stores. On AMD HDP write-combine can deliver
    # count update before slot stores → PhysX polling sees count=N+1
    # with slot[N]=NULL → Class B vtable NULL deref at
    # PhysX3_x64.dll+0x20ced9. ZLUDA_MS3_POST_SYNC triggers a sync
    # POST-launch of MS_3 to force HDP flush before PhysX polls.
    #   '0' = off
    #   '1' = hipDeviceSynchronize (cross-stream drain + HDP flush)
    #   '2' = hipStreamSynchronize (this stream only)
    # 2026-06-08 SHIPPED default '1': n=2 with TEXREF_SYNC=1 +
    # MS3_POST_SYNC=1: 312,495 launches / 15.13 min and 336,833 / 17.11
    # min — both SIGABRT (Class A GPU fault), 0 of 2 SIGSEGV (Class B
    # host crash). vs prior n=1 at same depth (run 4 17.84 min)
    # which hit Class B. Cost = one hipDeviceSync per MS_3 launch.
    set_default_environment('ZLUDA_MS3_POST_SYNC', '1')
    # 2026-06-05 Class B residual: PhysX reads from host-mapped buffer that
    # MemCopyBatchedAsync populates. If PhysX skips cuEventSync, HDP write-
    # combine holds the writes. hipDeviceSync after MemCopyBatchedAsync
    # triggers HDP flush MMIO. A/B candidate.
    # 2026-06-05 SHIPPED default-ON: n=2 → mean 1.10M (~2.6× v5+Phase 2
    # baseline 429k), Class B 1/2 but at much later launch counts.
    set_default_environment('ZLUDA_MEMCPY_HDP_FLUSH', '1')
    # 2026-06-05 Class B residual: PhysX uses cuStreamQuery as un-synced
    # "is this kernel done?" poll (6,015 calls/run vs 51k cuEventSync).
    # When query returns SUCCESS, PhysX reads mapped-host but HDP can hold
    # writes. Hook hipDeviceSync on SUCCESS to flush HDP.
    set_default_environment('ZLUDA_QUERY_HDP_FLUSH', '1')
    # 2026-06-06 Class B residual fix at AQL packet level: OR-in
    # hipEventReleaseToSystem (0x80000000) on every cuEventCreate. Every
    # cuEventRecord then triggers a system-scope release fence which on
    # RDNA includes HDP MMIO flush — drains HDP buffer at every event
    # record (43k/run). Replaces per-API hipDeviceSync hooks with a
    # single semantic change. Default ON; set =0 to opt out.
    set_default_environment('ZLUDA_EVENT_RELEASE_SYSTEM', '1')
    # 2026-06-06 Class A upstream-writer hunt — diagnostic only.
    # ZLUDA_MEMCPY_SRC_TRACE=1 widens the MemCopyBatchedAsync descriptor
    # log to include descriptors whose dst lands near the relay struct
    # (off ~1 MiB). ZLUDA_MEMCPY_SRC_WRITER=1 then logs every kernel
    # whose arg pointer falls within the src allocation tracked from
    # those descriptors — identifying which compute kernel writes the
    # over-counted shape count. Both default OFF (diagnostic).
    set_default_environment('ZLUDA_MEMCPY_SRC_TRACE', '0')
    set_default_environment('ZLUDA_MEMCPY_SRC_WRITER', '0')
    # 2026-06-09 Class A bf000 investigation: ZLUDA_FREE_AFTER_LAUNCH_SCAN
    # enables narrow per-kernel ptr-range capture for IOFX_actorRangeKernel,
    # IOFX_newRadixSortStepKernel, TurbulenceFS_dynamicGridMG*, makeSortKeys,
    # remap, and newRadixSortBlock. On SIGABRT/SIGSEGV the signal handler
    # dumps these into /tmp/zluda_tex.log so we can identify whether the
    # bf000 buffer (= start of a 1.04 MiB particle buffer) is one of the
    # proximate kernel's args. Default OFF (diagnostic; small per-launch
    # cost but the data is invaluable when crash fires).
    set_default_environment('ZLUDA_FREE_AFTER_LAUNCH_SCAN', '0')
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

    # Generic .env passthrough: apply ANY override key from 208650.env that the
    # explicit set_default_environment() calls above did not already handle.
    # Without this, investigation-only knobs (ZLUDA_D3D11_COPY_PATH,
    # ZLUDA_D3D11_HSA_ISOLATE, ZLUDA_D3D11_SUBRESOURCE, ...) were read into
    # FILE_OVERRIDES but never injected into the game env — they only took effect
    # if named in an explicit call here. deploy_batman.sh's whitelist controls
    # which shell vars reach the .env; this loop ensures every one of them lands.
    for _key, _value in FILE_OVERRIDES.items():
        util.set_environment(_key, _value)

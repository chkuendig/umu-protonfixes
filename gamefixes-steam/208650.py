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
        # 2026-04-28: instrument TFS_fieldSamplerPointsKernel to capture per-thread
        # velocity field reads from TurbulenceFS_dynamicgrid_uvwaTexFS — confirms
        # whether the "missing smoke effects" symptom is a zero/junk velocity grid.
        set_default_environment('ZLUDA_TFS_INSTRUMENT', '1')
        # 2026-04-28: log every cuMemcpyDtoH{,Async,_ptds} dst/src/size +
        # first 16 dwords. Mode-B SIGSEGV root-cause hunt: PhysX reads back
        # rare scalars from GPU; one of them is suspected to drive a
        # corrupt-count host-side allocation that ends in a NULL-prefix
        # array iteration crash. Async path syncs the stream before logging.
        # DISABLED for Phase 2 disambiguation: heavy logging (130k+ lines
        # per run) perturbs timing — runs went from ~5min crash to 22min
        # success-but-timeout. Light instrumentation needed for clean A vs
        # B disambiguation. Re-enable for targeted Phase 3 probes.
        set_default_environment('ZLUDA_DTOH_LOG', '0')
        # 2026-04-28: one-shot kernel-arg logger. For each kernel's FIRST
        # launch, walks kernel_params and logs any 8-byte arg value within
        # the ZLUDA ALLOC range (likely device pointer). Cross-ref against
        # DtoH source pointers — confirms whether the always-zero DtoH'd
        # buffers (e.g., 0x7f84b2400000) are ever bound to a kernel.
        # 2026-04-28 evening: re-enabled after defensive fix in function.rs:
        # - Cap walk at 12 indices
        # - Pre-check kernel_params[i] is in plausible user-space ptr range
        #   (0x1000 < p < 0x8000_0000_0000_0000) before deref
        # - Bail on bogus pointers, not just NULL
        set_default_environment('ZLUDA_KARG_LOG', '1')
        # 2026-04-29 Phase 3a Path 1b: sample first qword of each 1040000-byte
        # cuMemHostAlloc'd buffer at alloc time and every 1000th kernel launch.
        # Goal: identify which buffer's slot-0 stays NULL leading up to crash;
        # diff PASS vs FAIL trajectories to localize the missing-init.
        set_default_environment('ZLUDA_HOSTBUF_SAMPLE', '1')
        # 2026-04-28 Phase 3a probe: insert membar.cta before every bar.sync 0
        # in IOFX_actorRangeKernel. A/B test for the LDS-publish hypothesis —
        # IOFX_actorRangeKernel was the last kernel before Phase 2 run 1's
        # SIGABRT (Mode A) and is upstream of Mode B SIGSEGV array-NULL data.
        # If this changes the crash distribution across 5+5 runs, PR #161638
        # is incomplete for some access pattern. Default OFF; protonfix
        # turns it ON for the probe runs.
        # 2026-04-29 Phase 3g: re-enable membar.cta before bar.sync 0 in
        # IOFX_actorRangeKernel. Phase 3e localized this kernel as the
        # dominant fault producer (100/274 = 37%). Single-run validation
        # showed AF=1009 SIGSEGV with 0 GPU page faults (IOFX wild-scatter
        # eliminated; crash moved downstream to IOFX_volumeMigrationKernel).
        # 2026-05-07 (Phase 2.2 REFUTED): wrapper-alone N=4 = 0/4
        # complete vs baseline 30%. 2× IOFX, 2× FleX ParticleIOS
        # crashes. The PTX-text membar.cta patches were NOT redundant
        # with the bar_sync wrapper's s_waitcnt — likely because:
        #   (a) membar.cta scope differs from workgroup s_waitcnt
        #   (b) some bar.sync sites don't route through the wrapper
        # Re-enabling. See project_phase22_wrapper_alone_refuted_2026_05_07.md
        set_default_environment('ZLUDA_IOFX_ACTOR_MEMBAR', '1')
        # 2026-04-29: also enable broad membar across PhysX/FleX/TFS/IOFX
        # kernels — the LDS visibility class affects more than just
        # actorRangeKernel (e.g. IOFX_volumeMigrationKernel was next).
        # Earlier 3-run test of broad alone showed 0/3, but that was
        # without the s_waitcnt restore + IOFX-actor-membar combo;
        # re-validating in this layered configuration.
        # 2026-05-06: smoke regression isolated to D3D11 SubResource
        # hipArray path (commit 70683673), NOT this. The broad_membar
        # PTX rewrite inserts membar.cta before bar.sync 0 in every
        # kernel; codegen-level Phase 2.2 supersedes it.
        # 2026-05-07 (Phase 2.2 REFUTED): same N=4 batch as
        # ZLUDA_IOFX_ACTOR_MEMBAR — wrapper alone cannot cover what
        # this broad PTX rewrite catches. Re-enabling.
        set_default_environment('ZLUDA_BROAD_MEMBAR', '1')
        # Option A REFUTED (2026-05-05, iter86-iter92): patch_scan_kernel_reader_acquire
        # (membar.cta AFTER bar.sync 0 in RadixSortScanKernel) was harmful.
        # 5 runs: iter87(29k MS_4), iter88(10k CUB), iter89(9k IOFX), iter91(95k IOFX),
        # iter92(3k TFS). 0/5 complete, all <100k vs baseline 35% complete with
        # range 14k-419k. The doubled membar.cta in ScanKernel (broad_membar adds
        # before, scan_acquire adds after) appears to break ScanKernel's algorithm.
        set_default_environment('ZLUDA_SCAN_READER_ACQUIRE', '0')
        # 2026-05-07 (Phase 3 REFUTED): hipDeviceSynchronize after
        # hipTexObjectCreate (default ON in tex.rs::object_create per
        # commit pending) caused 0/3 valid runs vs baseline 30% — all
        # 3 crashed in FleX kernels (MemCopyBatchedAsync, ParticleIOS_*).
        # Hypothesis: per-tex-object device-wide sync stalls disrupt
        # FleX timing, exposing latent races. Disable default.
        # See project_phase3_texobj_sync_refuted_2026_05_07.md.
        set_default_environment('ZLUDA_TEXOBJ_SYNC', '0')
        # 2026-05-06 (Phase 2.1): superseded by codegen-layer fix in
        # zluda/ptx/src/pass/llvm/emit.rs::emit_ld — `ld.shared` loads
        # are now unconditionally emitted as volatile LLVM loads at the
        # PTX→IR translation level. The PTX-text patch_broad_volatile_lds
        # is now redundant; disabling. See plan resilient-sprouting-pie.md.
        set_default_environment('ZLUDA_BROAD_VOLATILE_LDS', '0')
        # 2026-04-29 Phase 3g: HDP-flush experiment. Mode B SIGSEGVs continue
        # even after LDS-visibility patches eliminated GPU page faults. Adds
        # hipDeviceSynchronize + CPU memory fence after every cuStreamSynchronize.
        # Tests whether the bug is HDP cache lag for zero-copy mapped memory
        # (NVIDIA's cuStreamSynchronize implicitly flushes HDP; HIP may not).
        set_default_environment('ZLUDA_HDP_FLUSH_ON_SYNC', '1')
        # 2026-04-29: enabled to capture input_324, input_329 + pre/post output
        # state for IOFX_actorRangeKernel — Phase 3e localized this kernel as
        # the dominant fault producer (100/274 events). We now want to inspect
        # input values to distinguish "wild input data" vs "kernel codegen bug".
        set_default_environment('ZLUDA_IOFX_CAPTURE', '1')
        set_default_environment('ZLUDA_IOFX_BBOX_CAPTURE', '0')
        set_default_environment('ZLUDA_DOWNSWEEP_CAPTURE', '1')
        # 2026-04-26 NIGHT: pair Scan capture with Downsweep capture so we
        # have post-Upsweep (= pre-Scan) AND post-Scan (= pre-Downsweep)
        # snapshots of d_spine. Lets us pinpoint whether wrong d_spine
        # comes from Upsweep or from Scan. Output: /tmp/cub_scan_captures.bin
        # See project_real_root_cause_is_scankernel_2026_04_26.md.
        set_default_environment('ZLUDA_SCAN_CAPTURE', '1')
        # Sort-guard probe: when '1', over-allocates 520000-byte allocations
        # (CUB sort temps) with a 512 KB DEADBEEF-filled pre-pad. The SIGABRT
        # handler scans the pad post-mortem — any non-DEADBEEF byte is a
        # wild store from Downsweep into what would otherwise be unmapped
        # memory. Identifies where the wild pointer comes from without
        # modifying any kernel.
        set_default_environment('ZLUDA_SORT_GUARD', '1')
        # WAR-barrier patch for CUB Downsweep digit-offset LDS read.
        # Inserts `bar.sync 0;` before the `mul.wide.u32 _, _, 4 ;
        # add.s64 _, %rd<lds_base>, _ ; ld.shared.u32 _, [_]` triplet to
        # close the LDS WAR hazard between phases that share temp_storage.
        # ZLUDA_DOWNSWEEP_WAR_BARRIER: env-var no longer read by the
        # current code (the corresponding patch was removed). Left as
        # dead config until full cleanup pass.
        set_default_environment('ZLUDA_DOWNSWEEP_WAR_BARRIER', '0')
        # Rank-mask DIAGNOSTIC: clamp the scatter-rank register to 14 bits
        # right after `add.s32` and before `cvt.s64.s32 / mul.wide.s32`.
        # If sort_guard's wild writes go away with this on, the high bits
        # of the rank are residue/garbage and a more targeted fix is
        # possible. Sort output is INCORRECT under this mask — diagnostic
        # only, do not ship enabled.
        set_default_environment('ZLUDA_DOWNSWEEP_RANK_MASK', '0')
        # SCAN-output clamp: insert `max.s32 %rDst, %rDst, 0;` between
        # every `sub.s32 %rDst, _, _;` → `st.shared.u32 [_], %rDst;`
        # sequence in CUB Downsweep. Targets the BlockScan exclusive-
        # prefix-sum subtract that can underflow on AMDGPU and produce
        # negative i32 LDS slots that then become wild scatter ranks.
        # Histogram counts are non-negative so a non-negative prefix is
        # always correct.
        #
        # **VERIFIED FIX (2026-04-25)**: 3 runs with this patch + sort_guard
        # show guard[0] dirty_words=0 (down from 7832 baseline). Sort
        # completes 10 passes per frame; crash now lands at CreateGrid
        # (a separate, known issue per
        # project_creategrid_dtod_overwrite.md).
        #
        # **2026-04-27**: now redundant after LLVM PR #161638 backport
        # (cross-wave LDS visibility fix). The underflow this was clamping
        # was a downstream symptom of warp 0's writes not being visible to
        # warp 1; with the fence fixed, the subtract produces correct
        # values. A/B test (5+5 runs) showed no functional impact (median
        # kernels: clamp=1 → 254k, clamp=0 → 257k; same crash families).
        # Default OFF; patch stays in tree as a gated rollback option.
        # See project_pr161638_breakthrough_2026_04_27.md.
        # SCAN_CLAMP and IOFX_ACTOR_64BIT_GUARD — REMOVED 2026-05-06.
        # User clarified the strict no-mutation policy: replacing a
        # crashing wrong value with a non-crashing wrong value is still
        # value mutation. The functions were deleted from module.rs.
        # Real fix path is at codegen layer (see plan
        # resilient-sprouting-pie.md): force volatile=true for ld.shared
        # in PTX translator, and use inline-asm s_waitcnt 0 in the
        # bar_sync C++ wrapper instead of LLVM-fence which gets optimized
        # away. These codegen fixes are universal across all kernels.
        # 2026-04-25 evening: ANDs out bit-31 on every keys_out scatter
        # write inside RadixSortDownsweepKernel. Default OFF until in-game
        # validation lands. Targets the post-SCAN-clamp CreateGrid crash —
        # see project_phase12_pr145902_live_2026_04_25.md and
        # project_creategrid_post_scan_clamp_2026_04_25.md.
        # CLEAR_BIT31 overwrites bit 31 of every keys_out scatter write
        # — mutates computed values to mask bugs. User direction
        # 2026-05-05: NEVER mutate values. Keep OFF.
        set_default_environment('ZLUDA_DOWNSWEEP_CLEAR_BIT31', '0')
        # 2026-04-26: ZLUDA_DOWNSWEEP_BLOCKSCAN_PRESYNC retired —
        # superseded by the explicit s_waitcnt(0) added to the bar_sync
        # wrapper in zluda_ptx_impl.cpp (commit 579bcc05), which covers
        # every bar.sync in every kernel.
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
        # the enable bits. 2026-04-24: enabled as partial fix (eliminates
        # v1/v2 wild writes, confirmed via instrumentation). Other VGPRs
        # (v3..v63) still carry residue, but this improves reliability.
        set_default_environment('ZLUDA_FORCE_WORKITEM_ID_YZ', '1')
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
        # 2026-04-27: RadixSortScanKernel instrumentation. Captures
        # warp-0-lane-K's exclusive-prefix `%r94` (what gets STORED to
        # the cross-warp LDS handshake slot) and thread-T's `%r9` (what
        # gets READ back after the bar.sync that should publish those
        # writes). Reuses the DS_DEBUG_BUFFER → /tmp/ds_instrument_witness.bin.
        # Default ON for the active investigation: distinguishes between
        # "warp scan failed" (r94=0) and "bar.sync failed to publish"
        # (r94=8109 but r9=0). See project_morning_briefing_2026_04_27.md.
        set_default_environment('ZLUDA_SCAN_INSTRUMENT', '1')
        # 2026-04-27: drop `+precise-memory` from the AMDGPU codegen
        # target-features for this game. Per ROCgdb docs, that flag is
        # a debugger fault-precision knob (drains memcnt after every
        # memory op for accurate fault PCs), NOT a scheduler-reorder
        # barrier as our prior comment claimed. The cited LLVM #131386
        # and Codeplay GROMACS posts are register-allocator bugs,
        # unrelated. Cost: ~5-10% perf, ~5 KB of extra s_waitcnt asm
        # per Scan kernel. Validated A/B (5+5 runs): 5/5 reach Scan
        # with =0 vs 4/5 with =1; same crash families; median 253k vs
        # 259k kernels (within run-to-run variance).
        # See reference_amdgpu_hw_claims_with_sources.md.
        set_default_environment('ZLUDA_PRECISE_MEMORY', '0')
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
        # Concurrent-operations diagnostics (2026-04-24). Logs every kernel launch
        # and D2D copy with stream info to detect if overlapping operations or 
        # stream ordering issues cause the Downsweep fault. Default 0; set to 1
        # via 208650.env to test. Can greatly increase log size (~multi-MB runs).
        set_default_environment('ZLUDA_LOG_CONCURRENT_OPS', '0')
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
        # Phase 0b probe (iter48+iter49, 2026-05-05): retest of disabling
        # FeatureBackOffBarrier. Result: HARMFUL. Both runs crashed early
        # (35k, 55k kernels) in CUB RadixSort with 10/20 dmesg page faults
        # vs the keep-enabled baseline of ~385k kernels with 0 dmesg. Revert.
        # This matches project_back_off_barrier_disable_hung_replay_2026_04_26.md.
        # Keep the env-gate in compile.rs but don't set it on by default.
        # iter66/iter67 probe (2026-05-05): disabled B→A reverse blit
        # to test whether dxvk-submit page faults are caused by our
        # CopyExternalBufferToSurface. Result: REFUTED. iter66 (332k
        # IOFX_actorRange) and iter67 (67k CUB Downsweep) both still
        # produced both TCP write (compute) and CB/DB read (graphics
        # cascade) page faults. The dxvk-submit fault is a downstream
        # consequence of the compute-side TCP write fault, not an
        # independent issue. Reverted (env-gate stays in code).
        # SubResource path findings (2026-05-05, iter46–iter54):
        # iter42/iter43 had clean TFS-late crashes (0 dmesg). After that
        # almost every run crashed in CUB RadixSort with 10+ dmesg page
        # faults. iter53 fully completed (418k) but iter54 crashed at
        # 47k — confirms that's a flaky single outcome.
        #
        # Diagnosis: the SubResource impl makes PhysX use real D3D11
        # textures via hipArray, which exercises CUB Radix Sort with
        # different/larger inputs than the no-interop fallback path.
        # Existing CUB Downsweep patches (scan_clamp, clear_bit31,
        # war_phase_barrier, blockscan_presync, broad_volatile_lds)
        # were tuned for the smaller workload — they don't cover this
        # new access pattern.
        #
        # 2026-05-09 Phase A.HSA-wired: SubResource ON with the new
        # path-(b) HSA-backed array (commits A.HSA.0–5 + this iteration).
        # ensure_hip_array now creates an HSA image bound to the
        # imported D3D11 buffer; surf::object_create returns the HSA
        # image's gfx10 T# pointer as the surface object. Kernel
        # surf2Dwrite writes through to the original D3D11 memory
        # (validated end-to-end by d3d11_interop_hsa_test).
        # If smoke renders: ship. If not: investigate which CUDA API
        # APEX inspects post-success that we haven't covered yet.
        set_default_environment('ZLUDA_D3D11_SUBRESOURCE', '1')
        set_default_environment('DXVK_LOG_LEVEL', 'warn')
        # DXVK normally routes logs through Wine's __wine_dbg_output, which
        # WINEDEBUG=-all silences. DXVK_LOG_PATH=/tmp makes DXVK write
        # <exe>_d3d11.log etc directly, bypassing Wine debug entirely so
        # our ZLUDA-DXVK warn/err messages from CopySurfaceToExternalBuffer
        # / CopyExternalBufferToSurface land somewhere inspectable.
        set_default_environment('DXVK_LOG_PATH', '/tmp')
        set_default_environment('WINEDEBUG', '-all')
        # HIP runtime logging OFF — AMD_LOG_LEVEL>0 triggers ProfilingSignal race
        set_default_environment('AMD_LOG_LEVEL', '0')

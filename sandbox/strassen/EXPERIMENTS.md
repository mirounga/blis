# SubCuber performance experiments

This note records the three initial follow-up experiments run on 2026-07-31
against the one-level BLIS SubCuber implementation, followed by the packing
and epilogue optimizations implemented from their recommendations. Experiments
1--3 used the original scalar two-source packer and are preserved as
historical results. Set `BLIS_SUBCUBER_BENCH_SCALAR_PACK=1` when reproducing
those measurements with the current harness.

## Method

The measurements used a clean out-of-tree Apple arm64 build configured with
`-s strassen -t pthreads -d opt arm64`. The benchmark in
`bls_subcuber_bench.c.in`:

- allocates and randomizes each matrix once;
- warms both configurations;
- targets at least 250 ms per sample;
- alternates configuration order;
- reports the median of seven samples;
- can switch schedules, packing paths, or block sizes before every GEMM in one
  process; and
- checks both configurations independently against an untimed
  conventional-BLIS result before timing.

The last property is important on macOS, where absolute four-thread DGEMM
times varied noticeably between process launches. Configuration decisions
below therefore use the paired, within-process ratio. Runtime change is

```text
100 * (alternate / original - 1)
```

so a negative value is an improvement. Each paired-batch result was repeated
in three process launches; the table reports the median of those three
seven-sample comparisons.

Build the harness from an out-of-tree build directory with:

```sh
cc -O3 -Iinclude/arm64 -x c \
  /path/to/blis/sandbox/strassen/bls_subcuber_bench.c.in \
  -x none lib/arm64/libblis.a -lpthread -lm -o subcuber_bench.x
```

It accepts either a square dimension or `m n k`:

```sh
BLIS_NUM_THREADS=8 ./subcuber_bench.x s 4096
BLIS_NUM_THREADS=8 ./subcuber_bench.x s 2304 2304 12288
```

The comparison modes select these within-process pairs:

| `BLIS_SUBCUBER_BENCH_COMPARE` | Original side | Alternate side |
| --- | --- | --- |
| `baseline` | Conventional BLIS | Default SubCuber, optionally modified by `BLIS_SUBCUBER_BENCH_ALT_MODE` |
| `trail` | Original term order | Locality-trail order |
| `pair` | Original schedule, vector pair fusion disabled | Paired schedule, vector pair fusion disabled |
| `simd` | Scalar two-source packing | Default SIMD two-source packing |
| `pair-fused` | Paired, ordinary scalar packing | Paired, fused scalar packing |
| `pair-fused-simd` | Paired, ordinary scalar packing | Paired, default SIMD and shared-load vector fusion |
| `pair-fusion` | Paired SIMD packing, two ordinary traversals | Paired SIMD shared-load vector fusion |
| `combined` | Original schedule with scalar packing | Paired schedule with default SIMD and vector pair fusion |
| `blocks` | Base NC/KC overrides | Alternate NC/KC overrides |

For `baseline`, `BLIS_SUBCUBER_BENCH_ALT_MODE` accepts `trail`, `pair`,
`scalar`, or `combined`. `BLIS_SUBCUBER_BENCH_SCALAR_PACK=1` forces scalar
two-source packing on both sides of any comparison; it is intended primarily
to reproduce the historical schedule and block-size measurements and should
not be combined with a mode whose purpose is to measure SIMD.

## Experiment 1: locality-trail term order

The experimental order is

```text
M6, M2, M4, M7, M5, M3, M1
```

or zero-based `{5,1,3,6,4,2,0}`. Every consecutive pair shares at least one
C quadrant, and each quadrant's first contribution remains positive so beta
can still be folded into the first touch.

Enable it with:

```sh
BLIS_SUBCUBER_EXPERIMENT_TERM_TRAIL=1 ./your-gemm
```

Direct comparison with the original schedule:

| Case | Original | Locality trail | Runtime change |
| --- | ---: | ---: | ---: |
| DGEMM 3072, 1 thread | 799.075 ms | 799.095 ms | +0.00% |
| DGEMM 3072, 4 threads | 221.780 ms | 220.622 ms | -0.52% |
| SGEMM 2304, 8 threads | 28.614 ms | 28.560 ms | -0.19% |
| SGEMM 4096, 8 threads | 159.129 ms | 159.152 ms | +0.01% |

Conclusion: this is noise-level and not a promotion candidate. A term order
cannot keep an individual C microtile resident because the current loop
finishes an entire quadrant-wide term before starting the next one.

Reproduce a direct comparison with:

```sh
BLIS_SUBCUBER_BENCH_COMPARE=trail \
BLIS_THREAD_IMPL=pthreads BLIS_NUM_THREADS=8 \
./subcuber_bench.x s 4096
```

## Experiment 2: NC and KC sweeps

The first structural result is that all four square cases already execute
one full half-N block and one full half-K block:

| Case | Half N | Half K | Context NC | Context KC |
| --- | ---: | ---: | ---: | ---: |
| DGEMM 3072 | 1536 | 1536 | 8184 | 3072 |
| SGEMM 2304 | 1152 | 1152 | 9600 | 4096 |
| SGEMM 4096 | 2048 | 2048 | 9600 | 4096 |

Consequently, increasing NC or KC on these squares is an exact no-op. The
useful square sweep is downward, to see whether tighter cache residency can
repay extra packing and C passes.

### NC sweep

KC was held at the full half-K value. The entries below are the runtime
change relative to full half-N NC.

| Case | Full NC | Half NC | Quarter NC | Eighth NC |
| --- | ---: | ---: | ---: | ---: | ---: |
| DGEMM 3072, 1 thread | 1536 | +0.90% | +2.20% | +5.50% |
| DGEMM 3072, 4 threads | 1536 | +0.82% | +3.93% | +8.30% |
| SGEMM 2304, 8 threads | 1152 | +4.14% | +10.50% | +26.88% |
| SGEMM 4096, 8 threads | 2048 | +4.89% | +11.70% | +20.44% |

Full half-N wins every comparison. Smaller NC repeats the entire A packing
schedule for each additional N block, overwhelming any cache-locality gain.

### KC sweep

NC was held at full half-N. The entries compare smaller KC values with one
full half-K pass. The DGEMM 3072/4 quarter-K point was repeated three times;
its median was -0.05%, confirming that its initial apparent 0.95% win was
noise.

| Case | Full KC | Half KC | Quarter KC |
| --- | ---: | ---: | ---: | ---: |
| DGEMM 3072, 1 thread | 1536 | -0.19% | +1.00% |
| DGEMM 3072, 4 threads | 1536 | -0.06% | -0.05% |
| SGEMM 2304, 8 threads | 1152 | +1.06% | +5.36% |
| SGEMM 4096, 8 threads | 2048 | +5.39% | +12.81% |

DGEMM is insensitive over this range; SGEMM strongly prefers one full K
pass because smaller KC adds pack collectives and revisits C.

### Long-K check

Rectangular cases make a larger-than-context KC a real experiment rather
than a no-op:

| Case | Context KC | Full half-K KC | Context time | Full time | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGEMM 2304x2304x12288, 8 threads | 4096 | 6144 | 149.636 ms | 149.478 ms | -0.11% |
| DGEMM 1536x1536x8192, 4 threads | 3072 | 4096 | 151.396 ms | 152.596 ms | +0.79% |

There is no general benefit from an adaptive larger KC. Retain the native
context KC, capped naturally by the half-problem dimension.

A direct block comparison uses:

```sh
BLIS_SUBCUBER_BENCH_COMPARE=blocks \
BLIS_SUBCUBER_BENCH_BASE_NC=2048 \
BLIS_SUBCUBER_BENCH_ALT_NC=1024 \
BLIS_SUBCUBER_BENCH_BASE_KC=2048 \
BLIS_SUBCUBER_BENCH_ALT_KC=2048 \
BLIS_NUM_THREADS=8 ./subcuber_bench.x s 4096
```

## Experiment 3: paired batch packing and C-tile accumulation

This path groups terms that share one A source, one B source, and one C
destination:

| Batch | Terms | Shared A | Shared B | Hot C |
| --- | --- | --- | --- | --- |
| 0 | M6 | - | - | - |
| 1 | M7, M1 | A22 | B22 | C11 |
| 2 | M2, M4 | A22 | B11 | C21 |
| 3 | M3, M5 | A11 | B22 | C12 |

Both packed forms belong to one collective generation. Each temporal slot
has fixed capacity for two forms, and shared communicators alternate two
complete batch slots. After one publication barrier, the microkernel applies
both products to the same owned C microtile before advancing. Its next-panel
prefetch pointers follow that actual form-zero/form-one execution order.

This layout is required for correctness. Retaining two consecutive ordinary
pack generations in the old two-slot ring would let a fast worker overwrite
the first form while a slower peer was still consuming it.

Enable the path with:

```sh
BLIS_SUBCUBER_EXPERIMENT_PAIR_BATCH=1 ./your-gemm
```

If both experimental schedule flags are set, the paired schedule takes
precedence. Direct comparison against the original SubCuber schedule:

| Case | Median runtime change | Three-run range | Decision |
| --- | ---: | ---: | --- |
| DGEMM 3072, 1 thread | -0.10% | -0.13% to +0.24% | neutral |
| DGEMM 3072, 4 threads | +1.04% | +0.57% to +1.27% | regression |
| SGEMM 2304, 8 threads | +0.34% | -0.04% to +0.55% | neutral |
| SGEMM 4096, 8 threads | -1.96% | -3.67% to -1.82% | useful niche |

At SGEMM 4096/8, a final conventional comparison measured 182.511 ms for
BLIS and 156.947 ms for paired SubCuber, a 14.01% runtime reduction. The
original schedule's adjacent run measured 182.964 ms and 160.673 ms, a
12.18% reduction. The direct schedule comparison, rather than the difference
between those separate processes, is the primary result.

The experiment is not enabled automatically. It doubles the packed
high-water mark and regresses four-thread DGEMM:

| Case | Buffer | Original | Paired batch |
| --- | --- | ---: | ---: |
| SGEMM 4096 | A | about 9.83 MB | about 19.66 MB |
| SGEMM 4096 | B | about 33.55 MB | about 67.11 MB |
| DGEMM 3072 | A | about 6.29 MB | about 12.58 MB |
| DGEMM 3072 | B | about 37.75 MB | about 75.50 MB |

At the time of this experiment the batch packer invoked the scalar pack loop
once per form. It reduced publication barriers from seven to four and improved
C locality, but did not reuse the pair's common input load. The current
implementation retains that path for controls, and adds the fused scalar and
shared-load vector paths described below.

Reproduce the direct comparison with:

```sh
BLIS_SUBCUBER_BENCH_COMPARE=pair \
BLIS_THREAD_IMPL=pthreads BLIS_NUM_THREADS=8 \
./subcuber_bench.x s 4096
```

Use `BLIS_SUBCUBER_BENCH_SCALAR_PACK=1` with this command to reproduce the
historical scalar comparison exactly.

## Follow-up optimization experiments

### SGEMM unit-beta two-C epilogue

The Firestorm 12x8 SGEMM two-C epilogue now dispatches zero, unit, and general
beta separately for both destinations and for both signs of the second
update. The unit path loads C and adds or subtracts the product registers
without multiplying C by one. This optimization is automatic and has no
environment switch.

The focused two-C test passed after this change, and a transient extension
covering an exact unit beta on the second destination passed for both signs.
The persistent focused test now includes that exact second-destination unit
case and passed its final rerun after the latest test hardening. End-to-end
SGEMM measurements were within process-to-process noise, so no standalone
speedup is attributed to this epilogue change.

### Default ARMv8a two-source SIMD packing

The new arm64 packer combines two Strassen sources with NEON and writes the
native packed layout directly. It supports real single precision with 8- and
12-row full panels and real double precision with 6- and 8-row full panels.
Both unit-row-stride and unit-column-stride sources are supported; the latter
path fuses add/sub with transpose/interleave. Scalar code handles K tails.
Short row panels, general strides, unsupported panel sizes, other datatypes,
and non-arm64 builds fall back to the portable packer.

This path is enabled by default. Set
`BLIS_SUBCUBER_DISABLE_SIMD_PACK=1` for the scalar control. The exact-sign
implementation was compared directly with `BLIS_SUBCUBER_BENCH_COMPARE=simd`
in three process launches:

| Case | Three launch changes | Median | Decision |
| --- | --- | ---: | --- |
| SGEMM 4096, 8 threads | {-1.034%, -0.386%, -1.525%} | -1.034% | promote |
| SGEMM 2304, 8 threads | {-1.082%, -1.339%, -1.051%} | -1.082% | promote |
| SGEMM 2304x2304x12288, 8 threads | {-1.782%, -2.882%, -1.711%} | -1.782% | promote |
| DGEMM 3072, 4 threads | {+0.112%, -0.004%, +0.025%} | +0.025% | flat |
| DGEMM 3072, 1 thread | {-1.099%, -1.127%, -0.949%} | -1.099% | promote |

Negative changes are improvements. Four cases improved by about 1--1.8% and
the remaining multithreaded DGEMM case was neutral, so SIMD two-source packing
is now the default.

Reproduce the direct comparison with:

```sh
BLIS_SUBCUBER_BENCH_COMPARE=simd \
BLIS_THREAD_IMPL=pthreads BLIS_NUM_THREADS=8 \
./subcuber_bench.x s 4096
```

### Shared-load paired packing

For each two-form batch, the implementation recognizes the one common source
and the zero or one unique source in each form. One panel traversal then loads
the shared stream once and emits both packed forms. When the complete packed
dimension consists of supported full arm64 panels, paired batching
automatically uses the vector implementation.
`BLIS_SUBCUBER_DISABLE_PAIR_FUSED_PACK=1` restores two ordinary pack
traversals. The paired schedule itself remains opt-in because it still doubles
the packed high-water mark and its historical schedule-only result regressed
four-thread DGEMM.

Before the vector path was added, the scalar single-traversal control measured
the following changes against ordinary paired packing:

| Case | Scalar fused change |
| --- | ---: |
| SGEMM 4096, 8 threads | +1.629% |
| SGEMM 2304, 8 threads | +1.868% |
| DGEMM 3072, 4 threads | +1.108% |

The scalar fusion is therefore retained only for control experiments, forced
with `BLIS_SUBCUBER_EXPERIMENT_PAIR_FUSED_PACK=1` while SIMD and automatic
vector pair fusion are disabled.

The final exact-sign shared-load vector kernel was compared with two ordinary
SIMD traversals using `BLIS_SUBCUBER_BENCH_COMPARE=pair-fusion`:

| Case | Median change | Three-run range | Decision |
| --- | ---: | ---: | --- |
| SGEMM 4096, 8 threads | -1.713% | -3.477% to -1.672% | useful |
| SGEMM 2304, 8 threads | +0.252% | -0.653% to +0.350% | neutral |
| SGEMM 2304x2304x12288, 8 threads | -0.375% | -0.443% to -0.130% | small win |
| DGEMM 3072, 4 threads | -0.392% | -1.042% to -0.080% | small win |
| DGEMM 3072, 1 thread | +0.132% | +0.104% to +0.164% | small regression |

The complete opt-in combination was also compared with the original schedule
and scalar packing using `BLIS_SUBCUBER_BENCH_COMPARE=combined`:

| Case | Median change | Three-run range |
| --- | ---: | ---: |
| SGEMM 4096, 8 threads | -2.919% | -3.763% to -2.239% |
| SGEMM 2304, 8 threads | -0.125% | -0.835% to +0.067% |
| SGEMM 2304x2304x12288, 8 threads | -1.906% | -2.060% to -1.069% |
| DGEMM 3072, 4 threads | -0.369% | -1.084% to +0.741% |
| DGEMM 3072, 1 thread | -1.087% | -1.171% to -1.047% |

Shared-load SIMD fusion is therefore automatic inside paired mode: it has a
meaningful upside on the larger SGEMM cases and safe fallbacks elsewhere.
Paired batching itself remains opt-in because it doubles packed-buffer
capacity and its benefit is workload-dependent.

### Final default versus conventional BLIS

The promoted default is the original seven-generation schedule plus automatic
two-source SIMD packing; it does not enable paired batching. Three independent
launches of the correctness-checking `baseline` comparison produced:

| Case | Median change | Three-run range |
| --- | ---: | ---: |
| SGEMM 4096, 8 threads | -13.266% | -14.344% to -12.410% |
| SGEMM 2304, 8 threads | -4.899% | -5.861% to -4.121% |
| SGEMM 2304x2304x12288, 8 threads | -5.694% | -6.845% to -5.641% |
| DGEMM 3072, 1 thread | -9.490% | -9.569% to -9.478% |

The four-thread DGEMM comparison ranged from -9.231% to -24.630% because the
conventional side was scheduler-sensitive across launches. Every launch
favored SubCuber, but the spread is too wide to publish a precise median as an
optimization claim.

## Validation

The historical scalar paired implementation passed:

- the focused armv8a SGEMM/DGEMM two-C kernel test;
- direct fast testsuite checks at one and eight pthreads;
- the SALT concurrent-caller suite with two threads per caller;
- a generic-context build, forcing portable scratch/scatter;
- an arm64 AddressSanitizer build with PBA pools disabled; and
- debug pack-generation stamps with `MC=24`, `NC=24`, and `KC=16`, forcing
  multiple block generations and parity wraps.

The testsuite result files were checked with `check-blistest.sh` directly.
The make wrapper is not sufficient by itself because this BLIS revision
ignores the checker's exit status.

The repository now also contains `bls_sc_packm_armv8a_test.c.in`, a focused
test for the two-source and shared-load pair packers. It covers float and
double panel sizes, both unit-stride layouts, all sign combinations, singleton
pair forms, K tails, destination guards, and unsupported-input rejection.

After the final exact-sign/default-selection changes, the following all
passed:

- the focused two-C test, including SGEMM unit beta for both destinations and
  both signs of the second update, plus zero-beta NaN sentinels in both the
  arm64 epilogue and portable backstop;
- 1,799 focused packer cases (94,544 packed elements), both optimized and
  under AddressSanitizer/UndefinedBehaviorSanitizer;
- direct fast tests at one and eight pthreads for the default and paired
  paths, including the forced scalar fused-pair control;
- the SALT four-caller suite with two BLIS threads per caller in paired mode;
- default and paired tests in a generic-context build, exercising portable
  packing and scratch/scatter fallbacks; and
- an arm64 AddressSanitizer build with PBA pools disabled, debug generation
  stamps enabled, and `MC=24`, `NC=24`, `KC=16`, for the default, vector-pair,
  and forced scalar-pair paths.

Build the focused pack test from the out-of-tree directory with:

```sh
cc -O2 -Iinclude/arm64 \
  -I/path/to/blis/sandbox/strassen -x c \
  /path/to/blis/sandbox/strassen/bls_sc_packm_armv8a_test.c.in \
  /path/to/blis/sandbox/strassen/bls_sc_packm_armv8a.c \
  -x none lib/arm64/libblis.a -lm -lpthread -o test_packm
./test_packm
```

## Recommendations

1. Hoist the paired kernel's invariant pointers, singleton state, and signs
   out of its vector loops. If assembly still branches per vector, specialize
   only the six operation-pair shapes used by the fixed Strassen batches and
   retain the generic helper for test coverage. This is an opt-in-path
   refinement, not a blocker for the default SIMD promotion.

2. If paired batching proves useful, test one batch slot plus a consumption
   barrier or a panel-local pair buffer to avoid doubling workspace. Small
   groups of I tiles may retain C locality without alternating two large B
   streams at every microtile.

3. Sweep SubCuber MC and explicit thread factorizations. NC and KC are
   settled for the measured squares; MC and the JC/IC/JR division remain
   relevant, especially around the SGEMM 2304 multithreaded crossover.

4. Reduce synchronization and pack imbalance for smaller multithreaded
   cases with panel-level work stealing or a persistent packing pipeline.
   SGEMM 4096 is already close to the one-level arithmetic reduction, while
   SGEMM 2304 still spends a much larger fraction in fixed overhead.

5. Consider a second Strassen level only for 8K-class and larger problems,
   gated so every leaf remains above the measured one-level crossover. This
   has meaningful large-N upside but substantially more scheduling,
   workspace, and numerical complexity.

Do not prioritize more global term permutations or smaller square-case
NC/KC values: these experiments directly reject both directions.

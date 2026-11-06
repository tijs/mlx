# M2 Max long-batched-reduction candidate

## Status and claim boundary

This change addresses the long batched reduction implicated in a reproduced Metal `Impacting Interactivity` failure. The integration decision is to land the locally validated, narrowly gated route and consume it through Swift and Osaurus. Physical M2 Max crash prevention remains unconfirmed; it is not a general inference-speed claim. The retained correctness and regression checks govern landing, and affected-hardware follow-up is recorded separately. No further reporter capture or release is part of this integration.

The candidate is limited to M2 Max. It preserves full context length, masks, softmax, model settings and cache contents. The separate Flint fused-attention failure is not resolved by a matrix-multiplication change.

## What the supplied evidence establishes

| Evidence | Observation | Limit |
|---|---|---|
| Actual Allosaurus app capture | Failed buffer contains only BF16 attention probability × value multiplication; earlier QK, masking and softmax complete | Identifies the operation, not Apple's private driver condition |
| Actual Flint app capture | Failed buffer contains RMS normalization and fused BF16 attention | Does not isolate which of those two operations failed |
| Standalone 96 GB M2 Max return | Long-context Allosaurus-shaped baseline succeeds once, then produces the same `0x0e`; four alternatives complete three evaluations each | Three successes do not establish reliable prevention |
| Standalone 32 GB M2 Max return | All 36 evaluations pass, including unchanged baseline | Shows the workload is not deterministically invalid on every M2 Max |
| Driver logs | Wiring-accounting warnings occur on both machines, including the fully passing run | Warnings alone are not proof of the cause |

The standalone reproduction does not load model weights, restore an SSD cache or run Osaurus's scheduler. Those components are not necessary to trigger this error class. This does not rule out their influencing failure frequency in the full app. Failed app buffers reported approximately 42 ms and 72 ms of GPU time; a five-second buffer execution was not established. Neither an NPU issue nor simple insufficient RAM has been demonstrated.

## Implementation and isolation

The existing SIMD split-K route only accepts a single output matrix. A captured attention workload has `M=512`, `N=256`, `K=115768`, and 24 query heads, so it takes the unsplit batched route.

The new route requires all of:

- Architecture generation 14, variant `s`: M2 Max.
- Non-NAX FP16 or BF16 ordinary Matmul, not AddMM.
- Between 2 and 32 output matrices.
- `K >= 65536`, `K >= max(M, N)`, and the existing small-output tile threshold.

It resolves collapsed broadcast strides on the host, using signed offsets for sliced/reversed batch views. Existing SIMD split-K kernels calculate FP32 partial sums. Existing accumulation kernels combine them into the requested output dtype. One scratch buffer is reused across matrices, protected by the encoder's existing read/write barriers. No Metal kernel signatures, serialized formats or public APIs change.

Partitions target approximately 4096 reduction elements and are capped at 32. This is a dispatch heuristic, not a watchdog deadline; above `K=131072`, partitions can exceed 4096 elements. Existing unbatched selection is preserved. Other architectures, NAX, shorter reductions, batches above 32, GEMV and AddMM retain their original routes. Quantized matmul kernels and cache code are not edited.

The route originally allowed 64 matrices. A performance sweep found a 28–29% slowdown for 64 matrices with 64×64 outputs, in both FP16 and BF16. The route was therefore narrowed to at most 32 matrices. The failed row is retained; it is not counted as a passing optimization.

## Numerical and regression coverage

Focused GPU versus CPU FP32 comparisons cover:

- FP16 and BF16; `K=65536` and `K=65537`.
- Both input transpose flags and all four combinations.
- Right-only, left-only and crossed batch broadcasting.
- Nonzero sliced offsets and reversed batch slices.
- Existing unbatched and batched AddMM with nontrivial alpha/beta.
- Existing basic Matmul contracts.
- Eight outstanding asynchronous submissions, validating scratch lifetime and numerical results.

Captured-shape comparisons use separate original and candidate libraries and compare saved outputs element by element. Relative RMS difference was approximately `0.000128656` (0.013%); maximum absolute difference was `0.0000038147`. This is floating-point reduction-order variation, not bitwise equivalence. It does not replace model-quality qualification.

The focused suite passes 74 assertions across five cases, followed by a separate asynchronous-submission test passing 16 assertions. These are 90 passing assertions across retained runs, not a claim of one combined run. The entire current core and shader build succeeded using Xcode's Metal toolchain. Earlier development-library tests are retained separately and are not substituted for the clean build.

## Performance evidence and limits

All local measurements use a physical M5 Max. Setting `MLX_METAL_GPU_ARCH=applegpu_g14s` exercises dispatch selection; it does not emulate M2 hardware.

- Native M5: eight workload shapes produce bit-identical outputs. The largest primitive median was 51.78 ms original versus 51.91 ms candidate.
- Clean captured-shape fallback attention: warm original 141/138 ms versus candidate 144/135 ms, with 16 MiB additional scratch. This is comparable, not a demonstrated general speedup.
- Wider sweep: 12 shapes × two dtypes × original/candidate × eight evaluations. It includes reduction lengths up to 262145, small/large batches and output tiles, and excluded boundaries. The 64-matrix regression above motivated the narrower route.
- Boundary sweep: batches 16, 24, 32 and 33 with 64×64 outputs improved locally under the original broader candidate. The conservative final cap is 32. Final FP16/BF16 checks at batches 33 and 64 are bit-identical to original with timing ratios 0.999–1.003. The retained 32-matrix row is about 19–20% faster locally; the captured 24-head multiplication is about 7% faster. These are M5 dispatch-override measurements, not M2 speed claims.
- One short excluded workload showed a noisy initial timing outlier. Five alternating-order pairs overlapped, without a consistent slowdown. Both the outlier and follow-up are retained.

These are primitive measurements, not tokens/second, full-model, multi-turn or all-Mac performance certification. No universal zero-regression guarantee is supported. The hardware gate prevents this candidate route from being selected on other Macs.

## Reproduction for a later source review

Use the PR branch and a full Xcode installation with Metal tools available. On the actual M2 Max, do not set an architecture override. Build with CMake in Release mode, Metal enabled, and tests enabled; run the focused cases in `tests/metal_batched_splitk_tests.cpp` plus the existing Matmul case. For local path testing on another supported Mac, an architecture override may exercise the routing, but label that result correctly.

Compare the original base and candidate on the same device, inputs and settings. Alternate order and retain failures as well as successes. A baseline that does not reproduce the error cannot establish prevention. Review long-context app behavior and the separate fused-attention report before declaring the whole issue closed. Do not reset accepted unrelated compaction, template, UI or cache evidence.

## Upstream maintenance

Changes are confined to Metal Matmul host dispatch, focused tests and this document. When upstream adds batched split-K, rebase this small change and retain the transpose, broadcast, sliced-offset and unaligned-K regressions. Reverting the source restores old routing; no persistent settings need migration.

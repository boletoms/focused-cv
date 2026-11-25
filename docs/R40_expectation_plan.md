# R40: expectation and surprise measurements before the expectation focus map (plan, fixed before the run)

Owner (2026-11-21): "run your diagnostic tests and then move to implementing all applicable solutions and fixes before validation testing." Surprise definition (owner, `DECISIONS.md` 2026-11-21): deviation from the expected content and positions of objects and background; surprise only promotes; false positives are preferred to false negatives; P5 training-free. Offline on the DG1 logs (`diag_dg1`, R26 head, LVOS V2 train, all 420 videos) and the frames; CPU; no training. Script `src/stage_b/vos/expect_study.py`; result `r40_expect.json`.

Features of the current frame available before the first merge: `pix` (4 x 4-pooled pixels, before the encoder), `emb` (patch embeddings), `x0` (block-0 attention output); each centred and normalised per frame (`surprise.unit`).

## A. Where is a tracked object now? (the in-focus expectation)
Samples: per video, up to 20 fast pairs (object tracked by the closed-loop map at t − 1, present at t, ground-truth centroid moved more than 2 patches) and 10 other pairs (seeded per video); all-frame figures weight each sample by its type's frequency in the video. The previous patch-level prediction M = the closed-loop map's object probability >= 0.5 per patch at t − 1.

| Candidate | Region |
|---|---|
| A0 (today) | M grown one patch |
| A1 | M with M shifted by its centroid's last displacement, grown one patch |
| A2 (feat) | M with M shifted by d*, grown one patch; d* within 8 patches maximises the mean cosine between M's features at t − 1 and the current frame's at the shifted positions (block matching on the current frame) |
| A3 (feat) | A1 and A2 together |

Measures: share of the object's ground-truth patches (coverage >= 0.25) covered at t, on fast pairs and over all pairs; region size relative to A0 over all pairs. Also, for each candidate, the same with a ring of 1 + ceil(speed) patches (speed: the candidate's displacement in patches) as the middle tier.

**Rule:** the in-focus expectation is the candidate with the highest fast-pair coverage among those at most 1.15 times A0's size over all pairs; a feature-matching candidate is adopted only if it beats A1 by at least 2 points of fast-pair coverage, otherwise A1; between features within 1 point, the cheapest (`pix`, then `emb`, then `x0`).

## B. Does the surprise check catch target arrivals outside the map?
Samples: per video, two runs of 15 consecutive frames (seeded); scored on the last 4 of each. Camera displacement g per frame: the integer shift within 4 patches with the highest mean cosine over all patches. Mismatch e(p) = 1 − cosine between the current feature at p and the previous frame's at p − g (radius 0), or the best within one patch of p − g (radius 1); positions entering from outside the frame count as new content (the border strip, always promoted). Tolerance: the mean and spread of e at p over the previous 10 frames, z = (e − mean) / sqrt(spread^2 + 0.05^2). Scores: e alone, or z.

Events: patches outside today's protected region (the closed-loop map's map grown one patch) with an object's ground-truth coverage >= 0.5 at t that were not that object's (coverage < 0.25) at p − g at t − 1 (the object arrived there). Cost: the share of the other out-of-focus patches (no object, coverage < 0.25) that fire. Static lost targets (the object's at both frames) cannot change and are counted separately (they are the search region's).

**Rule:** the surprise check is the (feature, radius, score) with the lowest cost at a threshold reaching at least 95% event recall (owner); that threshold is the closed-loop setting. A frame whose fired share of out-of-focus patches exceeds 0.5 is a global change: all its out-of-focus patches take the baseline's rules (fixed, not tuned).

## B2. Can a lost target be found inside its search region?
Samples: lost frames of the closed-loop map (object present, IoU < 0.1) within loss episodes with a tracked frame before, up to 15 per video. Search region: a disc around the last tracked predicted centroid moved by its last displacement per lost frame, radius 2 + n patches after n lost frames (R39 Q8: 81% containment). Prototype: mean feature over M at the last tracked frame. Measures: containment of the target's patches by the region; area under the ROC curve of the prototype cosine inside the region; at the threshold reaching 95% recall of the target's patches inside the region, the flagged share of the region.

**Rule:** promote prototype-matched patches inside a lost object's search region to high interest if, at 95% recall, they are at most half of the region (feature chosen as in A); otherwise the search region stays in the middle tier only.

## C. Closed-loop gate and validation (fixed before any closed-loop run)
Setting: `sitome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98@<exp name>` (`src/stage_b/expect.py`; the map region is never merged, the middle tier is merged by ALGM 0.98's rules among itself, the rest by our budget; the fallback, whenever every object that appeared is lost or the map is empty, is plain ALGM 0.98, owner rule), with the expectation, surprise check and prototype settings taken from A, B and B2 by the rules above; safety nets on: speed-scaled ring, lost-object search disc. Deterministic kernels. Dense pass skipped (`VOS_SKIP_DENSE=1`; dense rows exist under the same kernels). Smoke first on one gate video (runs end to end; FLOPs below ALGM's on it are not required).

**Gate** (R26 head, LVOS V2 train; references DG1's rows of the same head and kernels; `src/stage_b/vos/r40_compare.py gate`): (1) on the gate subset (`r40_gate_subset.txt`: the 129 videos whose fast frames are at least 10% of present frames, ground truth only), new − closed-loop map with a lower 95% bound above 0; (2) on the gate subset, new − ALGM 0.98 with a lower bound above −3.0 at lower FLOPs; (3) on all 420 videos, as (2). All three pass: validation. Otherwise: stop and report to the owner. No setting changes after a gate result.

**Validation** (R-25 protocol, `r40_compare.py val`): heads R31c, h2, h3; LVOS V1 val (50 videos) and the 90 V2 valid videos not in V1 val (`v2valid_new.txt`); references: ALGM 0.98 rows of each head under the same kernels (`eval_r31c_lvos_det`, `eval_r38_h{2,3}_lvos`, `eval_*_v2val`). Non-inferiority: lower bound of the head-averaged per-object difference above −3.0 at lower head-averaged FLOPs; superiority above 0.

Amendment (2026-11-21, after the smoke run, before any gate result): the closed-loop inner method is `ALGMTiered` (`titome_...`) in place of `ALGMSurprise` (`sitome_...`): every token outside the high tier is merged by ALGM's own rules and the low tokens also by our budget (a scattered middle tier was almost never merged under ALGMSurprise). Gate and validation rules unchanged.

## D. Full resolution in the map against the baseline's rules (owner question 2026-11-21, fixed before these runs)
Owner: "do we have evidence to show that running high interest areas/regions with full resolution is better than using the baselines?" Evidence so far: short clips (R6, R10), the map at full resolution with ALGM elsewhere holds dense-level accuracy at about 0.36 FLOPs where plain ALGM needs 0.57 or more; long videos (R32, LVOS val, one draw under default kernels, before the expectation focus map), ALGM's rules inside the map cost −5.55 [−10.28, −0.98] against keeping it whole. Runs (R26 head, LVOS V2 train, all 420 videos, deterministic kernels, after the gate verdict and before validation):
- D1 `palgm_t0.98_dil1~algm_t0.98` (map at full resolution, ALGM's rules everywhere else) against plain ALGM 0.98 (DG1 rows): the general question, nothing else changed. Reported (no decision attached).
- D2 `titome_t0.98_l0.7_floor64_dil1_cls64_ha1~algm_t0.98@exp-pix-mot-rl-r0-e0.24-ring-srch` (the expectation focus map with the high tier also merged by ALGM's rules) against the expectation focus map (gate rows). Rule: if D2 − expectation focus map has a lower 95% bound above −1.0 J&F at FLOPs at least 10% lower, the high tier's treatment goes to the owner as a design change (validation waits; a new gate would be needed); otherwise full resolution stays and validation proceeds.

## E. Second baseline: ToMe (owner 2026-11-21, "i accept the recommended second baseline"; fixed before these runs)
After the ALGM validation, with the design unchanged: `ttome_r64_l0.7_floor64_dil1_cls64~tome_r64_from1_noprop@exp-pix-mot-rl-r0-e0.24-ring-srch` (`baselines.ToMeTiered`: the map never merged, every other token by ToMe r64's rules as protected ToMe, the low tokens also by our budget; fallback plain ToMe r64), against plain `tome_r64_from1_noprop` on the same videos and heads. Validation as section C (heads R31c, h2, h3; LVOS V1 val and the 90 new V2 valid videos; head-averaged per-object difference; non-inferiority above −3.0 at lower FLOPs, superiority above 0); the plain ToMe rows run in the same jobs (same kernels). DAVIS val short-clip check as for ALGM (R26 head). No separate gate: the design and its settings are fixed by sections A to C.

## F. After the gate: the cause of frame-wide surprise (owner 2026-11-21 accepted the recommendation: fix false frame-wide surprises at their source, then one more gate run; fixed before these measurements)
B3 (`expect_camera.py`, `r40_camera.json`): whole-frame sub-pixel camera compensation does not reduce frame-wide surprise (global-frame share 0.265 against 0.257 for the integer shift; cost share 0.312 against 0.299 at 95% recall), so camera misalignment is not the cause. B4 (`src/stage_b/vos/expect_floor.py`, `r40_floor.json`; same samples, events and costs as B, feature `pix`, the closed loop's integer camera shift) tests the two remaining training-free explanations: per-patch normalisation amplifying the noise of flat or blurred patches (a variance floor sigma in {0, 0.01, 0.02, 0.05, 0.1}, sigma 0 being the current check), and local motion of a few pixels (tolerance s in {0, 1, 2} offsets of 4 pixels). Secondary events: arrivals outside the expectation focus map's expected regions (A3 core grown one patch with the speed ring, from the closed-loop map's predictions).
**Rule:** per candidate, the threshold is the highest reaching 95% recall on both the B events and the secondary events (false negatives worse than false positives, owner); the score is the expected middle-tier share of out-of-focus patches (per frame: 1 if the fired share exceeds 0.5, the global rule, unchanged; otherwise the fired share). The candidate with the lowest score is adopted if it is at most 0.8 times the current check's; otherwise the check stays and the result goes to the owner. If adopted, the gate (section C, rules unchanged) runs once more with it, recorded as the second gate run on LVOS V2 train; validation only if it passes.

## G. The high tier under the baseline's rules at the final setting, validation protocol (owner question 2026-11-21 after the validation, "have we tested using the baseline instead of full resolution in high interest areas?"; fixed before these runs)
Section D2 answered it on LVOS V2 train with the first surprise check (one head): −2.47 [−3.50, −1.56] J&F for 23% fewer FLOPs, so full resolution stayed. Run again with the final setting on the validation protocol: `titome_t0.98_l0.7_floor64_dil1_cls64_ha1~algm_t0.98@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1` (everything as the expectation focus map, the high tier also merged by ALGM 0.98's rules), heads R31c, h2, h3, LVOS V1 val and the 90 new V2 valid videos, deterministic kernels (`eval_r40d2_*`; `r40_compare d2val` → `r40_d2val.json`). Measures: head-averaged per-object difference against the expectation focus map (its validation rows) and against ALGM 0.98, with FLOPs. **Rule:** if the difference to the expectation focus map has a lower 95% bound above −1.0 J&F at FLOPs at least 10% lower, the high tier's treatment goes to the owner as a design question (any change would need a new gate and a fresh held-out test, since this split has been used); otherwise full resolution in the map stands (closed by result). The difference to ALGM is reported.

## H. Where the validation gain comes from (review flag R-33; owner "continue"; reported, no rule; fixed before these runs)
Logged rerun (`VOS_DIAG=1`, deterministic kernels) of dense, ALGM 0.98 and the expectation focus map with head R31c on LVOS V1 val and the 90 new V2 valid videos (`eval_r40_attr_r31c_*`; `src/stage_b/vos/r40_attr.py` → `r40_attr.json`). Check first: the logged rows equal the validation rows (R31c expectation focus map `eval_r40_r31c_*`; ALGM and dense `eval_r31c_lvos_det`, `eval_r31c_v2val`) to 1e-6 per object. Decomposition (exact): each object's J&F difference is the sum over frame classes of (share of its scored frames) x (mean per-frame difference in the class). Classes from the expectation focus map's own state: (a) frames before the video's first fallback frame, (b) fallback frames (every object lost, plain ALGM 0.98 runs), (c) non-fallback frames after a fallback episode (after re-acquisition). Per class: frame share, mean per-frame difference against ALGM 0.98 and against dense, and the contribution to the object-mean difference with a 95% bootstrap interval over objects. Also, descriptively (classes defined on outcomes): the cross-table of per-frame loss (J below 0.1) under the expectation focus map and under ALGM 0.98 with each cell's share and contribution.

## I. ToMe arm, second version: ToMe's rule as a rate on the tiers (owner 2026-11-21, "continue with recommended", after the ToMe arm failed section E; review flag R-34; fixed before these runs)
Cause addressed: ToMe merges a fixed r = 64 tokens per block; in the arm these come from the tokens outside the high tier, and once our budget has brought the low tokens to their floor they fall mostly on the middle tier, which collapses (ALGM's threshold rule does not depend on the pool). Change (`ttome_..._tr1`, `baselines.ToMeTiered(as_rate=True)`): in block b, the arm merges round(rho_b x n_b) tokens outside the high tier (n_b: those tokens now), where rho_b = r_b / N_b is the share plain ToMe r64 merges at that block (N_b: plain ToMe's token count at block b from the frame's patch count, r_b = min(64, N_b // 2)); matching, size-weighted means and our budget on the low tokens unchanged; plain ToMe starts at block 0 (`tome_r64_from1` is 1-based), as the arm. This adapts ToMe's rule to a subset (the authors' code has no subset form); without a high tier and without our budget it equals plain ToMe's count. Everything else as section E: `ttome_r64_l0.7_floor64_dil1_cls64_tr1~tome_r64_from1_noprop@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1`.

**Gate** (R26 head, LVOS V2 train, deterministic kernels; `r40_compare tome_gate` → `r40_tome_gate.json`): smoke on PEyHFztC first (runs; FLOPs below plain ToMe's); plain ToMe r64 rows run in the same jobs; the closed-loop map (`ftome_...~tome_r64_from1_noprop`) from DG1. (1) gate subset: new − closed-loop map, lower bound above 0; (2) gate subset: new − plain ToMe r64, lower bound above −3.0 at lower FLOPs; (3) all 420 videos: as (2). All pass: validation; otherwise stop and report (the other queued runs continue).

**Validation** (second attempt on this split for the ToMe arm, review flag R-34): section E's protocol (heads R31c, h2, h3; LVOS V1 val and the 90 new V2 valid videos; plain ToMe r64 rows of section E as references; `eval_r40f_*`; `r40_compare tome2` → `r40_tome2_val.json`): non-inferiority if the head-averaged lower bound is above −3.0 at lower FLOPs, superiority above 0. Both attempts are reported.

## J. Evidence for the review package (owner 2026-11-21, "i agree"; the package goes without the ToMe arm, whose runs follow; fixed before these runs; reported, no rules)
1. **Short clips inside Cutie with every baseline** (DAVIS val, 30 videos, one run per backbone so all rows pair; fast path, bf16 SDPA attention as the expectation focus map's evaluator; deterministic kernels): ViT-S, R26 head, R26's set (AiluRus 0.39 / 0.5 at 3, ALGM 0.95 / 0.98, ToMe r64 / r80, protected ALGM 0.77 / 0.83 / 0.91, closed-loop map rate 0.3 / 0.7) plus the expectation focus map (`eval_r40_davis_all`); ViT-B, R25 head, R25's set (closed-loop map rate 0.7 / 0.3 at floor 128 and margin 2, protected ALGM 0.83 / 0.95, ALGM 0.95 / 0.97, ToMe r64, AiluRus 0.5 at 3, PiToMe 0.97, Expedite 0.49 at 5) plus the expectation focus map for ViT-B (`eval_r40b_davis`). Reported as R10 and R25: change from dense and paired differences against each baseline, with FLOPs. The older short-clip tables used eager fp32 attention (R29: baselines move by at most 0.11 J&F between the paths, the closed-loop map by up to 0.54).
2. **ViT-B long videos** (R25 head; LVOS V1 val and the 90 new V2 valid videos; deterministic kernels; `eval_r40b_v1val`, `eval_r40b_v2new`): dense, plain ALGM 0.95 (R25's rule: DAVIS FLOPs closest to 0.44), and the expectation focus map with each component at this backbone's earlier choice, `titome_t0.95_l0.7_floor128_dil2_cls64~algm_t0.95@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1` (ALGM threshold from R25's rule; budget floor and margin from R4/R25's ViT-B choice; expectation and surprise settings are pixel-based and unchanged). One head: per-object difference against ALGM 0.95 and dense with intervals and FLOPs (the R-25 protocol needs three heads; reported, no criterion).
3. **Wall clock** (GPU otherwise idle): per-frame encoder time at batch 1 of the expectation focus map (expectation and surprise check included), today's design, ALGM 0.98, ToMe r64 and dense, ViT-S, on DAVIS val frames with closed-loop maps, following R8's protocol (warm-up, medians); eager for every method, and CUDA graphs where a method allows them.
3b. **Where the expectation focus map's wall clock goes** (added 2026-11-21 after J3 timed the expectation focus map at 10.46 ms per frame against dense 3.17 eager at ViT-S; `src/stage_b/vos/r40_timing_parts.py` → `r40_timing_parts_vits.json`, `r40_timing_parts_vitb.json`; reported, no rule). Same frames, maps and protocol as J3, ViT-S and ViT-B (R25's ViT-B setting as J2). Per expectation focus map frame, CUDA events split the step into the run-time state update outside the encoder (map motion, object states), the encoder call, and inside it the expectation step (expectation map, surprise check, tiers); the host clock gives the time until the step returns before the GPU is waited on (host time close to the total: bound by Python and kernel launches). Beside it: dense (eager, graph), today's design (eager, graph), the backbone's ALGM fallback (eager), and per-frame GPU kernels of dense and the expectation focus map (eager, PyTorch profiler). Runs right after the ViT-B long-video rows on the idle GPU.

## K. The expectation focus map under CUDA graphs without a method change (owner 2026-11-21, "continue with recommended", after R-35; addendum after the review package; fixed before these runs)
Implementation only (`src/stage_b/tiered_graph.py`; the method, its settings and every rule are unchanged). (1) The tier decision moves in front of the encoder: the expectation step uses pixels and the previous masks only (`exp-pix-...`), so running it before block 0 gives the same tiers as running it after block 0's attention, where it sits now. (2) ALGM's merges (layer-1 windows, layer-5 matching, both outside the high tier) and our budget on the low tier run on fixed-capacity token buffers: the real token count is a device value, tokens past it are padding that attention masks out (exactly zero weight) and that no merge can select; the same tokens merge in the same order as now. (3) The forward pass is cut after each block's merge; the host reads the new count once per block and replays the graph captured for (block, capacity class), capacity rounded up to a multiple of 32 tokens. The fallback (plain ALGM on frames where every object is lost) runs through the same code with no high tier and no budget. Counted FLOPs are unchanged by construction; FLOPs at the padded capacity are reported beside them.
- K1 (CPU, fp32, tests): the tiers planned before the encoder equal the current ones exactly; per block, the padded merges give the same assignment of original patches to tokens as the current code and features, sizes, positions and interest within 1e-5, on synthetic tokens and DAVIS frames, including frames with no ALGM merge, no budget merge and the floor reached; the whole padded forward (eager, no graphs) matches the current forward within 1e-4 on DAVIS frames.
- K2 (idle GPU): graph replay equals the padded eager forward within bf16 tolerance (fast path) and within 1e-4 (fp32 attention path) on DAVIS frames. (Corrected before any run: the long-video rows were produced on the fast path, so K3 compares on the fast path for both datasets; the padding mask moves SDPA from the flash kernel to a masked kernel, a rounding-level change; if K3 fails on that account, the variable-length flash kernel with device-side lengths is the next implementation step.)
- K3 (idle GPU; equivalence, rules fixed here): (a) DAVIS val, fast path, ViT-S (R26 head) and ViT-B (R25 head), the expectation focus map with graphs against its current rows (`eval_r40_davis_all`, `eval_r40b_davis`): passes if the 95% bootstrap interval of the per-object difference lies within ±0.3 J&F and counted FLOPs agree within 0.002, per backbone. (b) Long videos, head R31c, LVOS V1 val and the 90 new V2 valid videos (239 objects), the long-video evaluator's path (fast bf16 attention, which `eval_lvos.py` always uses; deterministic kernels), graphs against the validated rows (`eval_r40_r31c_v1val`, `_v2new`): passes if the interval lies within ±1.0 J&F and counted FLOPs agree within 0.002. Deterministic reruns of the eager rows are bitwise equal (section H's check), so any difference comes from the graph path; the closed loop can amplify rounding on long videos, hence the wider margin. A failure is an implementation defect to find and fix (owner rule), not a reason to change the method; every attempt is reported.
- K4 (idle GPU; reported, no rule): wall clock as J3 and J3b with the graph path, ViT-S and ViT-B, beside dense with graphs.
- K2, second attempt (2026-11-21, recorded before the next run): graph capture worked and graph replay equalled the padded eager path exactly (difference 0.0) on both backbones and both attention paths, which is K2's rule above. The check script had added a condition that is not in this plan (the padded path's patch assignment equal to the current path's on 90% of frames); it failed (0 to 1 of 54 frames on the GPU, against every frame on the CPU in K1), and the chain skipped K3 and K4. The container forces TF32 for fp32 matmuls (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, in every evaluation so far), whose rounding depends on matrix shape, and padding changes shapes; merge decisions near a threshold can then flip. From now on the script applies K2's rule as written; the agreement with the current path is reported. K2b (diagnostic, reported, no rule): the ViT-S check rerun with TF32 off; if the disagreement is matmul rounding, agreement rises. K3 keeps its rules and remains the test of whether the difference matters for accuracy.
- After K3's first attempt (2026-11-21, recorded before any further K run): K2 passed by its rule; K2b refuted TF32 as the cause; K3 failed (results file, section K). The diagnostic and the rounding control locate the cause in bf16 merge scores under autocast, shared by every method, which turn any kernel or layout change into different merges; the closed loop then gives object-level differences as large as the baseline's own kernel noise (ALGM 0.98: mean absolute 4.53 per object between default and deterministic kernels). No further K3 rerun on the current graph path (it cannot pass without bitwise agreement, which padding does not allow). Proposal to the owner, not adopted: the accuracy of the timed implementation is measured directly: once the implementation is final for speed, the validation protocol (section C's rules; heads R31c, h2, h3; LVOS V1 val and the 90 new V2 valid videos) and the DAVIS rows run on it, and those rows carry the claim; the eager rows are reported as a second draw.
- Owner (2026-11-21) adopted the proposal above: the final timed implementation is validated directly (section C's rules; heads R31c, h2, h3; LVOS V1 val and the 90 new V2 valid videos; DAVIS ViT-S and ViT-B; ViT-B long videos with the R25 head as J2), and those rows carry the claim; the eager rows are a second draw.

## L. Wall clock without a method change (owner 2026-11-21: "check for any additional optimization avenues and stratergies to reduce wall clock time. you can alter implementation but not design and functionality. if there is no stone left unturned, I will accept the final performance results for the publication"; fixed before these runs)
Implementation only: the method, its settings, its outputs (tiers, merges, the fallback decision) and its rules stay as they are; rounding may change (R-37), which is why the final implementation is validated directly (owner decision after K3). Avenues, each measured and kept if it lowers the wall clock:
1. No host reads inside a frame except one decision per frame (main or fallback) and one read of the per-block token counts at its end: the state update (`objects_from_predictions`), the map motion (`shift_by_velocity`) and the expectation step run on device values (no `item`, `nonzero`, `unique`, Python branches on device values or host-to-device copies; variable growth radii by Chebyshev distance; per-object branches by selection), so each captures in a CUDA graph.
2. Encoder capacities chosen before the frame from the previous frame's token counts plus a margin; the device records each block's count; a frame whose count exceeds the chosen capacity at any block is recomputed with per-block reads (K's runner), so outputs never depend on the guess.
3. Fewer GPU kernels per block: merge bookkeeping in fewer operations (one scatter for features, positions, interest and sizes), fused element-wise work (`torch.compile` or hand-written kernels) where outputs are unchanged up to rounding.
4. No work whose result is unused by the method at the adopted setting: prototypes (`proto` 0) and, on fallback frames, the similarity product, camera estimate and mismatch (score `e`: no history is read); counted FLOPs then report the work done.
5. FLOPs bookkeeping and diagnostics out of the timed path (computed from the per-block counts after the frame).
Rules:
- L1 (CPU, fp32, tests): the device state update, map motion and expectation step equal the current code exactly (object states, the all-lost decision, the map, tiers, diagnostics) on DAVIS sequences and synthetic cases (lost objects and search regions, empty templates, new content at the border, the frame-wide rule, the first frame, fallback frames); the frame runner gives the same tiers, token counts, patch assignments and counted FLOPs as K's runner, including frames whose capacity guess is exceeded.
- L2 (idle GPU): graph replay equals the same implementation run eagerly on DAVIS frames (fast path within 1e-3 relative, as K2); agreement with the current path is reported (R-37).
- L3 (idle GPU, reported): wall clock as J3 and K4 (ViT-S and ViT-B; map motion, state update, tier decision and encoder per frame) beside dense with graphs, with kernel counts and GPU busy time, after each avenue.
- L4: the validation of the final implementation (owner decision after K3; rules of section C and the K addendum).
- L1 amended before any run (2026-11-21): centroids in the device state update are exact (integer sums) where the current code rounds a float32 mean, so object speeds and velocities agree within 1e-5 patch, not bitwise; every mask, loss decision, count, map and tier agrees exactly. The size-class promotion keeps the current `topk(k)` with k read on the host together with the frame's decision (a fixed-size `topk` took different equal-score patches).
- ToMe arm after its second gate (owner 2026-11-21, recommended option): logged rerun of the five loss videos (`r40_tome_lossdiag.txt`; R26 head, LVOS V2 train, plain ToMe r64 and the second version, `VOS_DIAG=1`, deterministic kernels; `eval_r40f_lossdiag`): where and why the object is lost; reported, no rule; a design proposal to the owner follows.
- L amendment after L4 (2026-11-21, R-38; recorded before the rerun): compiled code must keep the eager path's rounding of bf16 intermediates. Without it Inductor keeps fp32 values inside fused kernels, so ALGM's threshold merges shift systematically (ViT-S token diagnostic `r40_wallclock_tokens_vits.json`: compiled segments +26 tokens per frame, 75 frames more against 29 fewer; plain ALGM on the same implementation +120; with Inductor's `emulate_precision_casts` −14 and −51, as the uncompiled runner's symmetric spread of −9 to −13), which section L excludes (merges unchanged up to rounding). The final implementation is the section L form with `emulate_precision_casts=1` (`R40_INDUCTOR_OPTS`). L4b: L2 on both backbones (guard), then the validation, DAVIS and ViT-B long videos rerun on it (`eval_r40m_*`, `r40_compare final eval_r40m final_emu`, `final_short eval_r40m final_emu`) under section C's rules; its rows carry the claim and its timing (`_emu` rows) replaces the section L timing; the L4 rows are reported as the record of the shift.

- L amendment after L4b (2026-11-22, R-38 residual; recorded before the rerun): the L4b rows still differ from the eager rows in one direction at ViT-S. Per video, counted FLOPs are below the eager rows' on 22 of 30 DAVIS videos (26 of 30 above before the fix) and on 300 of 420 head-video pairs of the validation; ViT-B DAVIS 15 above and 15 below. The token diagnostic shows the same lean in the uncompiled runner, from block 4 (ALGM's layer-5 matching). Cause, found in code review: the current path's threshold `node_max.mean() + node_max.std() / layer` rounds the mean and the std to bf16 (autocast keeps their dtype), while the graph runner builds both from `sum` and `pow`, which autocast runs in fp32, and rounds once at the comparison. On 2,000 synthetic draws the two thresholds differ on 24.5%. A threshold one bf16 step away moves every token tied at that value, and the direction depends on the score distribution, so on the backbone. Section L excludes this (merges unchanged up to rounding; the amendment above: keep the eager path's rounding of bf16 intermediates). The final implementation adds `tiered_graph._algm_thr`, the current path's roundings (equal to its threshold on 2,000 of 2,000 draws; test `test_algm_threshold_rounds_as_current_path`). L4c: L2 on both backbones (guard; `~/ml_work/gpuq/10_thr_check.sh`), the token diagnostic (rows `_thr`, reported) and the timing (rows `_thr`, replacing the `_emu` rows; `11_thr_time.sh`), then the validation, DAVIS and ViT-B long videos rerun on it (`eval_r40n_*`, `r40_compare final eval_r40n final_thr`, `final_short eval_r40n final_thr`; `12_thr_val.sh`) under section C's rules. Its rows carry the claim; the L4 and L4b rows are reported as the record.
- L amendment after L4c (2026-11-22, R-38 third cause; recorded before the rerun): the L4c rows agree with the eager rows on short clips at both backbones (per-video FLOPs above / below: ViT-S 16 / 14, ViT-B 17 / 13; tier shares within 0.0005), but on long videos their middle and high tiers are larger (head-averaged, +1.7 and +0.6 points of patches at ViT-S, +1.9 and +0.6 at ViT-B, intervals above 0), carried by a few videos with equal J&F and equal fallback frames (7K7WVzGG: +32 points of middle tier at all three heads, and in L4 and L4b). A replay of the ground-truth masks through both paths (`src/stage_b/vos/r40_replay.py`; CPU, fp32) reproduces it on 7K7WVzGG from frame 98: an object lost for 83 frames has a search region (its last mask moved by n times its last velocity, grown by 2 + n patches) moved out of the grid, so empty; `expect_dev.grow_dev` marks rows without a pixel by the distance H + W + 1, which a radius of 2 + n reaches after H + W − 1 frames of loss, and then grows the empty mask to the whole grid (the current path's `grow` keeps it empty). The whole grid becomes the search zone: middle tier on low patches, surprise patches promoted to high. Long losses occur on long videos only. Fix (implementation only): the radius clamped to H + W, which already covers the grid (test `test_shift_and_grow_match_current`, which the old form fails). The per-video FLOPs sign of L4c at ViT-S long (176 above, 244 below) is the frame runner not computing the expectation products on fallback frames, where the current path computes and counts them (`test_frame_runner_matches_current`); with them added back, 228 / 192 at ViT-S and 75 / 64 at ViT-B (reported). L4d: the replay on all 140 long videos (reported), L2 on both backbones (guard; `~/ml_work/gpuq/13_fix_check.sh`), the validation, DAVIS and ViT-B long videos rerun on the fixed implementation (`eval_r40o_*`, `r40_compare final eval_r40o final_fix`, `final_short eval_r40o final_fix`; `14_fix_val.sh`) under section C's rules, and the timing (rows `_fix`, replacing the `_thr` rows; `15_fix_time.sh`). Its rows carry the claim; the L4, L4b and L4c rows are reported as the record.

## M. ToMe arm, third version (owner 2026-11-21, "continue with recommended", after the loss videos; review flag R-34; fixed before these runs)
The second version (section I) with two additions, each aimed at one failure mode of the loss videos (results file, "the loss videos"); everything else unchanged (setting `ttome_r64_l0.7_floor64_dil1_cls64_tr1~tome_r64_from1_noprop@membase@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1-mb`):
1. Motion beyond the expectation's reach (`-mb`): on a frame where a tracked object's displacement found by block matching reaches the search radius in either axis (|dy| or |dx| = `r_obj`, 8 patches: the true motion may lie beyond the window) or its speed from the last two predictions reaches 8 patches per frame, the frame runs the baseline itself (plain ToMe r64 with the input a fallback frame gets) in place of the tiered method; the expectation state is kept as on any frame. The threshold is the design's own block-matching radius, not a tuned value.
2. Memory written from the baseline's features (`@membase`): frames on which Cutie writes working memory (every `mem_every` frames and whenever a mask is given) run the baseline (the fallback) in place of the tiered method, so the memory the fallback later reads was written from the same encoder.
Counted FLOPs include these frames. Reported with the rows: frames switched by motion, memory frames switched.
- Smoke (PEyHFztC, R26 head): runs; FLOPs below plain ToMe's.
- Gate (R26 head, LVOS V2 train, deterministic kernels; plain ToMe r64 rows of section I's gate run `eval_r40f_gate`, same head, videos and kernels; the closed-loop map from DG1; new rows `eval_r40g_gate`, all 420 videos in one run): section I's criteria: (1) gate subset: new − closed-loop map, lower bound above 0; (2) gate subset: new − plain ToMe r64, lower bound above −3.0 at lower FLOPs; (3) all 420 videos: as (2). All pass: validation; otherwise stop and report.
- Validation (third attempt on this split for the ToMe arm): section E's protocol (heads R31c, h2, h3; LVOS V1 val and the 90 new V2 valid videos; plain ToMe r64 rows of section E as references; `eval_r40g_*`; `r40_compare tome3` → `r40_tome3_val.json`): non-inferiority if the head-averaged lower bound is above −3.0 at lower FLOPs, superiority above 0. All three attempts are reported.
- Disclosure (R-30, R-34): both additions come from loss videos of the gate subset, so the gate is not independent of the design; the validation on held-out videos decides.

## N. ViT-L inside Cutie (queue row R34; owner 2026-11-21, "ViT-L as an addendum"; R34's evaluations waited for the expectation focus map; fixed 2026-11-22 before any R34 evaluation; reported, no rules)
R34's head (stage 2 done). Sections J1 and J2 at ViT-L, the expectation focus map on the final implementation as L4d:
1. Short clips (DAVIS val, 30 videos; fast path, deterministic kernels, as J1): the R34 row's set (dense; closed-loop map rate 0.7 and 0.3 at floor 128 and margin 1, R4's ViT-L choice; protected ALGM 0.83 / 0.95; ALGM 0.95 / 0.97; ToMe r64; AiluRus 0.5 at 4; PiToMe 0.97; Expedite 0.49 at 9) in `eval_r34_cutie`; the plain ALGM threshold by R25's rule (the one of 0.95 / 0.97 whose DAVIS FLOPs are closest to 0.44, `r34_algm.txt`); R25's readings (`r25_compare r34`).
2. The expectation focus map at ViT-L with each component at this backbone's earlier choice (as J2): `titome_t<A>_l0.7_floor128_dil1_cls64~algm_t<A>@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1`, A the threshold of step 1 (`r40_compare.SL_FINAL`); expectation and surprise settings pixel-based and unchanged. Final implementation (L4d's evaluator settings). Guard: L2 at ViT-L (`r40_wallclock check vitl L_csvo`, R40_TAG=fix); the expectation focus map's ViT-L rows run only if it passes.
3. Long videos (LVOS V1 val and the 90 new V2 valid; deterministic kernels): dense and plain ALGM A on the long-video evaluator (`eval_r34_v1val`, `_v2new`); the expectation focus map on the final implementation (`eval_r40ol_v1val`, `_v2new`) and its short-clip row (`eval_r40ol_davis`). One head: per-object differences against plain ALGM A and dense with intervals and FLOPs, per split and pooled (`r40_compare vitl` → `r40_vitl.json`; the R-25 protocol needs three heads, so no criterion). The R34 row's earlier long-video list (the closed-loop map with the ToMe fallback, the C2 detector, ground-truth maps, protected ALGM, the R35 form) is replaced: the expectation focus map is the method of the long-video claim and the generic detector is retired (owner 2026-11-20).
4. Wall clock at ViT-L (L3's protocol): dense, the expectation focus map and plain ALGM A on the final implementation, default and deterministic kernels (`r40_wallclock time vitl`).
Jobs `~/ml_work/gpuq/20_r34_davis.sh` to `23_r34_time.sh` (chain `~/ml_work/r40w_chain.sh`).
Amendment (2026-11-22, before any section N long-video row): the long-video rows run on all 140 LVOS V2 valid videos in one run per method with masks saved (section O); the first run of step 3, which saved no masks, was stopped after the V1 val baseline rows and is not used.

## O. Official metrics (owner 2026-11-22, "continue with recommended"; fixed before these runs; reported, no new rules)
Every J&F so far comes from our scorer (`p2_davis.j_and_f`: every annotated frame after the object's first appearance, our boundary thinning). The claim rows and the short-clip tables are rerun with prediction masks saved (`VOS_SAVE_MASKS=1`, `vos/masks.py`; reads only) and scored by the official toolkits, used unchanged (`vos/official_eval.py`; image `ml-base:vos-eval` adds OpenCV): LVOS (LingyiHongfd/lvos-evaluation @ 25cb02d, semi-supervised, on LVOS V2 valid, whose frames and annotations equal LVOS V1 val's for the 50 V1 videos byte for byte, so the 140 videos of the protocol are exactly LVOS V2 valid; seen / unseen as the toolkit) and DAVIS 2017 val (davisvideochallenge/davis2017-evaluation @ ac7c43f).
- Rows: ViT-S heads R31c, h2, h3 (dense, ALGM 0.98, the expectation focus map on the final implementation; `eval_om_<head>_base`, `_r40`); DAVIS ViT-S (J1's set, R26 head, `eval_om_davis_all`; the expectation focus map on the final implementation, `eval_om_davis_r40`); DAVIS ViT-B (J1's set, R25 head; `eval_omb_davis_*`); ViT-L as section N (`eval_oml_*`); ViT-B heads of section P.
- Checks: with deterministic kernels the reruns should reproduce the claim rows; per object our scorer's J and F on the rerun are compared with the stored rows (reported: share equal, largest difference). If they differ, the rerun rows are a new draw and every table reports the rerun with both scorers.
- Reporting: section C's comparison (head-averaged paired difference against ALGM 0.98, intervals, FLOPs) under the official metric beside our scorer's; per split and seen / unseen; Tables 3 and 4 under the official metric (`r40_compare official` → `r40_official.json`). Section C's verdict stays the one fixed on our scorer; the official metric is reported, not a second rule.

## P. ViT-B three-head protocol (owner 2026-11-22, "continue with recommended"; fixed before these runs)
Heads r41_b1, r41_b2, r41_b3: R31c's recipe on the R25 head (fine-tuned 10,000 iterations, batch 2, dense), Cutie seeds default, 2 and 3, as R31c, h2 and h3 on R26. Rows on all 140 LVOS V2 valid videos with masks saved: dense and plain ALGM 0.95 (R25's rule) on the long-video evaluator, the ViT-B expectation focus map (J2's setting) on the final implementation (`eval_omb_b<k>_*`). Section C's rules against ALGM 0.95: head-averaged per-object paired difference, non-inferior if the lower 95% bound is above −3.0 at lower head-averaged FLOPs, superior if above 0; our scorer decides as in section C, the official metric reported beside it (`r40_compare vitb3` → `r40_vitb3.json`). J2's one-head rows stay as reported.
Jobs `~/ml_work/gpuq/24_om_vits` to `28_vitb_heads_eval` (chain `~/ml_work/r40w_chain.sh`).

## Q. I-frames for the expectation focus map (owner 2026-11-23, after both audits; fixed before any run)
Context: the closed-loop design had full resets (dense I-frames, R13 and R15); the expectation focus map has only the
uniform reset (the fallback) and the partial one (the frame-wide surprise rule), never a full-fidelity frame. The audits
found the gain over ALGM near 0 on videos longer than 600 frames and on videos with frequent fallback. Variants, each
the final setting (`S_FINAL`) plus one suffix, on the reference implementation (dense frames run a passive observer, so
the expectation sees every frame; I-frames count dense in FLOPs):
- Q1 `@key20`, Q2 `@key30`: a dense I-frame every 20 or 30 frames.
- Q3 `@lostkey`: one dense I-frame on each frame where an object has newly become lost.
Development (R26 head, LVOS V2 train, deterministic kernels, paired per object against the method's rows on the same
videos and kernels, `eval_r40_gate2`): the 44 videos longer than 600 frames (`r40_q_long.txt`) and 60 of the others
drawn with seed 0 (`r40_q_sample.txt`). A variant is a candidate if (1) on the long videos its difference against the
method is at least +1.0 J&F (point estimate), (2) on the sample its lower 95% bound is above -1.0, and (3) its FLOPs on
all 104 videos stay below plain ALGM 0.98's on the same videos (`diag_dg1`). The candidate with the largest long-video
difference is chosen (ties: lower FLOPs). No candidate: reported as negative, no held-out run.
Held-out (only for the chosen variant): heads R31c, h2, h3 on the 140 videos, reference implementation, paired against
the method's reference rows (`eval_r40_<head>_*`, the same implementation) and against ALGM 0.98; reported on all
objects, on objects in videos longer than 600 frames and on the 90 new videos. Status: a reported ablation. The
held-out videos were already used for this method, so the preprint's method and claim do not change; adoption needs
fresh data in a later version.
Amendment (owner 2026-11-23, 17:30, before any section Q run): is a fallback frame (plain ALGM 0.98, the uniform reset)
enough as the reset, or does it need full resolution? Two more variants: Q4 `@fbkey20` (every 20th frame runs the
fallback) and Q5 `@lostfb` (the frame where an object is newly lost runs the fallback). Same development videos, rules
and selection over all five variants; the gate also reports each dense variant against its fallback twin (Q1 against
Q4, Q3 against Q5) on the long videos and in FLOPs. Reading fixed in advance: the fallback suffices if its twin's
long-video difference against the method is within 0.5 J&F of the dense variant's (point estimates) at lower FLOPs.
Amendment 2 (owner 2026-11-23, 17:40, before any section Q run): add Q6 `@key10` and its fallback twin Q7 `@fbkey10`
(at LVOS's 6 frames per second, 10 frames is about 1.7 s, the order of a streaming codec's I-frame interval). Same
rules; the gate reports the pair Q6/Q7 beside the others. Run order inside the gate: key20, lostkey, fbkey20, lostfb,
key10, fbkey10, key30.

## R. Cutie's memory on dense I-frames (owner 2026-11-23, 17:45; fixed before any section R run)
Question: an I-frame refreshes the encoder for one frame, but Cutie's memory keeps what reduced frames wrote (the
memory-contamination hypothesis for the fading gain on videos longer than 600 frames). A full memory reset is excluded:
the target is defined only by its given mask in memory, and re-seeding from our own prediction would make any drift
permanent. Base schedule B: section Q's chosen variant if it is a dense one (`key10`, `key20`, `key30`, `lostkey`);
otherwise the dense variant with the largest long-video point estimate in section Q's gate. Variants on dense I-frames
only (`eval_lvos.py`, counted as `mem_events`):
- R1 `B@iwrite`: the I-frame is written to memory (Cutie otherwise writes every fifth frame).
- R2 `B@irefresh`: the non-permanent memory (working and long-term) is cleared, then the I-frame is written; the
  permanent memory (the given masks) is kept.
- R3 `B@iperm`: the I-frame is written to the permanent memory (never evicted).
Development: section Q's 104 videos, head, kernels and implementation; each variant paired per object against B's rows
and against the method's rows. A variant is a candidate if (1) on the long videos it gains at least +1.0 J&F over B
(point estimate), (2) on the sample its lower 95% bound against the method is above -1.0, and (3) its FLOPs stay below
plain ALGM 0.98's. The candidate with the largest long-video gain over B is chosen (`r40_r_gate.json`).
Held-out (amends section Q's held-out step, before it runs): the held-out run takes section R's chosen variant if there
is one (it contains section Q's schedule), else section Q's chosen variant, else none. Reported as an ablation, as in
section Q.

## S. Do I-frames allow tighter focus areas and cheaper settings? (owner 2026-11-23, 20:30; fixed before any section S run)
Question: the checks (ring around the expected core, search region, frame-wide surprise rule with its fallback, floor,
ALGM thresholds) exist to stop drift on fast-moving objects. If periodic or loss-triggered I-frames bound the drift,
some checks could be narrowed and the compute lowered at equal accuracy. Section S runs after sections Q and R, on the
base schedule B of section R (its chosen variant; else section Q's; else the dense variant with the largest long-video
point estimate).
Diagnostics (audit of sections Q and R, 2026-11-23): their rows hold per-video totals only (J, F, FLOPs, frames, fallback
frames, triggers, tier shares, `iframes`, `fb_iframes`, `mem_events`). Missing for this question: (1) fast-motion
strata, available offline now by pairing the rows with the 129-video fast subset (`r40_gate_subset.txt`) with no new run;
(2) per-frame traces (tier and fallback flag, surprise trigger, I-frame flag, and J on annotated frames), to measure
recovery after an I-frame and where drift starts; a default-off `VOS_TRACE` option, added after the section Q gate ends
(a source edit now would reach its remaining runs), used for section S's runs only, with a CPU test that tracing does
not change decisions or scores.
Variants (each B plus one relaxation, development split, R26 head, deterministic kernels, same 104 videos plus the
fast-motion subset videos of the 420 for the fast stratum):
- S1 no ring (middle tier off: ring patches take the low tier), S2 no search region, S3 surprise rule off (a threshold
  the score never reaches), S4 ALGM 0.95 for the middle tier and the fallback (more aggressive merging), S5 floor 32.
  Exact config names are fixed in the job file from `ExpectConfig`; S3's off-switch is confirmed in code first.
Gate (all fixed now): a relaxation is a candidate if (1) on the fast-motion stratum and on the long videos its
difference against B has a point estimate above -0.5 J&F, (2) on the sample its lower 95% bound against the method
(no I-frames) is above -1.0, and (3) its FLOPs on all videos are at least 3% below B's. Reported per stratum with paired
intervals and the trace diagnostics; the most aggressive candidate (lowest FLOPs) is the combination run next (one
combined run of all candidates, same gate). A held-out run only if the combination passes, as a reported ablation; the
preprint's method and claim do not change.

### Section Q result (2026-11-23, 21:33; `r40_q_gate.json`): no candidate, no held-out run
Long videos (44), difference against the method, J&F [95% interval]: key20 +0.41 [+0.05, +1.08], key30 +0.36
[+0.01, +0.98], key10 +0.34 [-0.04, +1.02], fbkey20 +0.30 [-0.13, +1.01], fbkey10 +0.19 [-0.47, +0.97], lostkey +0.02,
lostfb +0.02; the rule needed +1.0. FLOPs on all 104 videos: key10 0.431, key20 0.398, key30 0.388, fbkey20 0.370,
fbkey10 0.372, lostkey 0.368, lostfb 0.366 (ALGM 0.98: 0.430; the method on the same videos about 0.365). The fallback
frame is enough as the reset: within 0.5 of its dense twin in every pair (key20 0.41 / 0.30, key10 0.34 / 0.19) at 0.03
to 0.06 lower FLOPs. Fast-motion stratum (33 development videos in the 129-video subset, 43 objects; offline from the
rows): key20 -0.47, key10 +0.31, key30 +0.35, fbkey20 -0.45, lostfb -0.78; no I-frame variant helps fast objects.
Reading: periodic I-frames give at most +0.4 on long videos for 0.03 to 0.07 more FLOPs; they do not repair the long-
video fade, and the loss-triggered ones barely fire (lostkey: 150 dense frames in 104 videos). (Correction 2026-11-23, 23:55: the
fallback-frame drop from 6,208 to about 1,500 reported here was an error of mine: the method's total was summed over all
420 videos of `eval_r40_gate2` and the variants' over the 104; on the same videos the counts agree, see the diagnostics.)
Amendment to section S (before any S run): S runs only if section R yields a candidate; it then carries a control arm
(the same relaxations on the method without I-frames) so that any saving can be attributed to the I-frames.
Amendment 2 to section S (owner 2026-11-23, 22:00, before any S run; replaces the amendment above): section Q judged I-frames
by accuracy gain, but the hypothesis of section S is different: I-frames bound drift, so tighter focus areas and cheaper
checks become safe, and the saving comes from the relaxation, not from the I-frames. Section S therefore does not
depend on section R. Base schedule: `@fbkey20` (a fallback frame every 20 frames: within 0.5 of its dense twin and
about 0.005 FLOPs above the method in section Q). Each relaxation S1 to S5 runs twice on the same videos: with the
base schedule and without it (control). Candidate: (1) with the schedule, a point estimate above -0.5 J&F against the
unrelaxed method on the fast-motion stratum and on the long videos, and a lower 95% bound above -1.0 on the sample;
(2) FLOPs at least 3% below the unrelaxed method's; (3) attribution: the schedule's run minus the control run is at
least +0.5 on the fast stratum or the long videos (otherwise the relaxation is a tuning result, not an I-frame result).
The most aggressive candidates are then combined, one run per arm, same rule. Section R may change the base schedule only
by an owner decision before these runs start. Held-out as in section S; reported ablation.
Amendment 3 to section S (owner 2026-11-23, 22:30, before any S run): (a) the 10-frame interval is added: stage 2 runs
every relaxation that passes the screening, and the combination, with `@fbkey10` as well (the combination also with
dense `@key10`), same rules; (b) diagnostics come first: `eval_r40diag_q` (`43a_iframe_diag.sh`: per-frame logs of the
method, `@fbkey20` and `@key20` on the 34 videos of `r40_q_diag.txt`), analysed by `r40_q_diag.py` against the
questions in its docstring (coverage and misses by phase since the last I-frame and by object speed, J by phase,
fallback accounting). Its result decides whether the screening runs as written: if object coverage is flat in the
phase and the same with and without I-frames, the schedule adds no tracking margin and section S's attribution rule
cannot be met; the screening is then run only for the controls (relaxations without I-frames) as a tuning result for
the venue version, recorded as such.
Stage 1 (screening) settings, exact names (E = `exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1`, M = `titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98`):
S1 no ring: E without `-ring`; S2 no search region: E without `-srch`; S3 surprise rule off: E with `e100` (patch firing
never reached; new content at the border still fires); S4 ALGM 0.95: M with both thresholds 0.95; S5 floor 32: M with
`floor32`. Order by expected effect on FLOPs: S3, S1, S2, S4, S5. Each: control and `@fbkey20`, the 104 development videos
(`r40_q_dev.txt`, R26 head, reference path, deterministic kernels), paired per object against the method's rows
(`eval_r40_gate2`) and against the unrelaxed `@fbkey20` rows (`eval_r40q_fbkey20`).

### Section R result (2026-11-23, 23:26; `r40_r_gate.json`): `key20@iperm` is a candidate and is chosen
Base schedule key20 (dense frame every 20; section Q's gate chose none, so the rule took the largest long-video estimate).
Long videos (44), J&F: `@iwrite` +0.00 [0.00, 0.00] against key20 (identical rows: at a 20-frame interval every I-frame
already falls on Cutie's every-5th-frame write, so forcing the write changes nothing; this also checks that the
machinery does not perturb anything); `@irefresh` -3.32 [-7.87, +0.33] against key20 (clearing the working and long-term
memory loses what the target needs; excluded); `@iperm` +1.16 [+0.07, +3.24] against key20 and +1.58 [+0.16, +3.96]
against the method, on the sample +0.80 [-0.03, +2.15] against the method, FLOPs 0.397 against ALGM 0.98's 0.430. All
three conditions of section R hold for `@iperm`. Caveats: development split only; the interval's lower bound is just above
0; encoder FLOPs do not count Cutie's added memory-read cost (the permanent memory grows by one frame per 20 frames), so
the end-to-end timing must be reported; the held-out run (heads R31c, h2, h3, `key20@iperm`) decides.
Amendment 4 to section S (before any S run that uses it): stage 2 adds the arm `@key20@iperm` (section R's chosen
schedule) beside `@fbkey10`, for the relaxations that pass the screening and for the combination, same rules.

### Section S diagnostics result (2026-11-23, 23:55; `eval_r40diag_q`, `r40_q_diag.json`)
34 development videos, 27,575 frames, 36,648 object-frames; the logged rows reproduce the gate rows exactly on the same
videos (fallback frames 285 and 257, FLOPs equal), so logging changes nothing. Method / `@fbkey20` / `@key20`:
- Margin by phase since the last I-frame (share of an object's pixels inside protected patches): 0.956, 0.954, 0.956,
  0.958 for the four phase bins of the method and 0.955 to 0.957 with either schedule: flat in the phase and the same
  with and without I-frames. Misses (coverage below 0.9): 7.8% to 8.2% in every bin and setting. J by phase is flat too
  (0.898 to 0.903). No drift between I-frames is visible, and the schedules add no coverage margin.
- By object speed (ground-truth centroid, patches per frame; shares of object-frames 38% slow, 45% mid, 17% fast):
  coverage 0.989 (misses 1.1%) slow, 0.970 (4.1%) mid, 0.843 (33.8%) fast; unchanged by the schedules (fast 0.841).
  In the 14 fast-and-long videos 22% of object-frames are fast and 36% of them are missed; in the other long videos 5%
  are fast.
- Fallback accounting: 285 / 287 / 257 fallback frames, 253 / 257 / 247 empty-map frames in 27,575 (about 1%); the
  dense I-frames of `@key20`: 1,392. 13 and 29 frames fell back in the method but not in the variants.
Reading: the checks (ring, search, surprise rule) are needed for the fast third of the frames' fast objects, where a
third of the frames already miss; the I-frame schedules do not change that. The slow objects (38% of frames, 1% misses)
have margin to spare, but that margin is there without I-frames. So the attribution rule of section S (a relaxation
safe with the schedule but not without) is unlikely to be met by the ring, search and surprise relaxations; the
screening runs as written (cheap controls already in the queue).

### Diagnostics 2 (owner questions 2026-11-24; written before the run)
Questions: (1) would promoting only the changed patches around the focus area, instead of a full ring, cut compute while
still catching deformations; (2) do objects that are new or returning for 1 to 5 frames trigger the surprise mechanism.
Facts from the code (`expect.py`): the ring is the middle tier (inner method's rules); a patch whose mismatch reaches
`thr` fires, and a fired patch inside the ring zone is promoted to high, elsewhere to middle; `zone` is built whether or
not `ring` is set, so section S1 (no ring) is exactly "changed patches only". Offline numbers (34 videos, method rows):
the middle tier holds 21.6% of tokens against 2.7% for the high tier (420 videos), so the ring, search region and fired
patches are where the budget is. Short appearances: 58 reappearances of an object in 47 objects, 10 of 1 to 5 frames
with J 0.45 on those frames (all frames 0.90; ten runs only); lost runs (visible, prediction empty): 44, median 6
frames, 48% of 1 to 5 frames.
Limit of diagnostics 1: its `map_cov` is the share of the object inside the evaluator's map (dilated by one patch), that
is before the ring, search region and surprise promotions; it therefore shows where the base map misses, not how much
the safety nets catch. Reading in the preceding result (the nets are needed for fast objects) is an inference, not a
measurement. Diagnostics 2 logs the final tiers (`tiers` in `diag.py`, logging only; test `test_diag_tiers.py`) and the
share of each object's ground truth inside the high tier and inside high or middle (`cov_h`, `cov_hm`) for the method and
`key20@iperm` on the same 34 videos (job `43b3_tier_diag`), analysed by `r40_tier_diag.py`: by speed, in the first five
frames after a reappearance, and on lost frames. No rule changes; it informs whether S1 and a hold-for-a-few-frames
variant of the surprise rule (a fired region keeps its tier for k frames; not built) are worth running.

## T. Full resolution, escalation and hold for surprise regions (owner questions 2026-11-24; fixed before any section T run)
Context: a fired patch outside every object's zone takes the baseline's rules (middle tier) for the one frame on which it
fires. Never tested in the expectation design: (T1) every fired patch at full resolution (`-fh`); (T2) a patch that fires
on two consecutive frames at the same (camera-compensated) place is high interest (`-esc2`; persistence separates an
object from noise); (T3, T4) a fired patch, grown by one, keeps its tier for 3 or 5 frames after it stops firing
(`-hold3`, `-hold5`; the closed-loop era's R28 found longer holds track better, hold 10 chosen, +2.01 [-4.34, +8.07]
over no surprise on LVOS val, not significant); (T5) `-esc2-hold3`. Reference implementation only (`expect.py`; the timed
implementation would need a port if one were adopted; the evaluator refuses these flags under `VOS_GRAPH=2`).
Setting: `titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1<suffix>`, no I-frames.
Development: section Q's 104 videos (R26 head, LVOS V2 train, deterministic kernels), paired per object against the method's
rows (`eval_r40_gate2`). Candidate if (1) on the 44 long videos the difference is at least +1.0 J&F (point estimate),
(2) on the sample its lower 95% bound is above -1.0, (3) FLOPs on all 104 videos stay below plain ALGM 0.98's. The largest
long-video difference is chosen (ties: lower FLOPs); none: negative, no held-out run. Held-out for the chosen one only (heads
R31c, h2, h3, 140 videos), reported ablation; the preprint's method and claim do not change. Also reported: fast-motion
stratum and FLOPs against the method. Job `49_t_variants.sh` (chain B, after the S screening), analysis
`r40_compare surprise_gate` -> `r40_t_gate.json`.

### Section S first result (2026-11-24, 00:53; `r40_s_gate.json`, development, R26 head, 104 videos): the surprise rule costs compute
S3 (surprise rule off: threshold 100; new content at the border still fires; the frame-wide rule can then not fire), control
arm without I-frames, against the method (all objects of the 104 videos): -0.15 J&F [-0.92, +0.78] at 0.263 against 0.366
FLOPs (28% fewer; ALGM 0.98: 0.430); long videos -0.25 [-0.46, -0.08] at 0.239 against 0.346; fast stratum -1.24
[-2.93, -0.20]. With `@fbkey20` it is -0.73 [-1.75, -0.03] (long -0.90, fast -0.57), and the attribution (schedule minus
control) is -0.66 on long and +0.67 on fast videos (intervals include 0): the schedule does not make the relaxation safe,
section S's candidate rule is not met for S3. The control result stands on its own as a tuning result: the surprise rule
spends about a quarter of the FLOPs for a small accuracy gain on this split. This is development data (one head, the
split the method was tuned on); nothing is adopted from it.

## U. How much surprise is needed? Threshold sweep (owner: more aggressive compute optimisation; fixed before any section U run)
Setting: the method with the surprise threshold `e0.12`, `e0.25`, `e0.5` (method: `e0.06`; off: `e100`, run in section S as
`eval_r40s_S3`), no I-frames, section Q's 104 videos, R26 head, reference path, deterministic kernels (`eval_r40u_<thr>`).
Paired per object against the method's rows (`eval_r40_gate2`). Candidate if (1) on all 104 videos the lower 95% bound is
above -1.0 J&F, (2) on the 44 long videos the point estimate is above -0.5, (3) FLOPs on all videos at least 10% below
the method's. The candidate with the lowest FLOPs is chosen (the off setting competes under the same rule). A held-out run
(heads R31c, h2, h3, 140 videos, reference path) for the chosen one only, reported as an ablation and as a proposal for the
next version; the preprint's method and claim do not change. Also reported: fast stratum, fallback share, middle-tier
share. Job `49_u_thr.sh` (chain B, before section T), analysis `r40_compare thr_gate` -> `r40_u_gate.json`.
Addition to diagnostics 2 (owner 2026-11-24, camera model question; before the run): the log also records the frame's camera
shift (`cam_y`, `cam_x`, patches) and `r40_tier_diag` reports the share of patches that fired and the share of frame-wide
frames by shift size (0, 1, 2 to 3, 4 or more). Reading fixed in advance: a zoom or affine camera model, and a
camera-compensated reuse test, are built only if (a) frame-wide fires or the fired share rise clearly with the shift
size (translation misses real camera motion), and (b) section S1 and section U do not already remove the compute those
fires cause. Neither is built now.

### Section Q/R held-out result (2026-11-24, 01:55; `r40_q_val.json`): `key20@iperm`, heads R31c, h2, h3, 140 videos, reference path
Head-averaged per object (239 objects): J&F 72.82 against the method's 68.63, ALGM 0.98's 65.72 and dense 69.47; FLOPs 0.437
against 0.411 (method) and 0.467 (ALGM). Against the method: +4.20 [+2.57, +5.74] all objects, +11.75 [+6.45, +16.62] on the 42
objects in videos longer than 600 frames, +3.38 [+1.76, +5.12] on the 90 new videos; against ALGM 0.98: +7.10 [+5.35, +9.04].
On the long videos it is 63.74 against dense 52.27. Development (44 long videos) had given +1.58 over the method.
Reading: the variant is above dense, as large as +11 on the longest videos, and far larger than development predicted.
That points to Cutie's own memory (permanent writes of the model's predictions every 20 frames), not the encoder
reduction. It must be attributed before it is interpreted: a result of this kind ("above dense") was withdrawn once before (R9).

## V. Controls for the permanent-memory schedule (fixed before any section V run)
Settings, same schedule on every compared setting: `dense@key20@iperm` (dense encoder everywhere; the frame at t mod 20 = 0
is written to the permanent memory) and `algm_t0.98@key20@iperm` (plain ALGM 0.98 with a dense frame and a permanent write
every 20 frames, the method's schedule). Heads R31c, h2, h3; LVOS V1 val and the 90 new V2 valid videos; reference path,
deterministic kernels (`eval_r40v_dense_<head>_*`, `eval_r40v_algm_<head>_*`; jobs `43b4_dense_iperm` on chain A and
`46d_algm_iperm` on chain B; analysis `r40_compare iperm_controls` -> `r40_v_controls.json`). Reported, head-averaged per
object, on all objects, on videos over 600 frames and on the 90 new videos: each setting's own gain from the schedule (against
its rows without it), method@schedule minus ALGM@schedule, method@schedule minus dense@schedule, FLOPs.
Reading fixed in advance: (a) if dense@schedule gains at least +3 J&F over dense on all objects (about 70% of the method's
+4.20), the gain is a property of Cutie's memory, available to every encoder, and is reported as a Cutie-level finding in an
appendix; the focus map's contribution is then judged by method@schedule against ALGM@schedule; (b) if dense and ALGM gain
little and the method gains much, the interaction of the focus map and the schedule is the finding and the held-out claim
may be widened only by an owner decision with fresh data; (c) in between, both are reported. The preprint's method and
claim do not change in any case (the held-out videos have been used for this method already).

### Diagnostics 2 result (2026-11-24, 02:42; `eval_r40diag_t`, `r40_tier_diag.json`; 34 videos; method and `key20@iperm` agree)
Share of an object's ground truth inside (base map | final high tier | high or middle), method, on frames where the expectation ran:
all object-frames 0.961 | 0.989 | 0.999; slow 0.994 | 0.996 | 0.999; mid speed 0.976 | 0.987 | 0.999; fast 0.851 | 0.982 | 0.999
(object-frames below 0.9: fast 33% | 2% | 0%). So the nets do what they were built for: the base map misses a third of the fast
frames, the final high tier misses 2%, high or middle none (this measures the inference of diagnostics 1, which stands). The
ring (middle tier) adds 1.7 points of coverage on fast frames beyond the high tier.
First five frames after a reappearance (302 object-frames): 0.514 | 0.694 | 0.972, J 0.51; lost frames (visible, prediction empty; 428
frames): 0.093 | 0.292 | 0.940, J 0.0 by definition. The object lies inside the protected tiers on 94% to 97% of those frames and is still
lost or poorly segmented: the failure there is the segmenter's matching, not the encoder's coverage; promoting middle to high is
unlikely to fix it by itself (section T tests it), and this matches the size of the permanent-memory effect.
Camera shift (frames; fired share of patches; frame-wide share): shift 0: 18,373; 0.194; 8.4%; shift 1: 4,358; 0.251; 12.8%; shift 2-3:
3,006; 0.260; 11.6%; shift 4 or more: 1,804; 0.297; 13.2%. A fifth of the patches fire with no camera shift at all, and the rise with
the shift is 10 points: the translation model is not the main source of fires. Rule recorded before the run (a clear rise with the
shift size): not met; a zoom or affine camera model and a camera-compensated reuse test are not built. The compute is in the
threshold (section U).
(Analysis note: the first version of `r40_tier_diag.py` pooled the two settings under one tag; fixed and rerun.)

### Section S, S1 result (2026-11-24, 02:15; no ring, changed patches only; `r40_s_gate.json`, development)
Control (no schedule) against the method: all objects +0.13 [-0.12, +0.53] at 0.341 against 0.366 FLOPs (7% fewer); long videos
+0.54 [-0.07, +1.41]; fast stratum +0.35 [-0.06, +1.12]. With `@fbkey20`: -0.28 [-0.85, +0.18] at 0.347; attribution (schedule minus
control) -0.32 long, -0.82 fast: the schedule does not help (candidate rule of section S not met). Dropping the ring costs no accuracy
here, as the coverage numbers above predict, and saves 7% of the FLOPs; S3 (surprise off) saves 28% at -0.15.
Amendment to section U (before any section U run): one more development run, `-o1` without `-ring` and with the threshold of the
lowest-FLOPs candidate of the sweep (`eval_r40u_combo`; not run if the sweep has no candidate), same candidate
rule; the held-out run takes the combination if it is a candidate and has lower FLOPs than the sweep's choice, else the sweep's choice.

### Section S, S2 result (2026-11-24, 03:42; no search region; development)
Control against the method: -0.00 [-0.01, +0.01] on all objects at equal FLOPs (0.366), long 0.00, fast 0.00: the search region changes
essentially nothing on these 104 videos (it only acts on frames where an object is lost, and the fallback and the tiers already
cover those). With `@fbkey20`: +0.14 [-0.64, +1.02]; attribution +0.31 long, -0.48 fast; candidate rule not met. Screening so far
(S1, S2, S3; S4 and S5 run on chain A later): no relaxation meets section S's candidate rule (the schedule never makes one safer);
as controls, S1 saves 7% and S3 28% of the FLOPs at no or small accuracy cost, S2 saves nothing.

### Section V, dense control (2026-11-24, 04:51; held-out, 3 heads, 140 videos, 239 objects; ALGM control still running; partial reading)
`dense@key20@iperm` (every 20th frame written to the permanent memory), J&F against dense 69.47: 72.88, a gain of +3.41 [+2.31, +4.51]
on all objects, +9.99 [+6.20, +14.22] on the 42 objects in videos over 600 frames (62.26 against 52.27), +2.40 [+1.25, +3.66] on the
90 new videos. The method with the same schedule (72.82): minus dense@schedule -0.06 [-1.03, +0.92] all, +1.48 [-1.65, +4.37] long,
-0.75 [-1.75, +0.23] new; its own gain from the schedule +4.20 / +11.75 / +3.38. Reading rule (a) of section V is met (dense gains
at least +3): the effect belongs to Cutie's memory, not to the focus map; the method with the schedule is level with dense Cutie
with the schedule at 0.437 against 1.0 FLOPs. The focus map's own contribution at equal schedule waits for the ALGM control
(`r40_compare iperm_controls`). Caveats: LVOS only, the held-out videos were already used for the method, the permanent memory
grows by one frame per 20 and Cutie's own cost (time, GPU memory) is not in the encoder FLOPs.

### Section V result (2026-11-24, 06:12; `r40_v_controls.json`; held-out, 3 heads, 140 videos, reference path)
J&F default / with `key20@iperm` / gain [95% interval], all objects (long videos, 42 objects): dense 69.47 / 72.88 / +3.41 [+2.31, +4.51]
(52.27 / 62.26 / +9.99 [+6.20, +14.22]); ALGM 0.98 65.72 / 69.49 / +3.77 [+2.40, +5.14] (51.26 / 62.32 / +11.06 [+6.25, +16.04]);
method 68.63 / 72.82 / +4.20 [+2.57, +5.74] (51.99 / 63.74 / +11.75 [+6.45, +16.62]); FLOPs method 0.411 -> 0.437, ALGM 0.467 -> 0.495.
At equal schedule: method minus ALGM +3.34 [+1.91, +4.94] (long +1.42 [-0.95, +4.21], new 90 +3.23 [+1.48, +5.13]); method minus dense
-0.06 [-1.03, +0.92] (long +1.48 [-1.65, +4.37], new -0.75 [-1.75, +0.23]). Rule (a) met (dense gains +3.41, at least +3): a Cutie-level
finding, reported in an appendix; the focus map's advantage over ALGM persists at equal schedule. Written into the paper as appendix
`app:perm` and a pointer in Limitations; kept out of the method and the claim; open: Cutie's step time and memory (`99b`).

### Section S4/S5 and U results (2026-11-24, 08:25; development, 104 videos, R26 head; `r40_s_gate.json`, `r40_u_gate.json`)
S4 (ALGM 0.95) control -0.59 [-1.77, +0.14] at 0.333 FLOPs (9% fewer), with `@fbkey20` -1.44; S5 (floor 32) +0.04 [-0.15, +0.23] at 0.359 (2%
fewer): neither meets section S's rule (attribution -2.77 fast for S4, +0.35 long / -0.47 fast for S5). Section S closed: no relaxation is made
safe by the schedule.
Section U, surprise threshold against the method (all objects; long; fast; FLOPs, method 0.366): `e0.12` -0.58 [-2.16, +0.03]; -0.03; -0.03; 0.323
(not a candidate: lower bound); `e0.25` -0.33 [-0.88, +0.17]; -0.14; -0.98 [-2.61, -0.05]; 0.287 (candidate); `e0.5` +0.05 [-0.11, +0.31]; +0.18; +0.19;
0.269 (candidate); off -0.15 [-0.92, +0.78]; -0.25 [-0.46, -0.08]; -1.24 [-2.93, -0.20]; 0.263 (candidate). Rule: the lowest FLOPs among candidates:
**off** (0.263, 0.006 below `e0.5`). Combination no ring + off: -1.73 [-3.77, -0.15] at 0.229 (not a candidate: the ring and the surprise rule substitute
for each other; one of the two is needed), so the held-out choice stays `off` by the rule.

## W. Held-out confirmation of section U (fixed before any section W run)
Held-out run of the rule's choice (`off`, threshold 100) and, by amendment before the run, of `e0.5`: within 0.006 FLOPs of `off` and at least
as accurate on every development stratum (all +0.05, long +0.18, fast +0.19 against -0.15, -0.25, -1.24), and unlike `off` it keeps the frame-wide
and scene-cut safety. Both: heads R31c, h2, h3; LVOS V1 val (50) and the 90 new V2 valid videos; reference implementation, deterministic kernels;
no schedule (`eval_r40w_<tag>_<head>_*`; job `51_w_heldout` on chain A; `r40_compare thr_heldout` -> `r40_w_val.json`). Reported, head-averaged per
object on all objects, videos over 600 frames and the 90 new videos: against the method's reference rows and against ALGM 0.98, with FLOPs. A setting is
confirmed if on all 239 objects its lower 95% bound against the method is above -1.0 and its FLOPs are at least 10% below the method's, and its lower
bound against ALGM 0.98 is above 0. Status: a reported ablation and a proposal for the next version; the preprint's method and claim do not change (the
held-out videos have served the method already).

### Section T result (2026-11-24, 11:13; development, 104 videos, R26 head; `r40_t_gate.json`): none
Against the method (long videos; fast stratum; FLOPs on all 104 videos, ALGM 0.98: 0.430, method about 0.366): `-fh` -0.35 [-1.51, +0.66]; +0.40
[-0.09, +1.25]; 0.428; `-esc2` -0.52 [-1.42, 0.00]; -0.04; 0.407; `-hold3` +0.77 [-0.81, +2.85]; +1.36 [+0.09, +3.16]; 0.477; `-hold5` +0.82 [-0.80, +2.93];
+1.42 [+0.15, +3.28]; 0.488; `-esc2-hold3` +0.67 [-0.81, +2.41]; +1.42 [+0.14, +3.26]; 0.529. No candidate (the rule needed +1.0 on long videos and FLOPs below
ALGM's). Reading: promoting surprise regions to full resolution (`-fh`) or on persistence (`-esc2`) changes nothing useful, as the coverage diagnostics
predicted; holding fired regions helps the fast-motion stratum (+1.4, interval above 0) but costs 30% more FLOPs. Closed by the rule.

## X. Replication of the permanent-memory finding on the other development videos (fixed before any section X run)
The permanent-memory schedule was chosen on 104 of the 420 LVOS V2 train videos (`r40_q_dev.txt`). The other 316 (`r40_x_rest.txt`; development videos for the
method, not for this schedule) give a second look before any held-out reuse: R26 head, reference implementation, deterministic kernels; settings
`<method>@key20@iperm` for the method, dense Cutie and ALGM 0.98 (`eval_r40x_<method|dense|algm>`; job `52_x_replicate` on chain B), against the existing
rows of the same videos and kernels (`eval_r40_gate2` for the method, `diag_dg1` for dense and ALGM 0.98). Reported per object: each setting's gain from the
schedule on all objects, in videos over 600 frames and in the others, and at equal schedule method minus ALGM and method minus dense (`r40_compare
rest_replication` -> `r40_x_rest.json`). Reading fixed in advance: the finding replicates if the gains of dense and of the method both have a lower 95% bound
above 0 on all objects; the long-video gain is reported with its interval. No change to the method or claim; strengthens or weakens the appendix.

### Section W result (2026-11-24, 12:11; `r40_w_val.json`; held-out, 3 heads, 140 videos, reference path, no schedule): not confirmed by the rule
Against the method (all objects; long; new 90), against ALGM 0.98, FLOPs: `e0.5`: -0.06 [-1.10, +1.00]; +0.07 [-2.52, +2.52]; -0.07 [-1.28, +1.13]; all +2.84 [+1.16, +4.70];
0.327 against the method's 0.411 and ALGM's 0.467. Off: -1.10 [-2.38, +0.12]; -0.33; -1.15; +1.80 [+0.04, +3.68]; 0.320. Rule (lower bound against the method above -1.0,
FLOPs at least 10% below, lower bound against ALGM above 0): `e0.5` misses on the first condition by 0.10, off by 1.38: not confirmed; the rule stays. Reading: the surprise
check at its tuned threshold buys little accuracy for about a fifth of the FLOPs, and the headline claim against ALGM holds at 0.327 FLOPs (30% fewer than ALGM). Written into the paper
as an ablation (not adopted); adoption would need fresh data.

### Section X result (2026-11-24, 13:10; `r40_x_rest.json`; 316 development videos, 454 objects, R26 head, reference path)
Replicates by the rule (lower bounds of dense and method above 0): dense 92.50 -> 93.21, +0.71 [+0.11, +1.36]; method 91.23 -> 92.36, +1.13 [+0.40, +1.89] (FLOPs 0.399 -> 0.433); ALGM 90.07 -> 91.40,
+1.33 [+0.53, +2.29] (0.451 -> 0.479). At equal schedule: method minus ALGM +0.96 [+0.31, +1.73]; method minus dense -0.85 [-1.66, -0.16]. Videos over 300 frames (121 videos, 145 objects): gains +0.79
(dense), +0.79 (method), +1.45 (ALGM), intervals include 0; under 300 frames: +0.67, +1.30, +1.27. No video here has more than 600 frames (all 44 are in the 104 selection videos), so the long-video effect
is covered only by the held-out run; the gain is smaller on shorter videos, as expected if the memory eviction is the cause. In the paper's appendix `app:perm`. (The first analysis crashed on an empty
stratum; fixed, rows were complete.)

## Y. Combinations of the late findings (owner question 2026-11-24, 14:30; fixed before any section Y run)
Reasoning: individually the surprise threshold `e0.5` saves compute at no development cost (held-out: -0.06, rule missed by 0.10), the permanent-memory schedule
gains accuracy at a cost in Cutie's step time, and holding surprise regions helps the fast stratum at a FLOP cost that scales with how many patches fire (19% of
patches at `e0.06`, far fewer at `e0.5`). The schedule was run only with a dense frame every 20 (`key20@iperm`); whether the dense frame matters at all, or only the
permanent write does, has not been separated. New evaluator option `perm<N>`: the frame at t mod N = 0 is written to the permanent memory as it was processed (no dense
frame, no change to the encoder). Variants (development: section Q's 104 videos, R26 head, reference path, deterministic kernels; M = `titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98`,
E = `exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1`, E5 = E with `e0.5`; rows `eval_r40y_<tag>`; paired per object against the method's rows `eval_r40_gate2`):
- Y1 `M@E@perm20`: the permanent write without the dense frame (is the dense frame needed). Y5 `M@E@perm40`: half the permanent writes (Cutie's cost).
- Y2 `M@E5@perm20`: the cheap threshold with the permanent write. Y4 `M@E5@key20@iperm`: the cheap threshold with the full schedule.
- Y3 `M@E5-hold3`: hold at the cheap threshold.
Rules (candidate; the largest gain on the 44 long videos is chosen; ties lower FLOPs): Y1 and Y5: long videos at least +1.0 J&F (point estimate), sample lower 95% bound above -1.0, FLOPs on all 104 videos at most 1.02
times the method's. Y2 and Y4: long videos at least +1.0, sample lower bound above -1.0, FLOPs at least 10% below the method's. Y3: fast stratum at least +1.0, all-object lower bound above -1.0, FLOPs at most the
method's. Reported for every variant: all, long, fast and sample differences, FLOPs, and the number of permanent writes (`mem_events`). Held-out (3 heads, 140 videos, reference path) for the chosen
candidate only, as an ablation; no change to the preprint's method or claim. Jobs `60_y_a` (chain A: Y1, Y2, Y5), `60_y_b` (chain B: Y4, Y3); `r40_compare combo_gate` -> `r40_y_gate.json`.

## Z. Additions for the paper (owner 2026-11-24, 17:30: "add everything that helps, except moving to a trained mechanism"; fixed before any section Z run; none started, all held for the owner's approval)
Done without runs (data existed): the coverage table and reading (diagnostics 2), non-inferiority wording on short clips with a margin of 1 J&F chosen after the results (stated as such), the official-Cutie positioning
(R30; corrected: our ViT-B head is -3.28 below the official small model, not the base), G2TM and DTEM rows from R18 and R21 (head R10, earlier protocol, labelled), and the held-out gain by video length
(`r40_length_gain.json`). Paper: Section 5.3, Appendix E, Scope item.
Held runs (jobs in `~/ml_work/hold`, none queued; every result is reported whatever it shows, no selection among them):
- Z1 `70_davis_g2tm_dtem`: G2TM (0.94, 0.96, 0.98) and DTEM (r64, r96, r128, Cutie loss) on DAVIS val on the paper's head (R26) and protocol, replacing the R10-head rows.
- Z2 `71_dev_oracle`: map-quality headroom, the method with the expectation's object states and the evaluator's map from the ground truth of the previous two frames (`@gtobj`, a diagnostic), 104 development videos;
  `r40_compare oracle_headroom`. Reading fixed in advance: if it is within 1 J&F of dense at about the method's FLOPs, the remaining gap to dense is the tracker's; if it is far below, it is the compute policy's.
- Z3 `72a_heldout_g2tm`, `72b_heldout_algm_matched`: G2TM (0.94, 0.98) and plain ALGM at 0.95 and 0.96 (the method's FLOPs) on the held-out long videos, three heads; `extras_heldout`. Reading: the claim at equal FLOPs holds if
  the method's lower bound against ALGM at the threshold whose FLOPs are closest to the method's is above 0 (reported either way).
- Z4 `73_memory_timing` (idle GPU): peak GPU memory and step time of dense Cutie without and with the permanent-memory schedule.
- Z5 `74_lvos_v2_test_masks`: predictions on the LVOS V2 test split (160 videos, images and first-appearance masks only; no V2 server is announced, the CodaLab competition 8767 is the only listed one) for an external evaluation:
  head R31c, reference path; settings fixed in advance and all to be submitted whatever they score: dense, ALGM 0.98, the method, and the method, dense Cutie and ALGM 0.98 with `key20@iperm`; packed per setting with
  `tools/make_submission.py` (the format must be confirmed on the competition's Evaluation tab first).
Not done by rule (training): more ViT-L heads, a stronger head, learned categories. Engineering items (shared graph cache, Cutie memory-readout pruning, codec surprise detector) stay in the future-work list.
Amendment to Z5 (owner 2026-11-24, after pasting the competition's Terms and Conditions): the CodaLab competition 8767 describes videos from VOT-LT 2019 and LaSOT, i.e. LVOS V1; its test split is the 50 V1 test videos.
They were downloaded (`data/downloads/lvos_v1_test/test.zip`, from the authors' Google Drive link with gdown inside Docker; extracted to `data/lvos/test`): 50 videos, 29,894 frames, 13 videos over 600 frames, first-appearance masks
only; all 50 are inside the V2 test split with equal frame counts (the other 110 V2 test videos have no announced server). Z5 therefore runs on these 50 videos (`reports/stage_b/queue/lvos_v1_test_videos.txt`); the settings
and the rule (all submitted, whatever they score) are unchanged. Data terms (competition page): annotations CC BY 4.0, toolkits BSD-3-Clause, data for non-commercial research only, videos not the organizers' property; we do not
redistribute data (the release contains code and result summaries only).

### Y held-out step: rule fixed before the analysis (2026-11-25, written about 04:30; the Y4 held-out runs had finished at about 02:45, so the criterion was written after the runs and before any held-out Y4 number was computed)
`r40_compare y4_heldout` (three ViT-S heads, 140 videos): Y4 is "confirmed" as an operating point (no change of method) if the lower bound of Y4 minus the method with the same schedule is above -1.0 J&F on all objects and its FLOPs are at least 10% below that method's. Reported either way: minus the plain method, minus dense Cutie, minus ALGM 0.98 and dense with the same schedule, all objects, videos over 600 frames, the 90 new videos.

### Results of sections Y (held-out) and Z2, Z3a (2026-11-25)
J&F differences in points [95% object bootstrap, 1,000 draws], head-averaged per object over three ViT-S heads, held-out (239 objects; 42 in videos over 600 frames; 174 in the 90 new videos).
- **Y4 (`E5@key20@iperm`), `r40_y4_heldout.json`:** rule met (confirmed as an operating point, no change of the method). Against the method with the same schedule: +0.79 [-0.28, +1.78] at 0.356 vs 0.437 FLOPs (18.5% fewer); long videos +1.59 [-0.67, +3.77]; new 90 +0.15 [-0.83, +1.11]. Against the plain method +4.99 [+3.17, +6.77] (long +13.34 [+7.70, +18.33]); against dense Cutie +4.14 [+2.49, +5.77] at 0.356 FLOPs (long +13.06 [+7.83, +18.45]); against dense Cutie with the same schedule +0.73 [-0.63, +2.00] (long +3.07 [-0.72, +6.59]); against ALGM with the schedule +4.13 [+2.68, +5.79].
- **Z2 oracle headroom (development, 104 videos, `@gtobj`), `r40_oracle_headroom.json`:** a perfect tracker's masks for the map give +0.11 [-0.09, +0.42] over the method at 0.364 vs 0.366 FLOPs (long +0.33 [-0.06, +1.03]; fast stratum +0.40 [-0.09, +1.31]); the map's inputs are not the limit.
- **Z3a G2TM held-out, `r40_extras_heldout.json`:** `g2tm_t0.94_at2` -15.53 [-18.65, -12.62] below the method at 0.328 vs 0.411 FLOPs; `g2tm_t0.98_at2` -1.54 [-3.23, +0.17] below the method at 0.613 vs 0.411 FLOPs (long videos -0.74 [-4.15, +2.75] at 0.594 vs 0.374). Matched-FLOPs ALGM (Z3b) follows.
- **Z1 G2TM and DTEM on DAVIS val (paper's head R26, fast protocol; `r40_davis_more.json`, 30 videos, 61 objects; dense 87.26, the method 87.07 at 0.462 FLOPs):** DTEM r96 87.43 at 0.594 (method minus it -0.37 [-1.11, +0.16]); DTEM r128 84.00 at 0.480 (method +3.07 [+0.95, +6.38]); DTEM r64 87.38 at 0.721; G2TM 0.98 87.35 at 0.791 (method -0.28 [-0.94, +0.20]); G2TM 0.96 86.16 at 0.595 (method +0.91 [-0.04, +1.72]); G2TM 0.94 78.26 at 0.414 (method +8.81 [+5.92, +11.89]). At matched FLOPs (0.48 to 0.60) the method is at or above both; the baselines that match its accuracy cost 28% to 71% more.
- **Z3 held-out baselines (`r40_extras_heldout.json`, complete):** minus the method: G2TM 0.94 -15.53 [-18.65, -12.62] at 0.328 FLOPs; G2TM 0.98 -1.54 [-3.23, +0.17] at 0.613; ALGM 0.96 (FLOPs matched) -6.14 [-8.20, -4.02] at 0.389 and -3.23 [-4.81, -1.71] against ALGM 0.98; ALGM 0.95 -9.25 [-11.42, -6.97] at 0.362. Videos over 600 frames: ALGM 0.96 -1.04 [-5.74, +3.10], ALGM 0.95 -5.09 [-9.99, -0.29].
- **Z4 memory timing (`r40_e2e_timing_dense_m.json`, `..._dense_iperm_m.json`; dense Cutie, 12 videos, 6,516 frames, one process per setting, idle GPU):** peak GPU memory (whole process) allocated 1.11 GB without and 6.60 GB with `key20@iperm` (reserved 1.28 and 6.98 GB); step time median 14.31 and 21.25 ms, p90 17.49 and 47.89, p99 31.00 and 62.08. The schedule's cost is real: about six times the memory on long videos.
- **Timing of the cheaper surprise setting and of plain ALGM (closed loop, 12 LVOS videos, idle GPU; `r40_e2e_timing_{vits,vitb}_{e5,algm}.json`):** `e0.5` encoder median 3.94 ms (ViT-S) and 5.09 (ViT-B) against the method's 4.20 and 5.32; whole step after startup 14.78 and 16.39 ms against dense 15.63 and 18.33 (-5% and -11%; the method at `e0.06`: -3% and -8%). Plain ALGM ran on the eager path in this harness (encoder median 7.70 and 7.68 ms, slower than dense): not the equally engineered ALGM, so no closed-loop comparison with engineered ALGM is claimed; Table 5's engineered comparison stands (ALGM faster at ViT-S and ViT-B).

### Correction 2026-11-25 (external audit, verified): frame-weighted FLOPs in the late summary files
The summary functions `iperm_controls`, `thr_heldout`, `rest_replication`, `y4_heldout` and `extras_heldout` of `r40_compare.py` averaged the FLOPs ratio over videos without weighting by frames, against the definition of Section 3.7 (frame-weighted across videos). Recomputed (files regenerated, accuracy numbers unchanged): the method 0.388 (was 0.411), ALGM 0.98 0.459 (0.467), the method with the schedule 0.413 (0.437), ALGM with the schedule 0.486 (0.495), Y4 0.335 (0.356; 18.9% below the method with the schedule, rule still met), `e0.5` 0.313 (0.327), surprise off 0.307 (0.320), G2TM 0.94 and 0.98 0.319 and 0.596 (0.328, 0.613), ALGM 0.96 and 0.95 0.383 and 0.356 (0.389, 0.362); 316 development videos: method 0.384 default and 0.418 with the schedule. Earlier lines of this file and of DECISIONS that quote the old values are superseded by this note.

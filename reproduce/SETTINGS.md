# Setting names

Every method is selected by a setting string, parsed by `src/stage_b/timing.py:from_name` and the evaluators. The
strings key the result files, so they keep their original prefixes.

| Setting | Meaning |
| --- | --- |
| `dense` | The unmodified ViT |
| `algm_t<τ>` | ALGM with similarity threshold τ (layers 1 and 5) |
| `tome_r<r>_from1_noprop` | ToMe, r merges per block from block 1, without proportional attention |
| `ailurus_m<m>_at<b>` | AiluRus, keep ratio m at block b |
| `pitome_k<k>` | PiToMe, keep ratio k |
| `expedite_m<m>_at<b>_t<t>` | Expedite, keep ratio m at block b, temperature t |
| `palgm_t<τ>_dil<d>` | ALGM outside the previous-mask focus map (dilated by d patches), full resolution inside |
| `ftome_l<rate>_t<θ>_floor<n>_dil<d>_noprop_cls64` | The earlier closed-loop focus map (previous mask, dilated by d patches): full resolution inside, a merge budget at rate `rate` outside down to a floor of n tokens |
| `titome_t<τ>_l<rate>_floor<n>_dil<d>_cls64~algm_t<τ>@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1` | **The method.** Three tiers (`titome`: high tier full resolution, middle tier under ALGM τ's rules, low tier merged at rate `rate` down to n tokens), `~algm_t<τ>` the fallback (the baseline itself), `@exp-...` the expectation and surprise settings (pixel-based motion, relocalisation, ring, search regions for lost objects, surprise tolerance) |

Final settings: ViT-S `titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98@...`; ViT-B `titome_t0.95_l0.7_floor128_dil2_cls64~algm_t0.95@...`; ViT-L `titome_t0.95_l0.7_floor128_dil1_cls64~algm_t0.95@...` (`src/stage_b/vos/r40_compare.py`: `S_FINAL`, `SB_FINAL`, `SL_FINAL`).

# PiToMe-derived code: CC BY-NC 4.0

**Material:** the PiToMe algorithm of Tran et al., "Accelerating Transformers with Spectrum-Preserving Token Merging" (NeurIPS 2024), from the repository `hchautran/PiToMe` at commit `550b5de` (files `algo/pitome/merge.py`: `pitome_vision`, `pitome`, `pitome_bsm`).

**Licence of the original:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Full legal code: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>. Summary: <https://creativecommons.org/licenses/by-nc/4.0/>.

**Where it is used here:** only in `src/stage_b/baselines.py`, in the two regions that begin and end with the lines `# SPDX-License-Identifier: CC-BY-NC-4.0 (derived from hchautran/PiToMe @ 550b5de; see NOTICE) -- begin` and `-- end`: class `PiToMe` (the PiToMe baseline) and the energy computation of class `Novelty` (a diagnostic map built from PiToMe's energy score). Nothing else in this repository is derived from PiToMe.

**Changes made (CC BY-NC 4.0, section 3(a)(1)(B)):** the algorithm was rewritten as a `Method` of this repository (the hook `after_attn(block, x, keys, ...)`, called after the attention of each block inside Cutie's pixel encoder on a frozen DINOv3 ViT), with the merge schedule given by a keep ratio per block (`pitome_k<k>`, the original's `ratio`) and the merge state kept in this repository's token-size bookkeeping. The energy score and the merge rule follow the original (keys averaged over heads, cosine similarity, the margin that decreases linearly over the blocks, ELU of the margin-shifted similarity). The energy computation is also reused, unchanged in form, as a diagnostic map in class `Novelty`.

**Terms for this part:** it may be copied, adapted and shared for non-commercial purposes only, with this attribution; it is provided without warranty. The rest of the repository is Apache-2.0 (`LICENSE`), except where `NOTICE` lists other terms. Anyone who wants to use the repository commercially must remove or replace the two marked regions.

# Perturbation label-preservation check

## Scope and method

Checked whether the three robustness perturbations (`jpeg3`, `typos3`, `crop2`) keep each post's gold label plausible. Source: `scripts/inspect_perturbations.py`, which renders original vs perturbed side by side for 10 D_rob posts per condition, taken round-robin over the four classes (seed 42), and writes `automated_report.json` and `hand_check.csv`.

Two kinds of evidence are kept separate:

- **Automated checks** (script output, `automated_report.json`).
- **Visual checks** (side-by-side images and captions looked at by eye). 


This is a sample (10 posts per condition), not an exhaustive check of all D_rob posts.

## Automated checks

| Condition | Examined | Skipped by rule | Unchanged image | Unchanged text | Size changed | Label fields changed | Mean abs pixel diff |
|---|---|---|---|---|---|---|---|
| jpeg3 | 10 | 0 | 0 | 0 | 0 | 0 | 5.67 |
| typos3 | 10 | 0 | 0 | 0 | 0 | 0 | n/a (text only) |
| crop2 | 10 | 25 | 1 | 0 | 10 | 0 | 36.64 |

Typo captions had text similarity to the original between 0.88 and 0.98 in this sample.

## Visual check results

| Condition | Inspected | Label preserved (Y) | Not preserved (N) | Uncertain (U) |
|---|---|---|---|---|
| jpeg3 | 10 | 10 | 0 | 0 |
| typos3 (captions only) | 10 | 9 | 0 | 1 |
| crop2 (real posts only) | 10 | 9 | 0 | 1 |

Per-example Y/N/U and notes: `hand_check.csv`.

### jpeg3
All 10 perturbed images show the same scene as the original, with mild softening and a small contrast/colour shift. Softest examples: `03_real` (dark stage regions smeared) and `06_mismatch` (hazier background, softer face). Edited `image_false` composites (`01_image_false`, `05_image_false`) still show the manipulation clearly. Scoreboard digits in `10_mismatch` are still legible. No semantic change seen.

### typos3
Only the caption changes; the image is untouched and was not part of this check. Most captions stay readable with names still guessable. Heavy cases: `06_mismatch` (Sajdets, Gelrge Mwson yniverwity) and `10_mismatch` (Tiqfoe, Ttpjcki, firdteound). `10_mismatch` is marked uncertain because the opponent's name is no longer recognisable without the original.

### crop2
Applied to real posts only, because cropping can remove the part of an image that makes a post fake. In 9 of 10 examples the main subject stays in frame. `10_real` loses the map title banner and logo, so some information is lost, though the caption still fits.

## Uncertain cases

- `crop2/03_real` (post `test-1395`): original and cropped panels are completely black (all pixels 0). Content cannot be judged. Matches `unchanged_image = 1` in the automated report. Needs a check of the source image.
- `typos3/10_mismatch` (post `test-7407`): see above.

## Missing / corrupted / unmatched

- Missing: 0 (all 20 image renderings and 30 caption files present).
- Unreadable: 0 (all 20 PNGs opened).
- Blank image pairs: 1 (`crop2/03_real`).
- Not covered by this check: total original images, perturbed counts per type, file-level matching against `data/raw/MMFakeBench`, and duplicate detection.

## Limitations

- Sample of 10 posts per condition; findings do not generalise to all D_rob posts.
- Images were viewed as side-by-side renderings at about 360 px height, so fine JPEG artifacts could not be judged. JPEG quality level was not inspected.
- Original gold labels (especially `mismatch`) were not re-verified; only that the perturbed content matches the original.
- For `image_false` posts, "preserved" means the edit is still visible, not that a detector can still find it.

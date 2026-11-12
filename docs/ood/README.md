# Photos that are not apple leaves

The scab detector answers for every photo it gets, so before the input check existed it reported scab lesions on
photos of sky, hands or documents. To measure this and to build the input check, 228 openly licensed photos from
Wikimedia Commons (CC0, public domain, CC BY, CC BY-SA; no NC or ND licences) were collected in [`set/`](set), one
folder per category.

- **Near** (122): things a user might photograph instead of an apple leaf, or that look like one: leaves of pear,
  cherry, grape, maple and other trees, apple fruit, apple blossom, bark and branches, grass.
- **Far** (106): hands, sky, people, documents, screenshots, scenes, everyday objects and other animals. Apart
  from hands and sky, these are the same Commons photos as the poultry repository's far categories.

[`build_ood_set.py`](../../scripts/build_ood_set.py) found candidates with fixed search queries per category; the
photos were then picked by hand from contact sheets, before any input check was scored
([`selection.json`](selection.json), with a reason for every skipped candidate), and downscaled to 640 px on the
long side. [`split_sets.py`](../../scripts/split_sets.py) split them once into ood-tune (97) and ood-test (131),
stratified by category with a fixed seed ([`split.json`](split.json)). Credits: [`CREDITS.md`](CREDITS.md) and
[`credits.json`](credits.json); each photo keeps its own licence.

## What the detector and the input check do

Before writing any input check code, the trained detector was run on every photo
([`evaluate_ood_detector.py`](../../scripts/evaluate_ood_detector.py),
[`detector_predictions.json`](detector_predictions.json)): it reported scab lesions on 179 of 228 and the app
called the other 49 "clean". The input check (`frostguard-tinyclip-8m16-proto-1`, threshold 0.8629,
[`input_check.json`](input_check.json)) then decides whether the detector is asked at all:

| Category | Near or far | Photos (tune / test) | Detector alone reports scab (all photos) | Passes the input check (ood-test) |
|---|---|---|---|---|
| apple blossom | near | 6 / 8 | 14 of 14 | 0 of 8 |
| apple fruit | near | 6 / 8 | 13 of 14 | 1 of 8 |
| bark branches | near | 6 / 8 | 11 of 14 | 0 of 8 |
| cherry leaves | near | 5 / 7 | 12 of 12 | 1 of 7 |
| grape leaves | near | 6 / 8 | 14 of 14 | 4 of 8 |
| grass | near | 6 / 8 | 13 of 14 | 0 of 8 |
| maple leaves | near | 6 / 8 | 13 of 14 | 0 of 8 |
| other leaves | near | 5 / 7 | 12 of 12 | 4 of 7 |
| pear leaves | near | 6 / 8 | 10 of 14 | 6 of 8 |
| documents | far | 6 / 8 | 5 of 14 | 0 of 8 |
| everyday objects | far | 6 / 8 | 5 of 14 | 0 of 8 |
| hands | far | 5 / 7 | 7 of 12 | 0 of 7 |
| other animals | far | 6 / 8 | 13 of 14 | 0 of 8 |
| people | far | 5 / 7 | 9 of 12 | 0 of 7 |
| scenes | far | 6 / 8 | 12 of 14 | 0 of 8 |
| screenshots | far | 6 / 8 | 7 of 14 | 0 of 8 |
| sky | far | 5 / 7 | 9 of 12 | 0 of 7 |
| **All** | | 97 / 131 | 179 of 228 | 16 of 131 |

The check stops photos that are not leaves, but it lets many leaves of other trees through, most of all pear (a
close relative of apple with similar leaves and diseases), walnut and other leaves, and grape. Rates with 95%
intervals, and the false-rejection rates on real apple-leaf photos, are in
[`results/input_check/evaluation.md`](../../results/input_check/evaluation.md).

# Orchard photos of apple leaves

The scab detector was trained on PlantVillage-style lab photos (one leaf on a plain background), so passing those
alone proves little about the input check. These 35 openly licensed photos from Wikimedia Commons show apple leaves
outdoors or on the tree, in three categories:

| Category | Photos | Prototype / tune / test |
|---|---|---|
| healthy | 13 | 4 / 4 / 5 |
| scab | 7 | 2 / 2 / 3 |
| other diseases and damage (cedar apple rust, powdery mildew, Marssonina blotch, leaf spot, aphids, insects) | 15 | 4 / 4 / 7 |

They were found with [`build_ood_set.py --set orchard`](../../scripts/build_ood_set.py) and picked by hand from
contact sheets, checking each title for the species: many "apple leaf" results show other trees (water apple,
sugar-apple, Philenoptera) and were skipped ([`selection.json`](selection.json)). Most Commons photos tagged
"apple scab" show scabbed fruit rather than leaves, which is why the scab category is the smallest.
[`split_sets.py`](../../scripts/split_sets.py) split them once with a fixed seed ([`split.json`](split.json)):
prototype photos help build the input check's prototypes, tune photos join the threshold choice, test photos are
only reported.

Result on the 15 test photos: 1 turned away (6.7%; 95% interval 1.2 to 29.8), a scab photo; 1 of the 2 scab tune
photos was turned away too ([`results/input_check/evaluation.md`](../../results/input_check/evaluation.md)). With so
few photos these rates are uncertain, and field photos of scab are the ones the check handles worst.

Credits: [`CREDITS.md`](CREDITS.md) and [`credits.json`](credits.json); each photo keeps its own licence.

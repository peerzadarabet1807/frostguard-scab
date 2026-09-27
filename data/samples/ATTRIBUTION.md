# Sample leaf images

These five photographs come from the **PlantVillage dataset** (Hughes & Salathé, 2015,
[arXiv:1511.08060](https://arxiv.org/abs/1511.08060)), via
[spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset), licensed
**[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)**. They are redistributed unmodified
under that licence; the repository's MIT licence does not apply to them.

| File | Original file (`raw/color/…`) | Note |
|---|---|---|
| `scab_leaf_1.jpg` | `Apple___Apple_scab/058d5e64-2c57-45ba-94cb-ac83fd1885a0___FREC_Scab 3181.JPG` | held out for validation (not trained on) |
| `scab_leaf_2.jpg` | `Apple___Apple_scab/270dd6be-95f4-4fd1-8423-b698663b3f72___FREC_Scab 2954.JPG` | held out for validation (not trained on) |
| `scab_leaf_3.jpg` | `Apple___Apple_scab/52d5723a-b498-4759-89a1-a7d815689716___FREC_Scab 3034.JPG` | held out for validation (not trained on) |
| `scab_leaf_4.jpg` | `Apple___Apple_scab/5cd3df15-31a2-4076-9d1a-8735fa1dccdf___FREC_Scab 3410.JPG` | held out for validation (not trained on) |
| `healthy_leaf_1.jpg` | `Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG` | negative control |

The four scab leaves are the images the Colab run held out as its validation split
(see `runs/scab_yolo11n_pred/`), so the detector never trained on them.

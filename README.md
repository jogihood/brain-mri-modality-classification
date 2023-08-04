# Brain MRI Modality(T1, T2, FLAIR) Classification with Modified ResNet-50

## Usage
1. Preprocess your MRI images(3D NIfTI) with ```python scripts/preprocess_images.py```
2. Train your model with ```python scripts/train.py```
3. Test your model with ```python scripts/test.py default oasis3```

## Datasets
|Dataset|Modality|Details|
|------------|-------|-------------------|
| ADNI1      | T1    | MPRAGE            |
| ADNI1      | T2    | T2-FSE            |
| ADNI2      | T1    | MPRAGE            |
| ADNI2      | FLAIR | FLAIR (Axial)     |
| ADNI3      | T1    | MPRAGE (Sagittal) |
| ADNI3      | FLAIR | FLAIR (Sagittal)  |
| ADNIGO     | T1    | MPRAGE            |
| ADNIGO     | FLAIR | FLAIR (Axial)     |
| CAMCAN     | T1    |                   |
| CAMCAN     | T2    |                   |
| IXI        | T1    |                   |
| IXI        | T2    |                   |
| Kirby-21   | T1    | MPRAGE            |
| Kirby-21   | T2    |                   |
| Kirby-21   | FLAIR |                   |
| MICCAI2017 | T1    |                   |
| MICCAI2017 | FLAIR |                   |
| MICCAI2018 | T1    |                   |
| MICCAI2018 | T2    |                   |
| MICCAI2018 | FLAIR |                   |

## References
- Çinar, A., & Yildirim, M. (2020). Detection of tumors on brain MRI images using the hybrid convolutional neural network architecture. *Medical hypotheses*, 139, 109684.

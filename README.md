# CMT-AMAI24paper

## Paper

**Quantifying Knee Cartilage Shape and Lesion: From Image to Metrics**

[AMAI’24](https://sites.google.com/view/amai2024/home) (MICCAI workshop) (in press)

![paper-CMT](README.assets/paper-CMT.png)

## TL;DR

[CMT](https://github.com/YongchengYAO/CartiMorph-Toolbox), a toolbox for knee MRI analysis, model training, and visualization.

## Contributions

- Joint Template-Learning and Registration Mode – **CMT-reg**
- CartiMorph Toolbox (CMT)

## Models for CMT

- [models](https://github.com/YongchengYAO/CMT-AMAI24paper/tree/main/Models) (both for segmentation and registration) for this work – can be loaded into CMT
- more models from the [CMT models page](https://github.com/YongchengYAO/CartiMorph-Toolbox/blob/main/Models/model_releases.md)

## Data

[Data for this repo](https://drive.google.com/drive/folders/1x_8vAgq8NRCKCoVBl-Y5jlk_kvfaYCdt?usp=sharing) for model training, inference, and evaluation

```
# data folder structure
├── Code
  ├── Aladdin
    ├── Model
  ├── LapIRN
    ├── Model
├── Data
  ├── Aladdin
  ├── CMT_data4AMAI 
  ├── LapIRN
```

- How to use files in the `Data` folder?
  1. clone this repo: `CMT-AMAI24paper`
  2. put the `Data` folder under `CMT-AMAI24paper/`
- How to use files in the `Code` folder?
  1. clone this repo: `CMT-AMAI24paper`
  2. put corresponding `Model` folders to
     - `CMT-AMAI24paper/Code/Aladdin/`
     - `CMT-AMAI24paper/Code/LapIRN/`

## Code

We compared the proposed **CMT-reg** with Aladdin and LapIRN.

- [Code](https://github.com/YongchengYAO/CMT-AMAI24paper/tree/main/Code/Aladdin/Study) for Aladdin training, inference, and evaluation
- [Code](https://github.com/YongchengYAO/CMT-AMAI24paper/tree/main/Code/LapIRN/Study) for LapIRN training, inference, and evaluation
- [Code](https://github.com/YongchengYAO/CMT-AMAI24paper/tree/main/Code/CMT_code4AMAI/study) for CMT evaluation (for reproducing the results in Table 3)
- Training, inference, and evaluation of **CMT-reg** are implemented in CMT, set these parameters in CMT:
  - *Cropped Image Size*: 64, 128, 128
  - *Training Epoch*: 2000
  - *Network Width*: x3
  - *Loss*: MSE+LNCC

## Raw Data

Data Sources:

- MR Image: [OAI](https://nda.nih.gov/oai/)
- Annotation: [OAI-ZIB](https://pubdata.zib.de)

Data Information: [here](https://github.com/YongchengYAO/CMT-AMAI24paper/tree/main/DataInfo/OAIZIB) (link CMT-ID to OAI-SubjectID)

## Citation

```
(paper in press)
```

## Acknowledgment

The training, inference, and evaluation code for Aladdin and LapIRN are adapted from these GitHub repos:

- Aladdin: https://github.com/uncbiag/Aladdin
- LapIRN: https://github.com/cwmok/LapIRN

CMT is based on CartiMorph: https://github.com/YongchengYAO/CartiMorph

```
@article{YAO2024103035,
title = {CartiMorph: A framework for automated knee articular cartilage morphometrics},
journal = {Medical Image Analysis},
author = {Yongcheng Yao and Junru Zhong and Liping Zhang and Sheheryar Khan and Weitian Chen},
volume = {91},
pages = {103035},
year = {2024},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2023.103035}
}
```


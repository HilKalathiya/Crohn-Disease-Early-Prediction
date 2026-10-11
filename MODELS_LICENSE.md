# Terms for the model weights and derived data

The MIT licence in `LICENSE` covers the code and documentation only. The files below are derived from third-party medical image datasets, so they carry the terms of those datasets instead. The sample images in `app/samples/` keep their original licence (CC BY 4.0: Kvasir-Capsule, HyperKvasir, LIMUC), with the attribution below.

## What these terms cover

| Files | Derived from |
| --- | --- |
| `app/models/crohnipi_*.onnx`, `crohnipi_*_head.npz`, `demo_study_cache.json` | CrohnIPI (CC BY-NC-ND, research use only) |
| `app/models/uc_mayo_head*`, `sb_findings_head*`, `backbone_effb0_ra.onnx` | LIMUC (CC BY 4.0), SEE-AI (Kaggle dataset licence), ImageNet-pretrained EfficientNet-B0 (timm) |
| `app/models/flare_*` | HMP2 / IBDMDB |
| `results/`, `app/models/results_summary.json`, `video_*.json` | All of the datasets below |

## Terms

1. **Non-commercial research and teaching only.** You may use, copy and share these files for non-commercial research and teaching. Commercial use is not permitted, because the main training set (CrohnIPI) is licensed for non-commercial research use only.
2. **Attribution.** Cite this repository and the source datasets (list below) in any work that uses these files.
3. **No clinical use.** CrohnSight is a research prototype, not a medical device. Do not use its outputs to diagnose, grade or treat any patient.
4. **Same terms for derivatives.** Models fine-tuned from these weights, and results computed with them, keep these terms.
5. **No warranty.** The files are provided as is, without warranty of any kind.
6. **The datasets' own licences still apply.** Nothing here grants rights the dataset licences do not give. The raw datasets are not in this repository; obtain them from their official sources.

## Source datasets to cite

- CrohnIPI: Vallée R., de Maissin A., Coutrot A., Mouchère H., Bourreille A., Normand N. *CrohnIPI: An endoscopic image database for the evaluation of automatic Crohn's disease lesions recognition algorithms.* Proc. SPIE 11317, Medical Imaging 2020. https://crohnipi.ls2n.fr/ (CC BY-NC-ND, research use only)
- Kvasir-Capsule: Smedsrud P.H. et al. *Kvasir-Capsule, a video capsule endoscopy dataset.* Scientific Data 8, 142 (2021). https://osf.io/dv2ag/ (CC BY 4.0)
- SEE-AI: Yokote A. et al. *The SEE-AI project dataset.* DEN Open (2024). Kaggle `capsuleyokote/the-see-ai-project-dataset`
- LIMUC: Polat G. et al. *Labeled Images for Ulcerative Colitis.* Zenodo 5827695 (CC BY 4.0)
- HyperKvasir: Borgli H. et al. *HyperKvasir.* Scientific Data 7, 283 (2020). https://osf.io/mh9sj/ (CC BY 4.0)
- HMP2 / IBDMDB: Lloyd-Price J. et al. Nature 569, 655–662 (2019). https://ibdmdb.org

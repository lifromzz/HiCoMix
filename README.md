# HiCoMix

Official PyTorch implementation of **HiCoMix** — hierarchical consistency mixup for whole-slide image (WSI) multiple-instance learning (MIL), combining OT-guided bag matching with prototype-level mixup.

This repository is a cleaned GitHub release of the training / preprocessing code. 

## Environment

```bash
conda create -n hicomix python=3.9
conda activate hicomix
pip install -r requirements.txt
# Install a CUDA build of PyTorch that matches your driver:
# https://pytorch.org/get-started/locally/
```

## Data layout (prepare locally)

HiCoMix expects **pre-extracted bag features** (one `.pt` per slide) and **K-fold split spreadsheets**. Place them outside git (or keep them local; they are gitignored):

```text
./dataset/{DATANAME}/uni_level1_256/*.pt
./splits/{DATANAME}/Kfold_4_seed0.xlsx   # sheets: train_fold{k}, val_fold{k}, test_fold{k}
./teacher_checkpoints/{MODEL}/{DATANAME}/checkpoint/{MODEL}_val_best_seed{k}.pth
./ot_table/   # written by preprocess.py
./train/      # written by main.py
```

Supported datasets in the code: `TCGA-NSCLC`, `CPTAC`, `TCGA-RCC`.  
Supported MIL backbones: `abmil`, `dsmil`, `ilra` (plus optional `wikg` / `LinMIL` / `gdamil`).

Override paths with environment variables if needed:

| Variable | Default | Meaning |
|---|---|---|
| `HICOMIX_DATASET_ROOT` | `./dataset` | Bag feature root |
| `HICOMIX_EXCEL_ROOT` | `./splits` | Split spreadsheet root |
| `HICOMIX_OT_TABLE_ROOT` | `./ot_table` | Precomputed OT neighbor tables |
| `HICOMIX_TEACHER_CKP_ROOT` | `./teacher_checkpoints` | Teacher checkpoints for preprocess |

Split spreadsheet columns: `file_name`, `label`.

> **Note:** Split CSV/XLSX files are intentionally **not** shipped with this repository.

## Quick start

```bash
# 1) Build OT neighbor tables (once per model / dataset / fold / k)
python preprocess.py --model_name abmil --dataname TCGA-NSCLC --fold 0 --encoder uni_level1_256

# 2) Train + evaluate (edit loops in main.py as needed)
python main.py
```

Default training setting in `main.py` / `config.py`:

- encoder: `uni_level1_256`
- `n_clusters=8`, `rand_n_candidates=8`, `proto_k=2`
- bag pairing: OT nearest neighbors
- mix type: `proto_baga+proto_bagb`

Outputs are written under `./train/...`.

## Repository layout

```text
HiCoMix-github/
├── main.py / main_ssl.py     # training entry
├── preprocess.py             # IG + k-means + OT table construction
├── config.py / config_base.py
├── data.py / engine.py / eval.py
├── mix_strategies.py / utils.py / coco_utils.py / loss_fn.py
├── Model_Club/               # ABMIL / DSMIL / ILRA / ...
├── eval_split_csvs.py        # optional split-wise re-evaluation helper
└── efficiency_records/       # optional runtime / memory profiling
```

## Citation

If you use this code, please cite the HiCoMix paper (bibtex will be added upon publication).

## License

Research use only unless otherwise stated by the authors.

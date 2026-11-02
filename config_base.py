# config_base.py
import argparse
import os
import os.path
from os.path import join
import torch

numclass = {
    "TCGA-NSCLC": 2,
    "CPTAC": 3,
    "TCGA-RCC": 3,
}

# Paths can be overridden by environment variables for local / cluster setups.
#   HICOMIX_DATASET_ROOT  — root of pre-extracted bag features (.pt)
#   HICOMIX_OT_TABLE_ROOT — root of precomputed OT neighbor tables
#   HICOMIX_EXCEL_ROOT    — root of K-fold split spreadsheets (.xlsx)
#   HICOMIX_TEACHER_CKP_ROOT — root of teacher checkpoints used by preprocess
DATASET_ROOT = os.environ.get("HICOMIX_DATASET_ROOT", "./dataset")
OT_TABLE_ROOT = os.environ.get("HICOMIX_OT_TABLE_ROOT", "./ot_table")
EXCEL_ROOT = os.environ.get("HICOMIX_EXCEL_ROOT", "./splits")
TEACHER_CKP_ROOT = os.environ.get(
    "HICOMIX_TEACHER_CKP_ROOT", "./teacher_checkpoints"
)

# Feature-extractor / encoder subdirectory under DATASET_ROOT/{dataname}/
ENCODER_STAGE_SUBDIR = {
    "ssl_BT_r50_level1_224": "ssl_BT_r50_level1_224/stage3",
    "uni_level1_256": "uni_level1_256",
}
ENCODER_CHOICES = tuple(ENCODER_STAGE_SUBDIR.keys())

ENCODER_FEATURE_DIM = {
    "ssl_BT_r50_level1_224": 1024,
    "uni_level1_256": 1024,
}


def default_feature_dim(encoder: str) -> int:
    if encoder not in ENCODER_FEATURE_DIM:
        raise ValueError(
            f"未知 encoder={encoder!r}，可选：{list(ENCODER_FEATURE_DIM.keys())}"
        )
    return ENCODER_FEATURE_DIM[encoder]


def get_data_path(dataname: str, encoder: str) -> str:
    """Return the instance-feature .pt root for a dataset + encoder."""
    if encoder not in ENCODER_STAGE_SUBDIR:
        raise ValueError(
            f"未知 encoder={encoder!r}，可选：{list(ENCODER_STAGE_SUBDIR.keys())}"
        )
    return join(DATASET_ROOT, dataname, ENCODER_STAGE_SUBDIR[encoder])


def get_excel_path(dataset, seed):
    return join(EXCEL_ROOT, dataset, f"Kfold_4_seed{seed}.xlsx")


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"{folder} created")
    else:
        print(f"{folder} exist")


def get_argp(run_root, model_name, dataname, seed, fold, encoder="uni_level1_256"):
    argp = argparse.ArgumentParser()
    argp.add_argument("--model_name", default=model_name, type=str)
    argp.add_argument("--dataname", default=dataname, type=str)
    argp.add_argument(
        "--encoder",
        default=encoder,
        type=str,
        choices=list(ENCODER_CHOICES),
        help="实例特征来源（对应 Dataset 下子目录，与训练输出路径中的 encoder 段一致）",
    )

    argp.add_argument("--data", default=get_data_path(dataname, encoder), type=str)
    argp.add_argument("--excel_path", default=get_excel_path(dataname, 0), type=str)
    argp.add_argument("--train_sheet", default=f"train_fold{fold}", type=str)
    argp.add_argument("--val_sheet", default=f"val_fold{fold}", type=str)
    argp.add_argument("--test_sheet", default=f"test_fold{fold}", type=str)

    # --- General Settings ---
    argp.add_argument("--seed", default=seed, type=int)
    argp.add_argument("--fold", default=fold, type=int)
    argp.add_argument("--start_fold", default=0, type=int)
    argp.add_argument("--end_fold", default=4, type=int)

    argp.add_argument("--num_workers", default=0, type=int)
    argp.add_argument("--pin_memory", default=False, type=bool)

    # --- Data Settings ---
    argp.add_argument("--num_classes", default=numclass[dataname], type=int)
    argp.add_argument(
        "--feature_dim",
        default=default_feature_dim(encoder),
        type=int,
    )

    # --- Training Settings ---
    argp.add_argument("--batch_size", default=1, type=int)
    argp.add_argument("--epochs", default=50, type=int)
    argp.add_argument("--lr", default=1e-4, type=float)
    argp.add_argument("--weight_decay", default=1e-5, type=float)

    # train/{run_tag}/{model_name}/{dataname}/{encoder}/checkpoint | log | *.csv
    run_leaf = join(run_root, model_name, dataname, encoder)
    checkpoint_dir = join(run_leaf, "checkpoint")
    create_folder(checkpoint_dir)
    argp.add_argument("--checkpoint_dir", default=checkpoint_dir, type=str)
    argp.add_argument(
        "--val_best_ckp",
        default=join(checkpoint_dir, f"{model_name}_val_best_fold{fold}.pth"),
    )
    argp.add_argument(
        "--train_last_ckp",
        default=join(checkpoint_dir, f"{model_name}_last_best_fold{fold}.pth"),
    )

    metic_dir = run_leaf
    create_folder(metic_dir)

    eval_tag_default = "clean"
    argp.add_argument(
        "--eval_tag",
        default=eval_tag_default,
        type=str,
        help="指标 CSV 文件名后缀，区分不同测试评估模式（无干扰/随机删/扰动/老师删等）",
    )
    argp.add_argument("--metic_dir", default=metic_dir, type=str)
    argp.add_argument("--log_dir", default=f"{metic_dir}/log", type=str)
    argp.add_argument(
        "--val_best_csv_path",
        default=join(metic_dir, f"val_{dataname}_{eval_tag_default}.csv"),
    )
    argp.add_argument(
        "--train_last_csv_path",
        default=join(metic_dir, f"train_{dataname}_{eval_tag_default}.csv"),
    )

    argp.add_argument("--device", default="cuda")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{DEVICE} is available")
    return argp

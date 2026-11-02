# main.py
"""
HiCoMix 主入口。

使用方式：

  # 步骤 1：预处理（每折运行一次）
  python preprocess.py --model_name abmil --dataname TCGA-NSCLC --fold 0

  # 步骤 2：训练
  python main.py

训练输出目录（相对于 ./train/{run_root}/）：
  {model_name}/{dataname}/{encoder}/checkpoint
  {model_name}/{dataname}/{encoder}/log
  {model_name}/{dataname}/{encoder}/val_*.csv 等

encoder 取 config_base.ENCODER_CHOICES：uni_level1_256、ssl_BT_r50_level1_224。

GPU：优先在启动 Python 前执行 export CUDA_VISIBLE_DEVICES=0；
勿在 import torch 之后再改该变量。
"""

import os
import sys

import torch
from colorama import Fore, init

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_train_args
from data import get_loaders
from engine import load_student_checkpoint, train_ig_ot_mixup
from eval import evaluate

init(autoreset=True)


def load_student_model(model_name, num_classes, device="cuda", feat_dim=1024):
    """加载 student MIL 模型。"""
    if model_name == "abmil":
        from Model_Club.abmil import ABMIL

        model = ABMIL(n_classes=num_classes, feat_dim=feat_dim)
    elif model_name == "dsmil":
        from Model_Club.DSMIL import DSMIL

        model = DSMIL(num_cls=num_classes, dim_in=feat_dim, dim_emb=feat_dim)
    elif model_name == "wikg":
        from Model_Club.WiKG import WiKG

        model = WiKG(dim_in=feat_dim, n_classes=num_classes)
    elif model_name in ("LinMIL", "linmil"):
        from Model_Club.LinMIL import LinMIL

        model = LinMIL(in_dim=feat_dim, num_classes=num_classes, embed_dim=512)
    elif model_name == "gdamil":
        from Model_Club.GDAMIL import GDAMIL

        model = GDAMIL(in_dim=feat_dim, num_classes=num_classes)
    else:
        from Model_Club.ILRA import ILRA

        model = ILRA(n_classes=num_classes, feat_dim=feat_dim)
    return model.to(device)


def run(run_root, model_name, dataname, fold, encoder="uni_level1_256", n_clusters=8):
    """单折：配置 → 数据 → 训练 → 测试评估。"""
    config = get_train_args(
        run_root,
        model_name,
        dataname,
        fold,
        encoder=encoder,
        n_clusters=n_clusters,
    )

    train_loader, val_loader, test_loader = get_loaders(config)
    mil_model = load_student_model(
        model_name, config.num_classes, config.device, feat_dim=config.feature_dim
    )
    train_ig_ot_mixup(mil_model, train_loader, val_loader, config)

    print("\n--- 测试集评估（最优权重）---")
    load_student_checkpoint(config.val_best_ckp, mil_model, device=config.device)
    test_auc, test_acc = evaluate(
        mil_model,
        test_loader,
        csv_path=config.val_best_csv_path,
        config=config,
        mean=(fold == config.end_fold - 1),
    )
    print(Fore.RED + f" Test ACC: {test_acc:.4f}  Test AUC: {test_auc:.4f}")


def run_fold(run_root, model_name, dataname, encoder="uni_level1_256", n_clusters=8):
    for fold in range(4):
        run(run_root, model_name, dataname, fold, encoder=encoder, n_clusters=n_clusters)


def main(model_name, encoder="uni_level1_256", n_clusters=8):
    run_root = (
        f"hicomix/cluster({n_clusters})_candidates8_proto2_"
        f"ot_nearest_mix(proto_baga+bagb)"
    )
    for dataname in ("CPTAC", "TCGA-RCC", "TCGA-NSCLC"):
        run_fold(
            run_root,
            model_name,
            dataname,
            encoder=encoder,
            n_clusters=n_clusters,
        )


if __name__ == "__main__":
    for model in ("abmil", "dsmil", "ilra"):
        main(model)

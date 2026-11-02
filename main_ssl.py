# main.py
"""
HiCoMix (SSL encoder) 主入口。

使用方式：

  # 步骤 1：预处理（每折运行一次）
  python preprocess.py --model_name abmil --dataname TCGA-NSCLC --fold 0

  # 步骤 2：训练
  python main.py

训练输出目录（相对于 ./train/{run_root}/）：
  {model_name}/{dataname}/{encoder}/checkpoint
  {model_name}/{dataname}/{encoder}/log
  {model_name}/{dataname}/{encoder}/val_*.csv 等

encoder 取 config_base.ENCODER_CHOICES：prov_level1_256、ssl_BT_r50_level1_224。
可在本文件中用 ACTIVE_ENCODERS 限制只跑部分编码器。

GPU：优先在启动 Python 前执行 export CUDA_VISIBLE_DEVICES=0；
勿在 import torch 之后再改该变量。若未设置，则此处默认使用物理 GPU 0。
"""

import os

import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
from data import get_loaders
from engine import train_ig_ot_mixup, load_student_checkpoint
from eval import evaluate
from colorama import init, Fore
# --- 1. 配置 ---
from config import get_train_args


init(autoreset=True)


# =====================================================================
# 模型加载（student 模型）
# =====================================================================

def load_student_model(model_name, num_classes, device="cuda", feat_dim=1024):
    """加载 student MIL 模型。"""
    if model_name == "abmil":
        from Model_Club.abmil import ABMIL
        model = ABMIL(n_classes=num_classes, feat_dim=feat_dim)
    elif model_name == "dsmil":
        from Model_Club.DSMIL import DSMIL
        model = DSMIL(num_cls=num_classes, dim_in=feat_dim, dim_emb=feat_dim)
    else:
        from Model_Club.ILRA import ILRA
        model = ILRA(n_classes=num_classes, feat_dim=feat_dim)
    return model.to(device)


# =====================================================================
# 单折训练
# =====================================================================

def run(run_root, model_name, dataname, fold, encoder):
    """
    完整的单折训练流程：
      1. 配置生成
      2. 预处理（如 OT 表不存在则自动触发）
      3. 数据加载
      4. 模型训练
      5. 测试评估
    """


    # n_clusters 可改为 2/4/8/16，需与 preprocess 预计算 OT 表时一致
    config = get_train_args(
        run_root, model_name, dataname, fold, n_clusters=k, encoder=encoder
    )
    # config.top_k = k
    config.n_clusters=k
    config.mix_align_mode = "cluster_nn"
    config.mix_lam_mode = "uniform"
    config.cluster_nn_k = k
    # 若需启用 batch 训练（含 fgw），请同时固定包长：
    # config.fixed_bag_size = 256
    # config.batch_size = 4

    # --- 2. 数据加载（OT 表已由 preprocess.py 预先生成）---
    train_loader, val_loader, test_loader = get_loaders(config)

    # --- 4. 加载 Student 模型与可选的实例对齐模块 ---
    mil_model = load_student_model(
        model_name, config.num_classes, config.device, feat_dim=config.feature_dim
    )

    # --- 5. 训练 ---
    train_ig_ot_mixup(mil_model, train_loader, val_loader, config)

    # --- 6. 测试评估 ---
    print("\n--- 测试集评估（最优权重）---")
    load_student_checkpoint(
        config.val_best_ckp, mil_model, device=config.device
    )
    test_auc, test_acc = evaluate(
        mil_model, test_loader,
        csv_path=config.val_best_csv_path,
        config=config,
        mean=(fold == config.end_fold - 1)
    )
    print(Fore.RED + f" Test ACC: {test_acc:.4f}  Test AUC: {test_auc:.4f}")


def run_fold(run_root, model_name, dataname, encoder):
    for fold in range(4):
        run(run_root, model_name, dataname, fold, encoder)


def main(model_name):
    # run_root = f"topk/cluster(2)_ot_top({k})_mix(a+b)+orga"
    run_root = f"coco/cluster({k})_cluster{2}_ot_cons_weight{0.2}"
    # 输出目录：./train/{run_root}/{model_name}/{dataname}/{encoder}/(checkpoint|log|csv)
    # run_fold(run_root, model_name, "CPTAC", "ssl_BT_r50_level1_224")
    run_fold(run_root, model_name, "TCGA-RCC", "ssl_BT_r50_level1_224")
    run_fold(run_root, model_name, "TCGA-NSCLC", "ssl_BT_r50_level1_224")


if __name__ == "__main__":
    for k in [4,8,16]:#,4,8,16
        for model in [ "abmil","dsmil","ilra"]: #"abmil", "dsmil","ilra","dsmil"
            main(model)

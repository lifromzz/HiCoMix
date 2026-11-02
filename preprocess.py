# preprocess.py
"""
预计算脚本：HiCoMix 的核心预处理流程

流程：
  1. 加载老师模型和训练集所有包
  2. 对每个包：用积分梯度找阳性实例（一次性完成）
  3. 对阳性实例分别用 n_clusters ∈ {2,4,8,16} 聚类，得到多套原型
  4. 对每套 n_clusters 分别计算 OT 距离矩阵，生成各自的排序表
  5. 保存为独立文件：ot_table_kmeans{k}_fold{fold}.pt

运行方式：
  python preprocess.py   # 全量循环所有模型/数据集/折
  python preprocess.py --model_name abmil --dataname TCGA-NSCLC --fold 0  # 单组运行
"""

import sys
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from os.path import join

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    integrated_gradients,
    find_positive_instances,
    cluster_positive_instances,
    compute_ot_distance_batch,
    load_ig_teacher_model,
    seed_everything,
    create_folder,
    build_feature_pt_stem_index,
    resolve_feature_pt_path,
)
from config_base import (
    ENCODER_CHOICES,
    OT_TABLE_ROOT,
    TEACHER_CKP_ROOT,
    default_feature_dim,
    get_data_path,
    get_excel_path,
)

# Alias kept for readability / backward compatibility with older notes.
get_teacher_model = load_ig_teacher_model

DEFAULT_ENCODER = "uni_level1_256"


# =====================================================================
# 全局常量
# =====================================================================

ALL_MODELS   = ["abmil", "dsmil", "ilra"]
ALL_DATASETS = [ "TCGA-NSCLC","CPTAC","TCGA-RCC"] #"TCGA-NSCLC",
ALL_FOLDS    = [0, 1, 2, 3]
KMEANS_LIST  = [2, 4, 8, 16]   # 需要同时生成的 n_clusters 列表

# OT：仅用 chunk_size 增大 Sinkhorn 批大小；勿用多线程并行 CUDA（PyTorch 下会极慢）
DEFAULT_OT_CHUNK_SIZE =4
DEFAULT_PARALLEL_ROW_WORKERS = 2
DEFAULT_PARALLEL_J_SPLITS = 1


# =====================================================================
# 1. 加载数据集
# =====================================================================

def load_bags_from_excel(
    data_root, excel_path, sheet_name, location="cuda", feature_dim=1024
):
    """
    从 Excel 中读取文件列表，加载所有 .pt 包文件。

    Returns:
        bags: list of (N_i, D) tensors
        labels: list of int
        slide_names: list of str
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    slide_names = df['file_name'].tolist()
    pt_labels   = df['label'].tolist()

    stem_index = build_feature_pt_stem_index(data_root)
    load_stats = {"dx_fallback": 0}
    bags, labels, names = [], [], []
    for slide_name, label in tqdm(zip(slide_names, pt_labels),
                                  total=len(slide_names), desc="Loading bags"):
        pt_path = resolve_feature_pt_path(
            data_root, slide_name, stem_index, stats=load_stats
        )
        if pt_path is None:
            print(
                f"[WARN] 找不到文件：{join(data_root, str(slide_name).strip() + '.pt')} "
                f"（已尝试去癌种后缀 / 大小写 / 同切片不同 DX），跳过。"
            )
            continue
        instances = torch.load(pt_path, map_location=location)

        if isinstance(instances, dict):
            instances = instances.get("l34", instances.get("l3"))
        if (
            instances is not None
            and instances.ndim == 2
            and instances.shape[1] > feature_dim
        ):
            instances = instances[:, :feature_dim]

        if instances is None or instances.size(0) <= 50:
            continue

        bags.append(instances)
        labels.append(label)
        names.append(slide_name)

    if load_stats.get("dx_fallback", 0) > 0:
        print(
            f"[INFO] 其中 {load_stats['dx_fallback']} 条通过「同 barcode 不同 DX」"
            f"映射到磁盘上的 .pt（Excel 与特征导出编号不一致时）。"
        )
    print(f"[INFO] 成功加载 {len(bags)} 个包。")
    return bags, labels, names


# =====================================================================
# 2. 积分梯度 + 多尺度聚类
# =====================================================================

def build_multi_scale_prototypes(model, model_name, bags, labels,
                                  kmeans_list, kmeans_iters,
                                  ig_steps, ig_threshold, device="cuda"):
    """
    对所有包：
      1. 运行积分梯度（一次，共用）
      2. 对阳性实例分别用 kmeans_list 中每个 k 做聚类
    
    Returns:
        all_prototypes: dict  {k: list of (k, D) tensors}
        all_weights:    dict  {k: list of (k,) tensors}
    """
    # 初始化各 k 的列表（直接保在 GPU 上，不转到 CPU）
    all_prototypes = {k: [] for k in kmeans_list}
    all_weights    = {k: [] for k in kmeans_list}

    for bag, label in tqdm(zip(bags, labels), total=len(bags),
                           desc="IG + Multi-scale Clustering"):
        bag = bag.to(device)

        # ---- Step 1: 积分梯度（每个包只算一次）----
        importance = integrated_gradients(
            model, bag, label,
            n_steps=ig_steps,
            device=device
        )

        # ---- Step 2: 筛选阳性实例 ----
        pos_idx, pos_importance = find_positive_instances(importance, threshold=ig_threshold)
        pos_instances = bag[pos_idx]   # (M, D)

        # ---- Step 3: 对每个 k 分别聚类，保在 GPU 上 ----
        for k in kmeans_list:
            centers, weights = cluster_positive_instances(
                pos_instances, pos_importance,
                n_clusters=k,
                max_iters=kmeans_iters,
                device=device
            )
            all_prototypes[k].append(centers)   # GPU tensor
            all_weights[k].append(weights)      # GPU tensor

    return all_prototypes, all_weights


# =====================================================================
# 3. 对单套原型计算 OT 距离矩阵，构建完整排序表
# =====================================================================

def build_ot_table(prototypes, proto_weights, slide_names,
                   ot_eps=0.5, device="cuda", chunk_size=None):
    """
    批量计算包间 OT 距离矩阵，构建完整排序表。

    对行 i 顺序遍历；对 j>i 在 compute_ot_distance_batch 内按 chunk_size 分批。
    不在此使用多线程并行 CUDA（会在 PyTorch 下严重变慢）。
    """
    if chunk_size is None:
        chunk_size = DEFAULT_OT_CHUNK_SIZE

    N = len(slide_names)
    print(f"  [OT] {N}×{N} 对 OT（顺序行 + chunk_size={chunk_size}）...")

    # 距离矩阵（上 CPU，节省显存）
    dist_matrix = torch.zeros(N, N)

    for i in tqdm(range(N), desc="  OT rows", leave=False):
        j_indices = list(range(i + 1, N))
        if not j_indices:
            continue

        row_dists = compute_ot_distance_batch(
            centers_a=prototypes[i],
            weights_a=proto_weights[i],
            centers_b_list=[prototypes[j] for j in j_indices],
            weights_b_list=[proto_weights[j] for j in j_indices],
            eps=ot_eps,
            device=device,
            chunk_size=chunk_size,
        )

        for k, j in enumerate(j_indices):
            dist_matrix[i, j] = row_dists[k]
            dist_matrix[j, i] = row_dists[k]

    print("  [OT] 距离矩阵完成，构建排序表...")
    dist_matrix.fill_diagonal_(float('inf'))

    ot_table = {}
    for i, name in enumerate(slide_names):
        row = dist_matrix[i]
        sorted_vals, sorted_indices = torch.sort(row)   # 升序（近→远）

        valid_mask     = sorted_vals != float('inf')
        sorted_vals    = sorted_vals[valid_mask]
        sorted_indices = sorted_indices[valid_mask]

        ot_table[name] = {
            "sorted_names": [slide_names[idx] for idx in sorted_indices.tolist()],
            "sorted_dists": sorted_vals.tolist(),
        }

    return ot_table


# =====================================================================
# 4. 单折预计算（支持多 k）
# =====================================================================

def preprocess_fold(model_name, dataname, fold, seed=0,
                    kmeans_list=None,
                    kmeans_iters=50,
                    ig_steps=20,
                    ig_threshold=0.5,
                    ot_eps=0.5,
                    ot_chunk_size=None,
                    parallel_j_splits=DEFAULT_PARALLEL_J_SPLITS,
                    parallel_row_workers=DEFAULT_PARALLEL_ROW_WORKERS,
                    teacher_ckp_root=TEACHER_CKP_ROOT,
                    ot_table_dir=OT_TABLE_ROOT,
                    encoder=DEFAULT_ENCODER,
                    device="cuda"):
    """
    对一个（模型, 数据集, 折）组合执行完整预计算。
    对 kmeans_list 中每个 k 值分别生成和保存 OT 表。

    保存路径格式：
      {ot_table_dir}/{model_name}/{dataname}/{encoder}/ot_table_kmeans{k}_fold{fold}.pt
    """
    if kmeans_list is None:
        kmeans_list = KMEANS_LIST

    seed_everything(seed)

    save_dir = join(ot_table_dir, model_name, dataname, encoder)
    create_folder(save_dir)

    # 检查是否所有 k 的 OT 表都已生成，若是则跳过整个折
    all_exist = all(
        os.path.exists(join(save_dir, f"ot_table_kmeans{k}_fold{fold}.pt"))
        for k in kmeans_list
    )
    if all_exist:
        print(f"  [跳过] {model_name}/{dataname}/fold{fold} 的所有 OT 表已存在。")
        return

    excel_path  = get_excel_path(dataname, seed)
    data_root   = get_data_path(dataname, encoder)
    train_sheet = f"train_fold{fold}"

    # --- Step 1: 加载数据 ---
    bags, labels, slide_names = load_bags_from_excel(
        data_root,
        excel_path,
        train_sheet,
        location=device,
        feature_dim=default_feature_dim(encoder),
    )
    if len(bags) == 0:
        print(f"  [ERROR] {dataname} fold{fold} 未加载到任何包，跳过。")
        return

    # --- Step 2: 加载老师模型 ---
    print(f"  [INFO] 加载老师模型：{model_name}...")
    teacher_model = get_teacher_model(
        model_name=model_name,
        dataname=dataname,
        seed=seed,
        teacher_ckp_root=teacher_ckp_root,
        device=device,
        feat_dim=default_feature_dim(encoder),
    )

    # --- Step 3: 积分梯度 + 多尺度聚类（一次 IG，多套聚类）---
    print(f"  [INFO] 运行 IG + 聚类（k = {kmeans_list}）...")
    all_prototypes, all_weights = build_multi_scale_prototypes(
        model=teacher_model,
        model_name=model_name,
        bags=bags,
        labels=labels,
        kmeans_list=kmeans_list,
        kmeans_iters=kmeans_iters,
        ig_steps=ig_steps,
        ig_threshold=ig_threshold,
        device=device
    )

    # --- Step 4: 对每个 k 分别计算 OT 表并保存 ---
    for k in kmeans_list:
        save_path = join(save_dir, f"ot_table_kmeans{k}_fold{fold}.pt")

        if os.path.exists(save_path):
            print(f"  [跳过] kmeans={k} 的 OT 表已存在：{save_path}")
            continue

        print(f"\n  [INFO] 计算 OT 表（kmeans={k}）...")
        if parallel_j_splits != 1 or parallel_row_workers != 1:
            print(
                "  [WARN] parallel_j_splits / parallel_row_workers 已弃用（多线程 CUDA 会极慢），"
                "已忽略；请设为 1。OT 仅通过 ot_chunk_size 调批大小。"
            )
        ot_table = build_ot_table(
            prototypes=all_prototypes[k],
            proto_weights=all_weights[k],
            slide_names=slide_names,
            ot_eps=ot_eps,
            device=device,
            chunk_size=ot_chunk_size,
        )

        torch.save(ot_table, save_path)
        print(f"  [✅] kmeans={k} OT 表已保存：{save_path}")
        print(f"       包含 {len(ot_table)} 个包，每包 {len(slide_names)-1} 个排序邻居。")


# =====================================================================
# 5. 全量循环（所有模型 × 所有数据集 × 所有折）
# =====================================================================

def preprocess_all(models=None, datasets=None, folds=None,
                   encoders=None,
                   kmeans_list=None,
                   kmeans_iters=50,
                   ig_steps=20,
                   ig_threshold=0.5,
                   ot_eps=0.5,
                   ot_chunk_size=None,
                   parallel_j_splits=DEFAULT_PARALLEL_J_SPLITS,
                   parallel_row_workers=DEFAULT_PARALLEL_ROW_WORKERS,
                   teacher_ckp_root=TEACHER_CKP_ROOT,
                   ot_table_dir=OT_TABLE_ROOT,
                   device="cuda"):
    """
    遍历所有模型 × 数据集 × 编码器 × 折，执行预计算。
    """
    if models      is None: models      = ALL_MODELS
    if datasets    is None: datasets    = ALL_DATASETS
    if folds       is None: folds       = ALL_FOLDS
    if encoders    is None: encoders    = [DEFAULT_ENCODER]
    if kmeans_list is None: kmeans_list = KMEANS_LIST

    total = len(models) * len(datasets) * len(encoders) * len(folds)
    current = 0

    for model_name in models:
        for dataname in datasets:
            for encoder in encoders:
                for fold in folds:
                    current += 1
                    print(f"\n{'='*65}")
                    print(f"  [{current}/{total}]  模型={model_name}  数据集={dataname}  encoder={encoder}  fold={fold}")
                    print(f"{'='*65}")
                    try:
                        preprocess_fold(
                            model_name=model_name,
                            dataname=dataname,
                            fold=fold,
                            kmeans_list=kmeans_list,
                            kmeans_iters=kmeans_iters,
                            ig_steps=ig_steps,
                            ig_threshold=ig_threshold,
                            ot_eps=ot_eps,
                            ot_chunk_size=ot_chunk_size,
                            parallel_j_splits=parallel_j_splits,
                            parallel_row_workers=parallel_row_workers,
                            teacher_ckp_root=teacher_ckp_root,
                            ot_table_dir=ot_table_dir,
                            encoder=encoder,
                            device=device
                        )
                    except Exception as e:
                        print(f"  [ERROR] {model_name}/{dataname}/encoder={encoder}/fold{fold} 处理失败：{e}")
                        import traceback
                        traceback.print_exc()
                        continue

    print(f"\n{'='*65}")
    print(f"  全量预计算完成！共处理 {current} 个组合。")
    print(f"{'='*65}\n")


# =====================================================================
# 6. 命令行入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HiCoMix 预计算脚本")

    # 运行模式：不指定则全量循环
    parser.add_argument("--model_name", type=str, default=None,
                        choices=["abmil", "dsmil", "ilra"],
                        help="指定单个模型（不指定则遍历所有）")
    parser.add_argument("--dataname", type=str, default=None,
                        choices=["TCGA-NSCLC", "CPTAC", "TCGA-RCC"],
                        help="指定单个数据集（不指定则遍历所有）")
    parser.add_argument("--fold", type=int, default=None,
                        help="指定单折（不指定则遍历所有折）")
    parser.add_argument(
        "--encoder",
        type=str,
        default=DEFAULT_ENCODER,
        choices=list(ENCODER_CHOICES),
        help="实例特征来源（与 Dataset 下子目录一致）",
    )

    # 算法参数
    parser.add_argument("--kmeans_list", type=int, nargs="+", default=KMEANS_LIST,
                        help="需要生成 OT 表的 n_clusters 列表，例如 --kmeans_list 2 4 8 16")
    parser.add_argument("--kmeans_iters", type=int, default=50)
    parser.add_argument("--ig_steps", type=int, default=20)
    parser.add_argument("--ig_threshold", type=float, default=0.5)
    parser.add_argument("--ot_eps", type=float, default=0.5)
    parser.add_argument(
        "--ot_chunk_size",
        type=int,
        default=DEFAULT_OT_CHUNK_SIZE,
        help="compute_ot_distance_batch 每批 j 个数；显存大时可试 128–256，OOM 则调小",
    )
    parser.add_argument(
        "--parallel_j_splits",
        type=int,
        default=DEFAULT_PARALLEL_J_SPLITS,
        help="已弃用，保留兼容；多线程 CUDA 会极慢，实际无效",
    )
    parser.add_argument(
        "--parallel_row_workers",
        type=int,
        default=DEFAULT_PARALLEL_ROW_WORKERS,
        help="已弃用，保留兼容；多线程 CUDA 会极慢，实际无效",
    )

    # 路径参数
    parser.add_argument("--teacher_ckp_root", type=str, default=TEACHER_CKP_ROOT)
    parser.add_argument("--ot_table_dir", type=str, default=OT_TABLE_ROOT)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 根据是否指定了参数，决定单组运行或全量运行
    models   = [args.model_name] if args.model_name else ALL_MODELS
    datasets = [args.dataname]   if args.dataname   else ALL_DATASETS
    folds    = [args.fold]       if args.fold is not None else ALL_FOLDS

    preprocess_all(
        models=models,
        datasets=datasets,
        folds=folds,
        encoders=[args.encoder],
        kmeans_list=args.kmeans_list,
        kmeans_iters=args.kmeans_iters,
        ig_steps=args.ig_steps,
        ig_threshold=args.ig_threshold,
        ot_eps=args.ot_eps,
        ot_chunk_size=args.ot_chunk_size,
        parallel_j_splits=args.parallel_j_splits,
        parallel_row_workers=args.parallel_row_workers,
        teacher_ckp_root=args.teacher_ckp_root,
        ot_table_dir=args.ot_table_dir,
        device=args.device
    )

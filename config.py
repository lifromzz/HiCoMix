# config.py
from os.path import join

import torch

from config_base import TEACHER_CKP_ROOT, OT_TABLE_ROOT, get_argp


def _build_eval_tag_from_test_corruption(cfg):
    """
    根据测试集干扰配置生成 eval_tag，用于区分 CSV 文件名：
      clean：三种干扰均为 0
      否则：drop{pct} / perturb{pct}_std{std} / tea{pct} 用 __ 连接（可同时多种）
    """
    td = float(getattr(cfg, "test_drop_instance_pct", 0.0) or 0.0)
    tp = float(getattr(cfg, "test_perturb_instance_pct", 0.0) or 0.0)
    ta = float(getattr(cfg, "test_drop_teacher_high_attn_pct", 0.0) or 0.0)
    std = float(getattr(cfg, "test_perturb_instance_std", 0.05) or 0.05)

    if td <= 0 and tp <= 0 and ta <= 0:
        return "clean"

    def _pct_token(prefix, v):
        s = f"{prefix}{v:g}".replace(".", "p")
        return s + "p"

    parts = []
    if td > 0:
        parts.append(_pct_token("drop", td))
    if tp > 0:
        st = str(std).replace(".", "p")
        parts.append(_pct_token("perturb", tp) + f"_std{st}")
    if ta > 0:
        parts.append(_pct_token("tea", ta))
    return "__".join(parts)


def get_train_args(
    run_root,
    model_name,
    dataname,
    fold,
    encoder="uni_level1_256",
    n_clusters = 8,
    test_drop_teacher_high_attn_pct = 0.0,
):
    """
    构造 HiCoMix 单折训练所需的 argparse Namespace（非 CLI：parse_args([])）。
    """
    run_root_train = f"./train/{run_root}"
    argp = get_argp(
        run_root_train, model_name, dataname, fold, fold, encoder=encoder
    )

    argp.add_argument(
        "--ot_table_dir",
        default=OT_TABLE_ROOT,
        type=str,
        help="OT 表根目录，完整路径为 {ot_table_dir}/{model}/{dataname}/{encoder}/ot_table_kmeans{k}_fold{f}.pt",
    )

    argp.add_argument("--ot_table_path", default=None, type=str)

  
    argp.add_argument(
        "--bag_pair_mode",
        default="ot",
        type=str,
        choices=[
            "ot",
            "mean",
            "z",
            "random",
        ],
        help="配对包 B 的邻居来源（仅选其一，在 IGOTMILDataset.__init__ 建表一次）："
             "OT 近邻/远邻、包均值余弦、老师包特征 z 余弦",
    )
    argp.add_argument(
        "--bag_pair_label_mode",
        default="default",
        type=str,
        choices=[
            "default",
            "same_class",
            "diff_class",
        ],
        help="配对包 B 的标签过滤方式：default=不过滤（原始默认采样），"
             "same_class=类内采样，diff_class=异类采样；same/diff/intra/inter 为别名",
    )
    argp.add_argument("--n_clusters", default=n_clusters, type=int,
                    help="KMeans 聚类数，需与预计算 OT 表时一致")
    argp.add_argument(
        "--rand_n_candidates",
        default=8,
        type=int,
        help="bag_pair_mode=ot_random_n_min 时，先随机采样的候选包数量 n",
    )
    argp.add_argument(
        "--select_mode",
        default="nearest",
        type=str,
        choices=["nearest", "farthest","random"],
        help="配对包 B 的邻居选择模式：nearest=最近邻，farthest=最远邻",
    )

    argp.add_argument("--proto_k", default=2, type=int, help="proto_dual 中 A/B 各自的 k-means 聚类数")
    argp.add_argument("--proto_kmeans_iters", default=10, type=int, help="proto_dual 中 k-means 迭代次数")
    argp.add_argument(
        "--proto_bagb_prob",
        default=1.0,
        type=float,
        help="mix_type=proto_baga+proto_bagb 时 proto_bagb 分支的保留概率；proto_baga 始终保留",
    )
    argp.add_argument(
        "--proto_metric",
        default="cosine",
        type=str,
        choices=["cosine", "l2"],
        help=" k-means 聚类、prototype 匹配与簇内 NN 的距离度量",
    )
    argp.add_argument(
        "--proto_pair_mode",
        default="nearest",
        type=str,
        choices=["nearest", "farthest", "random"],
        help="proto mix 时 A/B 原型配对：nearest=最近，farthest=最远，random=随机",
    )
    argp.add_argument(
        "--mix_type",
        default="proto_baga+proto_bagb",
        type=str,
        choices=[
            "nn_baga",
            "nn_bagb",
            "cat",
            "random",
            "constant",
            "nn_baga+nn_bagb",
            "proto_baga",
            "proto_bagb",
            "proto_baga+proto_bagb",
            "proto_baga+nn_baga",
            "proto_baga+nn_bagb",
            "proto_baga+proto_bagb+nn_baga",
            "proto_baga+proto_bagb+nn_bagb",
            "proto_baga+proto_bagb+nn_baga+nn_bagb",
            "instance_level_random",
            "instance_level_constant",
            "instance_level_max",
        ],
        help="NN-Mixup：baga/bagb/cat 单支；plus/+ 双支；instance_level_*=proto 聚类+指定簇内混合（max=最远实例）",
    )
  
    argp.add_argument(
        "--mix_include_original_bag",
         default=False,
        type=bool,
        help="训练时在 mixup 损失外，再对原始包 A 加一项 CE（权重 mix_aux_weight）",
    )
    argp.add_argument(
        "--mix_aux_weight",
        default=0.5,
        type=float,
        help="训练时在 mixup 损失外，再对原始包 A 加一项 CE（权重 mix_aux_weight）",
    )




    argp.add_argument(
        "--train_sample_pct",
        default=100.0,
        type=float,
        help="训练时仅使用随机采样的 p%% 个包作为 epoch 样本（按 seed 固定子集）；100 为全量",
    )
    argp.add_argument(
        "--train_label_randomize_prob",
        default=0.0,
        type=float,
        help="训练时每次 __getitem__ 以该概率将包标签随机替换（逐次采样）；0 为关闭",
    )
    argp.add_argument(
        "--train_label_corruption_pct",
        default=0.0,
        type=float,
        help="加载训练集后，将约 p%% 个包的标签固定随机化为 [0,num_classes-1]；0 为关闭（按 seed 固定子集）",
    )
    argp.add_argument(
        "--test_drop_instance_pct",
        default=0.0,
        type=float,
        help="测试集：随机删除每个包中约 p%% 的实例；0 为关闭（仅 test_sheet）",
    )
    argp.add_argument(
        "--test_perturb_instance_pct",
        default=0.0,
        type=float,
        help="测试集（SimpleDataset）：随机对约 p%% 的实例加高斯扰动；0 为关闭；训练集 IGOTMILDataset 不使用",
    )
    argp.add_argument(
        "--test_perturb_instance_std",
        default=0.05,
        type=float,
        help="上述测试集实例扰动的标准差（默认 0.05，较 0.01 更易在指标上体现差异）",
    )
    argp.add_argument(
        "--test_drop_teacher_high_attn_pct",
        default=test_drop_teacher_high_attn_pct,
        type=float,
        help="测试集：按老师注意力删除最高的约 p%% 实例；0 为关闭（仅 test_sheet，懒加载老师）",
    )
   


    argp.add_argument("--lam_scheme", default="beta", type=str)
    argp.add_argument("--beta_alpha", default=1.0, type=float)
    argp.add_argument(
        "--teacher_ckp_root",
        default=TEACHER_CKP_ROOT,
        type=str,
    )

    config = argp.parse_args([])

    if config.ot_table_path is None:
        config.ot_table_path = join(
            config.ot_table_dir,
            model_name,
            dataname,
            encoder,
            f"ot_table_kmeans{n_clusters}_fold{fold}.pt",
        )

    # 按测试集干扰模式设置 eval_tag，并更新指标 CSV 路径（与 config_base 默认 clean 区分多模式）
    config.eval_tag = _build_eval_tag_from_test_corruption(config)
    config.val_best_csv_path = join(
        config.metic_dir, f"val_{dataname}_{config.eval_tag}.csv"
    )
    config.train_last_csv_path = join(
        config.metic_dir, f"train_{dataname}_{config.eval_tag}.csv"
    )

    # 与历史 main 一致：最后 epoch 权重名为 last_fold，而非 config_base 的 last_best_fold
    config.train_last_ckp = join(
        config.checkpoint_dir, f"{model_name}_last_fold{fold}.pth"
    )

    print("-------- HiCoMix config ------------")
    print(f"\n{'='*60}")
    print(
        f"  model={model_name}  dataset={dataname}  encoder={encoder}  fold={fold}"
    )
    print(f"  data={config.data}")
    print(f"  ot_table_path={config.ot_table_path}")
    print(f"  eval_tag={config.eval_tag}  val_csv={config.val_best_csv_path}")
    print(f"{'='*60}")

    _dev = str(getattr(config, "device", "cuda")).lower()
    if _dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "无法使用 CUDA：torch.cuda.is_available() 为 False。\n"
            "说明：PyTorch 常提示「在程序启动后修改 CUDA_VISIBLE_DEVICES」，但同一报错也常见于 "
            "GPU 驱动/vGPU 异常、上一进程占满或拖垮 GPU、容器未正确映射设备等，未必与脚本里设置环境变量有关。\n"
            "建议：① 终端执行 nvidia-smi 看 GPU 是否可见、是否有僵尸进程；② 重启实例或联系平台释放 GPU；\n"
            "③ 在运行 python 之前执行 export CUDA_VISIBLE_DEVICES=0（勿在 import torch 之后再改）；\n"
            "④ 若需 CPU 调试，将 config_base.get_argp 中 --device 默认改为 cpu（大包会很慢且易 OOM）。"
        )

    return config

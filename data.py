# data.py
"""
IG-OTMix 的数据集模块。

核心类：
  - IGOTMILDataset: 训练集，加载 OT 预计算表，支持按多种邻居策略采样配对包。
  - SimpleDataset:  验证/测试集，只返回包和标签（可选测试集干扰）。
"""

import os
import random
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

OTMIX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if OTMIX_ROOT not in sys.path:
    # 为避免模块名冲突（例如 config_base），把上层目录放到末尾，让当前目录优先解析。
    sys.path.append(OTMIX_ROOT)

from utils import (
    apply_train_label_corruption,
    compute_teacher_instance_attention,
    load_ig_teacher_model,
    sample_train_subset,
)


@torch.no_grad()
def _macro_kmeans_bag(bag: torch.Tensor, k: int, iters: int, metric: str = "cosine"):
    """单包轻量 k-means，返回 cluster_id 和 prototype。"""
    if bag.dim() != 2:
        raise ValueError(f"_macro_kmeans_bag 需要 (N,D)，收到 {tuple(bag.shape)}")
    n = bag.size(0)
    k = max(1, min(int(k), n))
    iters = max(1, int(iters))
    x = bag.float()

    # deterministic farthest initialization: 先取最接近均值的点，再逐步取最远点。
    mean = x.mean(dim=0, keepdim=True)
    if metric == "cosine":
        sim_to_mean = F.cosine_similarity(x, mean, dim=1)
        first = sim_to_mean.argmax()
        centers = [x[first]]
        min_cost = 1.0 - F.cosine_similarity(x, centers[0].view(1, -1), dim=1)
        for _ in range(1, k):
            idx = min_cost.argmax()
            centers.append(x[idx])
            cost = 1.0 - F.cosine_similarity(x, x[idx].view(1, -1), dim=1)
            min_cost = torch.minimum(min_cost, cost)
    elif metric == "l2":
        first = torch.cdist(x, mean, p=2).squeeze(1).argmin()
        centers = [x[first]]
        min_cost = torch.cdist(x, centers[0].view(1, -1), p=2).squeeze(1).pow(2)
        for _ in range(1, k):
            idx = min_cost.argmax()
            centers.append(x[idx])
            cost = torch.cdist(x, x[idx].view(1, -1), p=2).squeeze(1).pow(2)
            min_cost = torch.minimum(min_cost, cost)
    else:
        raise ValueError(f"未知 cluster metric={metric}")
    centers = torch.stack(centers, dim=0)

    assign = torch.zeros(n, dtype=torch.long, device=bag.device)
    for _ in range(iters):
        if metric == "cosine":
            dist = 1.0 - F.normalize(x, dim=1) @ F.normalize(centers, dim=1).t()
        else:
            dist = torch.cdist(x, centers, p=2).pow(2)
        assign = dist.argmin(dim=1)
        new_centers = []
        for ci in range(k):
            mask = assign == ci
            if mask.any():
                new_centers.append(x[mask].mean(dim=0))
            else:
                far = dist.min(dim=1).values.argmax()
                new_centers.append(x[far])
        centers = torch.stack(new_centers, dim=0)

    prototypes = []
    for ci in range(k):
        mask = assign == ci
        if mask.any():
            prototypes.append(bag[mask].float().mean(dim=0))
        else:
            prototypes.append(centers[ci])
    return assign, torch.stack(prototypes, dim=0).to(bag.dtype)


class SimpleDataset(Dataset):
    """验证集 / 测试集：返回 (bag, label)。"""

    def __init__(self, data_root, excel_path, sheet_name, config=None):
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        self.slide_names = df["file_name"].tolist()
        self.pt_labels = df["label"].tolist()
        self.data_root = data_root
        self.config = config
        self.sheet_name = sheet_name
        self._teacher_model = None

        print(f"[SimpleDataset] 加载 {sheet_name}...")
        self.bags, self.labels = self._load(config.device)

    def _load(self, location):
        bags, labels = [], []
        for name, label in tqdm(zip(self.slide_names, self.pt_labels), total=len(self.slide_names)):
            bag = self._load_pt(name, location)
            if bag is None:
                continue
            bags.append(bag)
            labels.append(label)
        return bags, labels

    def _load_pt(self, slide_name, location):
        pt_path = os.path.join(self.data_root, slide_name + ".pt")
        alt_path = (
            pt_path.replace("_LUAD", "")
            .replace("_LUSC", "")
            .replace("_KIRC", "")
            .replace("_KIRP", "")
            .replace("_KICH", "")
        )

        if os.path.exists(pt_path):
            inst = torch.load(pt_path, map_location=location)
        elif os.path.exists(alt_path):
            inst = torch.load(alt_path, map_location=location)
        else:
            return None

        if isinstance(inst, dict):
            inst = inst.get("l34", inst.get("l3"))
            if inst is not None and inst.shape[1] > 1024:
                inst = inst[:, :1024]

        if inst is None or inst.size(0) <= 50:
            return None
        return inst

    def _get_teacher_model(self):
        """懒加载 UNI teacher，用于 test_drop_teacher_high_attn_pct。"""
        if self._teacher_model is not None:
            return self._teacher_model
        if self.config is None:
            return None
        self._teacher_model = load_ig_teacher_model(
            model_name=self.config.model_name,
            dataname=self.config.dataname,
            seed=self.config.seed,
            teacher_ckp_root=self.config.teacher_ckp_root,
            device=self.config.device,
            feat_dim=getattr(self.config, "feature_dim", 1024),
        )
        return self._teacher_model

    def _apply_test_corruption(self, bag, label):
        """仅在 test_sheet 生效：随机删实例 + 高斯扰动 + 老师高注意力删实例。"""
        if self.config is None:
            return bag
        if self.sheet_name != getattr(self.config, "test_sheet", None):
            return bag

        out = bag
        n = out.size(0)
        if n <= 1:
            return out

        # 1) 随机删实例
        drop_pct = float(getattr(self.config, "test_drop_instance_pct", 0.0) or 0.0)
        if drop_pct > 0:
            keep_n = max(1, int(round(n * (1.0 - drop_pct / 100.0))))
            keep_idx = torch.randperm(n, device=out.device)[:keep_n]
            out = out[keep_idx]
            n = out.size(0)

        # 2) 随机扰动实例
        perturb_pct = float(getattr(self.config, "test_perturb_instance_pct", 0.0) or 0.0)
        perturb_std = float(getattr(self.config, "test_perturb_instance_std", 0.05) or 0.05)
        if perturb_pct > 0 and n > 0:
            m = max(1, int(round(n * perturb_pct / 100.0)))
            idx = torch.randperm(n, device=out.device)[:m]
            noise = torch.randn_like(out[idx]) * perturb_std
            out = out.clone()
            out[idx] = out[idx] + noise

        # 3) 按老师注意力删除高注意力实例
        tea_drop = float(getattr(self.config, "test_drop_teacher_high_attn_pct", 0.0) or 0.0)
        if tea_drop > 0 and n > 1:
            teacher = self._get_teacher_model()
            if teacher is None:
                return out
            attn = compute_teacher_instance_attention(
                teacher,
                out,
                label,
                self.config.device,
                self.config.model_name,
            )
            drop_n = max(1, min(n - 1, int(round(n * tea_drop / 100.0))))
            _, drop_idx = torch.topk(attn.to(out.device), k=drop_n, largest=True)
            keep_mask = torch.ones(n, dtype=torch.bool, device=out.device)
            keep_mask[drop_idx] = False
            out = out[keep_mask]

        return out

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        bag = self._apply_test_corruption(self.bags[idx], label)
        return bag, label


class IGOTMILDataset(Dataset):
    """
    训练集数据集。

    从预计算 OT 表加载邻居信息，支持通过 bag_pair_mode 控制采样来源：
      - ot_topk
      - ot_bottomk
      - mean_topk
      - feat_topk（老师模型 bag feature z 余弦近邻）
      - random
    支持通过 bag_pair_label_mode 控制配对包标签过滤：
      - default：不按标签过滤（保持原行为）
      - same_class：类内采样
      - diff_class：异类采样
    """

    def __init__(self, data_root, excel_path, sheet_name, ot_table_path, config=None):
        self.config = config
        self.device = config.device if config else "cpu"
        self.top_k = int(getattr(config, "top_k", 32) if config else 32)
        self.bag_pair_mode = str(getattr(config, "bag_pair_mode", "ot_topk") if config else "ot_topk")
        self.bag_pair_label_mode = self._normalize_bag_pair_label_mode(
            getattr(config, "bag_pair_label_mode", "default") if config else "default"
        )
        self.rand_n_candidates = int(getattr(config, "rand_n_candidates", 8) if config else 8)
        self.train_sample_pct = float(getattr(config, "train_sample_pct", 100.0) if config else 100.0)
        self.train_label_randomize_prob = float(
            getattr(config, "train_label_randomize_prob", 0.0) if config else 0.0
        )
        self.train_label_corruption_pct = float(
            getattr(config, "train_label_corruption_pct", 0.0) if config else 0.0
        )
        self.num_classes = int(getattr(config, "num_classes", 2) if config else 2)
        self.seed = int(getattr(config, "seed", 0) if config else 0)
        self._rng = random.Random(self.seed)
        # 最近/最远选择：nearest | farthest
        self.pair_select_mode = str(getattr(config, "pair_select_mode", "nearest") if config else "nearest").lower()

        # 读取数据列表
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        self.slide_names_all = df["file_name"].tolist()
        self.pt_labels = df["label"].tolist()
        self.data_root = data_root


        print(f"[IGOTMILDataset] 加载训练集 {sheet_name}...")
        bags, labels, names = self._load(self.device)

        # train_sample_pct：按 seed 固定子集
        self.bags, self.labels, self.slide_names, keep_n, total = sample_train_subset(
            bags, labels, names, self.train_sample_pct, self._rng
        )

        if total > 0 and self.train_sample_pct < 100.0:
            print(f"[IGOTMILDataset] train_sample_pct={self.train_sample_pct:.2f}%，保留 {keep_n}/{total} 个包。")

        self.labels, n_label_corrupted = apply_train_label_corruption(
            self.labels,
            self.train_label_corruption_pct,
            self.num_classes,
            self._rng,
        )
        if n_label_corrupted > 0:
            print(
                f"[IGOTMILDataset] train_label_corruption_pct={self.train_label_corruption_pct:.2f}%，"
                f"已随机化 {n_label_corrupted}/{len(self.labels)} 个包的标签。"
            )

        # 建立 slide_name -> bag / label 的快查字典（供 sample_bag_b 使用）
        self.name2bag = {name: bag for name, bag in zip(self.slide_names, self.bags)}
        self.name2label = {name: lbl for name, lbl in zip(self.slide_names, self.labels)}
        self._name_to_idx = {name: i for i, name in enumerate(self.slide_names)}

        # 邻居表 3 选 1：ot | mean | z
        self.neighbor_table = {}
        
        if self.bag_pair_mode in {"ot", "ot_topk", "ot_random_n_min", "ot_bottomk"}:
            if not os.path.exists(ot_table_path):
                raise FileNotFoundError(
                    f"OT 表不存在：{ot_table_path}\n"
                    f"请先运行 preprocess.py 生成预计算表。"
                )
            self.neighbor_table = torch.load(ot_table_path)
            print(f"[IGOTMILDataset] 成功加载 OT 邻居表：{ot_table_path}")
        elif self.bag_pair_mode == "mean":
            self.neighbor_table = self._build_mean_topk_neighbors()
        elif self.bag_pair_mode == "z":
            self.neighbor_table = self._build_feat_topk_neighbors()


        self.proto_metadata = self._build_proto_metadata()

        print(
            f"[IGOTMILDataset] 数据加载完成，共 {len(self.bags)} 个包。"
            f" bag_pair_mode={self.bag_pair_mode}, "
            f"bag_pair_label_mode={self.bag_pair_label_mode}, top_k={self.top_k}"
        )

    @staticmethod
    def _normalize_bag_pair_label_mode(mode):
        mode = str(mode).strip().lower()
        aliases = {
            "none": "default",
            "all": "default",
            "same": "same_class",
            "intra": "same_class",
            "intra_class": "same_class",
            "diff": "diff_class",
            "different": "diff_class",
            "inter": "diff_class",
            "inter_class": "diff_class",
        }
        mode = aliases.get(mode, mode)
        valid_modes = {"default", "same_class", "diff_class"}
        if mode not in valid_modes:
            raise ValueError(
                f"未知 bag_pair_label_mode={mode}，可选：default/same_class/diff_class"
            )
        return mode

    @torch.no_grad()
    def _build_proto_metadata(self):
        """为 proto_dual 预计算每个包的 k-means assign / prototype。"""

        print(
            f"[IGOTMILDataset] 预计算 Proto-Mix 元信息: "
            f"k={self.config.proto_k}, iters={self.config.proto_kmeans_iters}, metric={self.config.proto_metric}"
        )
        metadata = {}
        for name, bag in tqdm(
            zip(self.slide_names, self.bags),
            total=len(self.slide_names),
            desc="[IGOTMILDataset] Proto metadata",
        ):
            assign, proto = _macro_kmeans_bag(bag, k=self.config.proto_k, iters=self.config.proto_kmeans_iters, metric=self.config.proto_metric)
            metadata[name] = {
                "assign": assign.long().cpu(),
                "prototype": proto.cpu(),
            }
        return metadata


    def _load(self, location):
        bags, labels, names = [], [], []
        for name, label in tqdm(zip(self.slide_names_all, self.pt_labels), total=len(self.slide_names_all)):
            bag = self._load_pt(name, location)
            if bag is None:
                continue
            bags.append(bag)
            labels.append(label)
            names.append(name)
        return bags, labels, names

    def _load_pt(self, slide_name, location):
        pt_path = os.path.join(self.data_root, slide_name + ".pt")
        alt_path = (
            pt_path.replace("_LUAD", "")
            .replace("_LUSC", "")
            .replace("_KIRC", "")
            .replace("_KIRP", "")
            .replace("_KICH", "")
        )

        if os.path.exists(pt_path):
            inst = torch.load(pt_path, map_location=location)
        elif os.path.exists(alt_path):
            inst = torch.load(alt_path, map_location=location)
        else:
            return None

        if isinstance(inst, dict):
            inst = inst.get("l34", inst.get("l3"))
            if inst is not None and inst.shape[1] > 1024:
                inst = inst[:, :1024]

        if inst is None or inst.size(0) <= 50:
            return None
        return inst

    def _build_mean_topk_neighbors(self):
        """基于 bag mean feature 的余弦相似度构建全局近邻排序。"""
        if len(self.bags) == 0:
            return {}

        means = [b.float().mean(dim=0, keepdim=True) for b in self.bags]
        means = torch.cat(means, dim=0)  # [N, D]
        means = F.normalize(means, p=2, dim=1)
        sim = means @ means.T  # [N, N]
        sim.fill_diagonal_(-1.0)

        neighbors = {}
        for i, name in enumerate(self.slide_names):
            vals, idx = torch.sort(sim[i], descending=True)
            valid = idx[vals > -1.0]
            valid_vals = vals[vals > -1.0]
            neighbors[name] = {
                "sorted_names": [self.slide_names[j] for j in valid.tolist()],
                "sorted_dists": (1.0 - valid_vals).tolist(),
            }
        print("[IGOTMILDataset] 已构建 mean_topk 邻居表。")
        return neighbors

    @torch.no_grad()
    def _build_feat_topk_neighbors(self):
        """基于老师模型 bag feature z 的余弦相似度构建全局近邻排序。"""

        teacher_model = load_ig_teacher_model(
            model_name=self.config.model_name,
            dataname=self.config.dataname,
            seed=self.config.seed,
            teacher_ckp_root=self.config.teacher_ckp_root,
            device=self.config.device,
            feat_dim=self.config.feature_dim,
        )

        z_list = []
        for bag in tqdm(self.bags, desc="[IGOTMILDataset] 计算 teacher bag feature z"):
            x = bag.to(self.config.device)
            batch_idx = torch.zeros(x.shape[0], dtype=torch.long, device=self.config.device)
            out = teacher_model({"x": x, "batch": batch_idx})
            if isinstance(out, dict) and "z" in out:
                z = out["z"]
            else:
                raise RuntimeError("老师模型前向结果不包含 'z'，无法执行 feat_topk。")
            z_list.append(z.detach().float().view(-1).cpu())

        z_feat = torch.stack(z_list, dim=0)  # [N, Dz]
        z_feat = F.normalize(z_feat, p=2, dim=1)
        sim = z_feat @ z_feat.T  # [N, N]
        sim.fill_diagonal_(-1.0)

        neighbors = {}
        for i, name in enumerate(self.slide_names):
            vals, idx = torch.sort(sim[i], descending=True)
            valid = idx[vals > -1.0]
            valid_vals = vals[vals > -1.0]
            neighbors[name] = {
                "sorted_names": [self.slide_names[j] for j in valid.tolist()],
                "sorted_dists": (1.0 - valid_vals).tolist(),
            }
        print("[IGOTMILDataset] 已构建 feat_topk（teacher z）邻居表。")
        return neighbors

    def _random_bag_with_name(self):
        idx = self._rng.randint(0, len(self.bags) - 1)
        return self.slide_names[idx], self.bags[idx], self.labels[idx]

    def _label_matches_pair_mode(self, label_a, label_b):
        if self.bag_pair_label_mode == "default" or label_a is None:
            return True
        if self.bag_pair_label_mode == "same_class":
            return label_b == label_a
        if self.bag_pair_label_mode == "diff_class":
            return label_b != label_a
        return True

    def _candidate_names_for_label_mode(self, label_a):
        return [
            name
            for name in self.slide_names
            if self._label_matches_pair_mode(label_a, self.name2label[name])
        ]

    def _random_bag_with_name_for_pair_mode(self, slide_name_a: str):
        label_a = self.name2label.get(slide_name_a)
        candidate_pool = self._candidate_names_for_label_mode(label_a)
        if not candidate_pool:
            return self._random_bag_with_name()
        chosen_name = self._rng.choice(candidate_pool)
        return chosen_name, self.name2bag[chosen_name], self.name2label[chosen_name]

    def _sample_from_neighbor_table_with_name(self, slide_name_a: str, select_mode: str):
        """
        从 slide_name_a 的邻居表中随机采样 n 个，再按最近/最远选一个。
        select_mode: nearest | farthest
        """
        row = self.neighbor_table.get(slide_name_a)
        if row is None:
            print(f"[IGOTMILDataset] 警告：{slide_name_a} 的邻居表不存在。")
            assert False

        # 三种邻居表统一为 dict: {"sorted_names": [...], "sorted_dists": [...]}
        sorted_names = row.get("sorted_names", [])
        sorted_dists = row.get("sorted_dists", [])
        if not sorted_names:
            print(f"[IGOTMILDataset] 警告：{slide_name_a} 的邻居表为空。")
            assert False

        # 过滤训练集中不存在的包，并同步维护 name/dist 对齐
        label_a = self.name2label.get(slide_name_a)
        name_dist_pairs = []
        for i, name in enumerate(sorted_names):
            if name not in self.name2bag:
                continue
            if not self._label_matches_pair_mode(label_a, self.name2label[name]):
                continue
            dist = float(sorted_dists[i]) if i < len(sorted_dists) else float("inf")
            name_dist_pairs.append((name, dist))

        if not name_dist_pairs:
            return self._random_bag_with_name_for_pair_mode(slide_name_a)
       
        n = max(1, min(self.rand_n_candidates, len(name_dist_pairs)))
        sampled_with_dist = self._rng.sample(name_dist_pairs, n)

        mode = str(select_mode).lower()
        if mode == "random":
            chosen_name = self._rng.choice([x[0] for x in sampled_with_dist])
        elif mode == "farthest":
            chosen_name = max(sampled_with_dist, key=lambda x: x[1])[0]
        else:
            chosen_name = min(sampled_with_dist, key=lambda x: x[1])[0]
        return chosen_name, self.name2bag[chosen_name], self.name2label[chosen_name]

    def sample_bag_b(self, slide_name_a: str):
        """
        从 slide_name_a 的排序列表中随机采样一个配对包。
        统一返回 (slide_name_b, bag_b, label_b, meta)。若当前策略不需要额外
        元信息，meta 为空字典；若 mix_type=proto_dual，meta["proto"] 内包含
        A/B 的预计算 cluster assign 与 prototype。

        Args:
            slide_name_a: 当前包名称
            top_k: 覆盖默认 top_k（None 时取 config.top_k）
        """
        if len(self.bags) == 0:
            raise RuntimeError("训练集为空，无法采样配对包。")

        if self.bag_pair_mode == "random":
            name_b, bag_b, label_b = self._random_bag_with_name_for_pair_mode(slide_name_a)
            return name_b, bag_b, label_b, self._sample_meta(slide_name_a, name_b)

        # 邻居表配对：来源 ot/z/mean；选择 nearest/farthest
        name_b, bag_b, label_b = self._sample_from_neighbor_table_with_name(
            slide_name_a, select_mode=self.config.select_mode
        )
        return name_b, bag_b, label_b, self._sample_meta(slide_name_a, name_b)

    def _sample_meta(self, slide_name_a: str, slide_name_b: str):
        meta = {}
        meta["proto"] = {
            "meta_a": self.proto_metadata[str(slide_name_a)],
            "meta_b": self.proto_metadata[str(slide_name_b)],
        }
        return meta

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        """返回 (bag, label, slide_name)，并可按概率随机化训练标签。"""
        bag = self.bags[idx]
        label = self.labels[idx]
        name = self.slide_names[idx]

        p = self.config.train_label_randomize_prob
        if p > 0 and self._rng.random() < p:
            label = self._rng.randint(0, self.config.num_classes - 1)

        return bag, label, name


# =====================================================================
# DataLoader 工厂函数
# =====================================================================

def get_loaders(config, test=False):
    """
    根据配置返回 DataLoader。

    训练集使用 IGOTMILDataset（需要 OT 表），验证/测试集使用 SimpleDataset。
    """
    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    if test:
        test_ds = SimpleDataset(config.data, config.excel_path, config.test_sheet, config)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        return test_loader

    train_ds = IGOTMILDataset(
        data_root=config.data,
        excel_path=config.excel_path,
        sheet_name=config.train_sheet,
        ot_table_path=config.ot_table_path,
        config=config,
    )
    val_ds = SimpleDataset(config.data, config.excel_path, config.val_sheet, config)
    test_ds = SimpleDataset(config.data, config.excel_path, config.test_sheet, config)

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(f"数据加载完成：train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader

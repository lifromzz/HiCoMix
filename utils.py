# utils.py
import re
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
from os.path import join
from typing import Dict, Optional, Set


# =====================================================================
# 1. 基本工具
# =====================================================================

# Excel 行常见癌种后缀；特征目录中的 .pt 常不带此后缀（如 ssl_BT 导出）。
CANCER_STEM_SUFFIXES = ("_LUAD", "_LUSC", "_KIRC", "_KIRP", "_KICH")


def _strip_cancer_stem_suffix(stem: str) -> str:
    for suf in CANCER_STEM_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def build_feature_pt_stem_index(data_root: str) -> Dict[str, object]:
    """
    扫描 data_root 下全部 .pt 文件名（不含扩展名），供 resolve_feature_pt_path 快速匹配。
    """
    stems = [f[:-3] for f in os.listdir(data_root) if f.endswith(".pt")]
    stem_set: Set[str] = set(stems)
    by_lower: Dict[str, str] = {}
    for s in stems:
        k = s.lower()
        if k not in by_lower:
            by_lower[k] = s
    return {"stem_set": stem_set, "by_lower": by_lower}


def resolve_feature_pt_path(
    data_root: str,
    slide_name,
    index: Dict[str, object],
    *,
    allow_dx_fallback: bool = True,
    stats: Optional[Dict[str, int]] = None,
    underscore_prefix_fallback: bool = False,
) -> Optional[str]:
    """
    将 Excel 中的 file_name 解析为磁盘上的特征 .pt 绝对路径。

    匹配顺序：
      1) 与 Excel 完全一致的 stem；
      1b) 若 underscore_prefix_fallback：取第一个 "_" 之前的子串作为 stem（与 stage3 中
          无后缀的 .pt 主干对齐，例如 TCGA-...-DXk_LUAD → TCGA-...-DXk）；
      2) 去掉末尾 _LUAD/_LUSC/… 后缀；
      3) 与 (2) 等价的链式 replace（兼容旧逻辑）；
      4) 大小写不敏感命中；
      5) TCGA 风格下，同一样本不同 DX 编号（如 Excel 为 -DX2，磁盘仅有 -DX4）时回退到
         同一前缀下字典序最小的 stem（若 stats 传入则累计 dx_fallback）。

    若仍找不到则返回 None。
    """
    slide_name = str(slide_name).strip()
    if slide_name.lower().endswith(".pt"):
        slide_name = slide_name[:-3]

    stem_set: Set[str] = index["stem_set"]  # type: ignore[assignment]
    by_lower: Dict[str, str] = index["by_lower"]  # type: ignore[assignment]

    def path_for_stem(s: str) -> str:
        return join(data_root, s + ".pt")

    def lookup_stem(s: str) -> Optional[str]:
        if s in stem_set:
            return path_for_stem(s)
        sl = s.lower()
        if sl in by_lower:
            return path_for_stem(by_lower[sl])
        return None

    # 1) 原样
    hit = lookup_stem(slide_name)
    if hit:
        return hit

    # 1b) TCGA-NSCLC 等：Excel 名在第一个 "_" 后可能带癌种等后缀，与磁盘 .pt 主干用前缀对齐
    if underscore_prefix_fallback and "_" in slide_name:
        stem_prefix = slide_name.split("_", 1)[0]
        if stem_prefix:
            hit = lookup_stem(stem_prefix)
            if hit:
                return hit

    # 2) 去掉癌种后缀
    stem_no_suf = _strip_cancer_stem_suffix(slide_name)
    if stem_no_suf != slide_name:
        hit = lookup_stem(stem_no_suf)
        if hit:
            return hit

    # 3) 链式去掉后缀（与旧 preprocess 一致）
    replaced = slide_name
    for suf in CANCER_STEM_SUFFIXES:
        replaced = replaced.replace(suf, "")
    if replaced != slide_name:
        hit = lookup_stem(replaced)
        if hit:
            return hit

    if not allow_dx_fallback:
        return None

    # 5) 同 barcode 不同 DX：…-DXk
    stem_try = _strip_cancer_stem_suffix(slide_name)
    m = re.match(r"^(.*)-(DX\d+)$", stem_try, flags=re.IGNORECASE)
    if not m:
        return None
    base, want_dx = m.group(1), m.group(2).upper()
    rx = re.compile("^" + re.escape(base) + r"-DX\d+$", re.IGNORECASE)
    flex = [s for s in stem_set if rx.match(s)]
    if not flex:
        return None
    want_full = f"{base}-{want_dx}".lower()
    chosen = None
    for s in flex:
        if s.lower() == want_full:
            chosen = s
            break
    if chosen is None:
        chosen = sorted(flex, key=str.lower)[0]
        if stats is not None and chosen.lower() != stem_try.lower():
            stats["dx_fallback"] = stats.get("dx_fallback", 0) + 1
    return path_for_stem(chosen)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'{folder} created')
    else:
        print(f"{folder} exist")


def sample_train_subset(bags, labels, names, train_sample_pct: float, rng: random.Random):
    """
    按给定比例与随机种子子采样训练集，保持 bags/labels/names 对齐。
    """
    total = len(bags)
    if total == 0:
        return [], [], [], 0, 0
    if float(train_sample_pct) >= 100.0:
        return bags, labels, names, total, total

    keep_n = max(1, int(round(total * float(train_sample_pct) / 100.0)))
    idxs = list(range(total))
    rng.shuffle(idxs)
    keep = set(idxs[:keep_n])
    out_bags = [bags[i] for i in range(total) if i in keep]
    out_labels = [labels[i] for i in range(total) if i in keep]
    out_names = [names[i] for i in range(total) if i in keep]
    return out_bags, out_labels, out_names, keep_n, total


def apply_train_label_corruption(
    labels,
    corruption_pct: float,
    num_classes: int,
    rng: random.Random,
):
    """
    将训练集中约 corruption_pct%% 的包的标签随机替换为 [0, num_classes-1]。
    按 rng 固定子集，整个训练过程不变（与 FeatTeacherDataset.label_corruption 一致）。

    Returns:
        (labels, n_corrupted)
    """
    labels = list(labels)
    n = len(labels)
    if n == 0 or float(corruption_pct) <= 0.0:
        return labels, 0

    n_classes = max(1, int(num_classes))
    k = int(round(n * float(corruption_pct) / 100.0))
    if k <= 0:
        return labels, 0
    k = min(k, n)

    idxs = rng.sample(range(n), k)
    upper = n_classes - 1
    for i in idxs:
        labels[i] = rng.randint(0, upper)
    return labels, k


# =====================================================================
# 2. 积分梯度（Integrated Gradients）
# =====================================================================

def integrated_gradients(model, bag, label, n_steps=20, device="cuda"):
    """
    使用积分梯度计算包中每个实例的重要性。

    原理：从"全零基准"插值到"真实输入"，沿路径积分梯度。
    满足完整性公理：所有实例的归因之和 = F(x) - F(baseline)

    Args:
        model: 已加载权重的老师模型（teacher），需要支持 data={"x":..,"batch":..} 的输入格式
        bag: (N, D) - 包的实例特征张量
        label: int - 此包的真实类别标签
        n_steps: int - 插值步数（越多越精确，默认20步）
        device: str

    Returns:
        importance: (N,) - 每个实例的积分梯度重要性得分（非负）
    """
    N, D = bag.shape
    baseline = torch.zeros_like(bag, device=device)  # 全零基准
    bag = bag.to(device)

    grads_list = []
    model.eval()

    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        # 插值点
        interp = (baseline + alpha * (bag - baseline)).detach().requires_grad_(True)

        # 前向传播
        batch_idx = torch.zeros(N, dtype=torch.long, device=device)
        result = model({"x": interp, "batch": batch_idx})

        # 取目标类别的包级别得分
        probs = result["probs"]  # (1, C) 或 (C,)
        if probs.dim() == 1:
            target_score = probs[label]
        else:
            target_score = probs[0, label]

        # 反向传播
        target_score.backward()
        grads_list.append(interp.grad.detach().clone())  # (N, D)

    # 平均梯度
    avg_grad = torch.stack(grads_list).mean(dim=0)  # (N, D)

    # 积分梯度 = (输入 - 基准) × 平均梯度
    ig = (bag - baseline) * avg_grad  # (N, D)

    # 每个实例的重要性 = L2 范数
    importance = ig.norm(dim=1)  # (N,)
    return importance.detach()


def find_positive_instances(importance, threshold=0.5):
    """
    根据积分梯度重要性，找出"最小充分集"。

    策略：对重要性降序排列，取累积重要性达到 threshold 的最小实例集合。

    Args:
        importance: (N,) - 每个实例的重要性得分
        threshold: float - 累积重要性比例阈值（0.5 = 覆盖50%重要性的最小集合）

    Returns:
        pos_idx: (K,) - 阳性实例的索引（按重要性从高到低排序）
        pos_importance: (K,) - 对应的重要性得分
    """
    sorted_importance, sorted_idx = torch.sort(importance, descending=True)
    cumsum = torch.cumsum(sorted_importance, dim=0)
    total = sorted_importance.sum()

    if total < 1e-8:
        # 重要性全零，退化为返回所有实例
        return sorted_idx, sorted_importance

    threshold_val = total * threshold
    mask = cumsum >= threshold_val
    if mask.any():
        k = mask.nonzero(as_tuple=False)[0].item() + 1
    else:
        k = len(importance)

    k = max(k, 1)  # 至少1个
    pos_idx = sorted_idx[:k]
    return pos_idx, importance[pos_idx]


# =====================================================================
# 3. GPU 上的 KMeans 聚类
# =====================================================================

def kmeans_pytorch(X: torch.Tensor, num_clusters: int, max_iters: int = 50,
                   tol: float = 1e-4, device="cuda"):
    """
    GPU 上的 KMeans 聚类。

    Returns:
        centers: (K, D)
        labels: (N,)
    """
    N, D = X.shape
    if N == 0:
        raise ValueError("Empty input tensor!")

    K = min(num_clusters, N)
    X = X.to(device)

    if K == 1:
        return X.mean(dim=0, keepdim=True), torch.zeros(N, dtype=torch.long, device=device)

    # 随机选 K 个初始中心
    indices = torch.randperm(N, device=device)[:K]
    centers = X[indices].clone()

    for _ in range(max_iters):
        dists = torch.cdist(X, centers, p=2)    # (N, K)
        labels = dists.argmin(dim=1)             # (N,)

        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(K, device=device)
        new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), X)
        counts.scatter_add_(0, labels, torch.ones(N, device=device))
        counts = counts.clamp(min=1)
        new_centers /= counts.unsqueeze(1)

        if torch.norm(centers - new_centers) < tol:
            break
        centers = new_centers

    return centers, labels


def cluster_positive_instances(pos_instances: torch.Tensor,
                                pos_importance: torch.Tensor,
                                n_clusters: int,
                                max_iters: int = 50,
                                device="cuda"):
    """
    对阳性实例做 KMeans 聚类，聚类的注意力权重 = 该簇中所有实例的重要性之和。

    Args:
        pos_instances: (K, D) - 阳性实例特征
        pos_importance: (K,) - 每个阳性实例的重要性得分
        n_clusters: int - 聚类数
        max_iters: int
        device: str

    Returns:
        centers: (n_clusters, D) - 类簇中心
        cluster_weights: (n_clusters,) - 类簇权重（归一化）
    """
    N = pos_instances.shape[0]
    K = min(n_clusters, N)

    if K == 1 or N <= 1:
        center = pos_instances.mean(dim=0, keepdim=True).to(device)
        weight = torch.tensor([1.0], device=device)
        return center, weight

    pos_instances = pos_instances.to(device)
    pos_importance = pos_importance.to(device)

    centers, labels = kmeans_pytorch(pos_instances, K, max_iters=max_iters, device=device)

    # 聚类权重 = 该簇内所有实例的重要性之和
    cluster_weights = torch.zeros(K, device=device)
    cluster_weights.scatter_add_(0, labels, pos_importance)

    # 归一化
    total = cluster_weights.sum().clamp(min=1e-8)
    cluster_weights = cluster_weights / total

    return centers, cluster_weights


# =====================================================================
# 4. Sinkhorn OT（Log-space，数值稳定）
# =====================================================================

def sinkhorn_log(cost: torch.Tensor,
                 mu: torch.Tensor,
                 nu: torch.Tensor,
                 eps: float = 0.1,
                 max_iters: int = 50,
                 thresh: float = 1e-6) -> torch.Tensor:
    """
    对数域 Sinkhorn 最优传输。

    Args:
        cost: (B, M, N) - 代价矩阵（batch支持）
        mu:   (B, M)    - 源分布（行边际）
        nu:   (B, N)    - 目标分布（列边际）
        eps:  float     - 正则化系数
        max_iters: int
        thresh: float   - 收敛阈值

    Returns:
        P: (B, M, N) - 最优传输矩阵
    """
    def M_fn(u, v):
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    log_mu = torch.log(mu + 1e-8)
    log_nu = torch.log(nu + 1e-8)

    for _ in range(max_iters):
        u_prev = u
        u = eps * (log_mu - torch.logsumexp(M_fn(u, v), dim=-1)) + u
        v = eps * (log_nu - torch.logsumexp(M_fn(u, v).transpose(-2, -1), dim=-1)) + v

        err = (u - u_prev).norm(dim=-1).mean()
        if err < thresh:
            break

    P = torch.exp(M_fn(u, v))
    return P


def compute_ot_distance(centers_a: torch.Tensor, weights_a: torch.Tensor,
                         centers_b: torch.Tensor, weights_b: torch.Tensor,
                         eps: float = 0.5, device="cuda") -> float:
    """
    计算两个包（由类簇代表）之间的 OT 距离（单对）。
    """
    centers_a = centers_a.to(device)
    weights_a = weights_a.to(device)
    centers_b = centers_b.to(device)
    weights_b = weights_b.to(device)

    C = torch.cdist(centers_a, centers_b, p=2) ** 2  # (Ka, Kb)
    C = C.unsqueeze(0)                                # (1, Ka, Kb)

    mu = (weights_a / weights_a.sum()).unsqueeze(0)
    nu = (weights_b / weights_b.sum()).unsqueeze(0)

    P = sinkhorn_log(C, mu, nu, eps=eps)
    return (C * P).sum().item()


def compute_ot_distance_batch(centers_a: torch.Tensor,
                               weights_a: torch.Tensor,
                               centers_b_list: list,
                               weights_b_list: list,
                               eps: float = 0.5,
                               device: str = "cuda",
                               chunk_size: int = 64) -> torch.Tensor:
    """
    批量计算：包 a 与 centers_b_list 中所有包的 OT 距离。

    由于各包的类簇数 K 可能不同，先按 Ka × Kb 分组，分组内并行；
    若所有包的 K 相同（通常如此，因为 n_clusters 固定），则完全并行。

    Args:
        centers_a:      (Ka, D)
        weights_a:      (Ka,)
        centers_b_list: list of B × (Kb, D)
        weights_b_list: list of B × (Kb,)
        eps:            正则化系数
        device:         GPU 设备
        chunk_size:     每次并行计算的包数（控制显存使用）

    Returns:
        dists: (B,) — 包 a 到每个包 b 的 OT 距离
    """
    centers_a  = centers_a.to(device)
    weights_a  = weights_a.to(device)
    B          = len(centers_b_list)
    dists      = torch.zeros(B, device=device)
    Ka, D      = centers_a.shape

    for start in range(0, B, chunk_size):
        end    = min(start + chunk_size, B)
        chunk  = centers_b_list[start:end]
        wchunk = weights_b_list[start:end]
        bs     = len(chunk)

        # 尝试统一 Kb，若不一致则回退到逐个计算
        Kb = chunk[0].shape[0]
        uniform = all(c.shape[0] == Kb for c in chunk)

        if uniform:
            # ---- 完全批量路径 ----
            # (bs, Kb, D)
            centers_b = torch.stack([c.to(device) for c in chunk])
            # (bs, Kb)
            weights_b = torch.stack([w.to(device) for w in wchunk])

            # 代价矩阵 (bs, Ka, Kb)
            ca_exp = centers_a.unsqueeze(0).expand(bs, -1, -1)  # (bs, Ka, D)
            C = torch.cdist(ca_exp, centers_b, p=2) ** 2        # (bs, Ka, Kb)

            # 归一化权重
            mu = (weights_a / weights_a.sum()).unsqueeze(0).expand(bs, -1)  # (bs, Ka)
            nu = weights_b / weights_b.sum(dim=1, keepdim=True)            # (bs, Kb)

            P = sinkhorn_log(C, mu, nu, eps=eps)                # (bs, Ka, Kb)
            dists[start:end] = (C * P).sum(dim=(-1, -2))       # (bs,)

        else:
            # ---- 回退：逐个计算 ----
            for k, (cb, wb) in enumerate(zip(chunk, wchunk)):
                cb = cb.to(device)
                wb = wb.to(device)
                C  = torch.cdist(centers_a.unsqueeze(0),
                                  cb.unsqueeze(0), p=2) ** 2   # (1, Ka, Kb)
                mu = (weights_a / weights_a.sum()).unsqueeze(0)
                nu = (wb / wb.sum()).unsqueeze(0)
                P  = sinkhorn_log(C, mu, nu, eps=eps)
                dists[start + k] = (C * P).sum()

    return dists.cpu()


def compute_ot_distances_for_pairs(
    prototypes: list,
    proto_weights: list,
    pairs: list,
    eps: float,
    device: str,
) -> torch.Tensor:
    """
    对一批 (i, j)（不要求 i<j 关系，仅作为索引对）批量计算 OT 距离。

    若该批内所有对的类簇数 Ka、Kb 分别相同，则走 (B, Ka, Kb) 的 batched Sinkhorn；
    否则对该批内各对回退到 compute_ot_distance。

    Args:
        prototypes:    list of (K, D)
        proto_weights: list of (K,)
        pairs:           list of (i, j) 索引元组
        eps:             Sinkhorn 正则
        device:          cuda / cuda:0 / cpu

    Returns:
        dists: (B,) float32 CPU
    """
    B = len(pairs)
    if B == 0:
        return torch.empty(0, dtype=torch.float32)

    i0, j0 = pairs[0]
    Ka0, Kb0 = prototypes[i0].shape[0], prototypes[j0].shape[0]
    uniform = all(
        prototypes[i].shape[0] == Ka0 and prototypes[j].shape[0] == Kb0
        for i, j in pairs
    )

    if uniform:
        centers_a = torch.stack([prototypes[i].to(device) for i, j in pairs])
        centers_b = torch.stack([prototypes[j].to(device) for i, j in pairs])
        wa = torch.stack([proto_weights[i].to(device) for i, j in pairs])
        wb = torch.stack([proto_weights[j].to(device) for i, j in pairs])
        C = torch.cdist(centers_a, centers_b, p=2) ** 2
        mu = wa / wa.sum(dim=1, keepdim=True)
        nu = wb / wb.sum(dim=1, keepdim=True)
        P = sinkhorn_log(C, mu, nu, eps=eps)
        return (C * P).sum(dim=(-1, -2)).detach().float().cpu()

    out = torch.empty(B, dtype=torch.float32)
    for t, (i, j) in enumerate(pairs):
        out[t] = compute_ot_distance(
            prototypes[i],
            proto_weights[i],
            prototypes[j],
            proto_weights[j],
            eps=eps,
            device=device,
        )
    return out


# =====================================================================
# 5. 模型加载工具
# =====================================================================

def get_teacher_ckp(teacher_ckp_root, model_name, dataset):
    from os.path import join
    return join(teacher_ckp_root, model_name, dataset, "checkpoint",
                f"{model_name}_val_best")


def load_ig_teacher_model(
    model_name, dataname, seed, teacher_ckp_root, device="cuda", feat_dim=1024
):
    """加载老师模型。"""
    from config_base import numclass
    nclass = numclass[dataname]

    if model_name == "abmil":
        from Model_Club.abmil import ABMIL
        model = ABMIL(n_classes=nclass, feat_dim=feat_dim)
    elif model_name == "dsmil":
        from Model_Club.DSMIL import DSMIL
        model = DSMIL(num_cls=nclass, dim_in=feat_dim, dim_emb=feat_dim)
    elif model_name == "ilra":
        from Model_Club.ILRA import ILRA
        model = ILRA(n_classes=nclass, feat_dim=feat_dim)
    elif model_name == "wikg":
        from Model_Club.WiKG import WiKG
        model = WiKG(dim_in=feat_dim, n_classes=nclass)
    elif model_name in ("LinMIL", "linmil"):
        from Model_Club.LinMIL import LinMIL
        model = LinMIL(in_dim=feat_dim, num_classes=nclass, embed_dim=512)
    elif model_name == "gdamil":
        from Model_Club.GDAMIL import GDAMIL
        model = GDAMIL(in_dim=feat_dim, num_classes=nclass)
    else:
        from Model_Club.ILRA import ILRA
        model = ILRA(n_classes=nclass, feat_dim=feat_dim)

    tbc = get_teacher_ckp(teacher_ckp_root, model_name, dataname)
    ckp_path = "{}_seed{}.pth".format(tbc, seed)
    ckp = torch.load(ckp_path, map_location=device)["state_dict"]
    model.load_state_dict(ckp)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_teacher_bag_z(teacher_model, bag: torch.Tensor, device: str):
    """
    将整包实例特征送入老师 MIL 模型，取返回的 bag 级特征 z（与 IG / preprocess 一致）。

    Args:
        teacher_model: get_teacher_model 加载的模型
        bag: (N, D) 实例特征
        device: 推理设备

    Returns:
        z: (D_z,) float32 CPU，用于与其它包的 z 做余弦相似度
    """
    teacher_model.eval()
    bag = bag.to(device)
    n = bag.shape[0]
    batch_idx = torch.zeros(n, dtype=torch.long, device=device)
    out = teacher_model({"x": bag, "batch": batch_idx})
    z = out["z"]
    return z.detach().float().cpu().view(-1)


@torch.no_grad()
def compute_teacher_instance_attention(
    teacher_model, bag: torch.Tensor, label, device: str, model_name: str
):
    """
    老师 MIL 前向得到的实例级注意力（用于删除高注意力实例等）。

    Args:
        bag: (N, D)
        label: 包标签 int / 0-dim tensor
        model_name: abmil | dsmil | ilra

    Returns:
        attn: (N,) float，非负，已按包归一化（和为 1）
    """
    teacher_model.eval()
    bag = bag.to(device)
    n = bag.shape[0]
    batch_idx = torch.zeros(n, dtype=torch.long, device=device)
    out = teacher_model({"x": bag, "batch": batch_idx})
    label_i = int(label.item()) if torch.is_tensor(label) else int(label)

    score = out["score"]
    if model_name == "abmil":
        # (K, N) softmax over instances
        attn = score[0].float().clamp(min=0.0)
    elif model_name == "dsmil":
        # (N, C) softmax over N per column
        attn = score[:, label_i].float().clamp(min=0.0)
    elif model_name == "ilra":
        sc = score.float()
        if sc.dim() == 4:
            attn = sc.mean(dim=(1, 2))
        elif sc.dim() == 3:
            attn = sc.mean(dim=1)
        else:
            attn = sc.flatten()
        while attn.dim() > 1:
            attn = attn.mean(dim=0)
        attn = attn[:n].clamp(min=0.0)
    else:
        attn = score.float().flatten()[:n].clamp(min=0.0)

    s = attn.sum()
    if s > 1e-12:
        attn = attn / s
    else:
        attn = torch.ones(n, device=attn.device, dtype=attn.dtype) / n
    return attn

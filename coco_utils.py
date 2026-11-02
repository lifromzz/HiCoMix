"""
CoCo-Mix utilities: DPP-based two-view sampling + cross-bag alignment helpers.

把和 CoCo-Mix 相关、与训练 step 无关的纯函数集中在这里:
  - 包内 novelty / DPP kernel / 贪心 MAP / 条件 DPP
  - 互补两视图采样 (dpp_two_views, precompute_coco_views)
  - 跨包对齐 (Sinkhorn 软对齐 / NN 硬对齐)

设计意图:
  - 让 mix_strategies.py 只保留"如何把视图和 B 组装成 mixup 输出"的逻辑;
  - 让 data.py 在预处理阶段就能把 view1_idx / view2_idx 算好缓存进 meta,
    训练时直接 O(1) 取用,而不是每个 step 重跑 DPP。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# =====================================================================
# 1. 包内 novelty / DPP 核
# =====================================================================


def bag_novelty_scores(
    sim: torch.Tensor, idx: torch.Tensor, density_topk: int = 8
) -> torch.Tensor:
    """novelty(i) = -mean topK sim(i, ·),越孤立的实例 score 越高。"""
    if idx.numel() <= 1:
        return torch.zeros(idx.numel(), device=sim.device)
    sub = sim[idx][:, idx].float().clone()
    sub.fill_diagonal_(-float("inf"))
    k = min(max(1, int(density_topk)), idx.numel() - 1)
    density = sub.topk(k=k, dim=1).values.mean(dim=1)
    return -density


def build_dpp_kernel(
    sim: torch.Tensor,
    idx: torch.Tensor,
    quality: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """L_ij = q_i · S_ij · q_j,sigma 控制 quality 的尖锐度。"""
    sub = sim[idx][:, idx].float()
    sub = 0.5 * (sub + sub.t())
    q = quality.float().clamp_min(0.0)
    q = (q / q.max().clamp_min(1e-9)).pow(float(sigma))
    L = q.unsqueeze(1) * sub * q.unsqueeze(0)
    L = L + 1e-6 * torch.eye(L.size(0), device=L.device, dtype=L.dtype)
    return L


# =====================================================================
# 2. DPP 采样
# =====================================================================


def dpp_greedy_map(L: torch.Tensor, k: int) -> torch.Tensor:
    """k-DPP 的贪心 MAP。返回选出的 k 个下标。"""
    N = L.size(0)
    k = min(int(k), N)
    if k <= 0 or N == 0:
        return torch.empty(0, dtype=torch.long, device=L.device)

    L = L.float()
    d = L.diag().clone().clamp_min(0.0) + 1e-9
    c = torch.zeros(k, N, device=L.device, dtype=L.dtype)
    selected: list[int] = []
    selected_mask = torch.zeros(N, dtype=torch.bool, device=L.device)

    for it in range(k):
        d_masked = d.clone()
        d_masked[selected_mask] = -float("inf")
        j = int(d_masked.argmax().item())
        if d_masked[j].item() <= 0.0:
            break
        selected.append(j)
        selected_mask[j] = True

        if it == 0:
            ej = L[j].clone()
        else:
            ej = L[j] - (c[:it, j] @ c[:it])
        denom = d[j].clamp_min(1e-12).sqrt()
        c[it] = ej / denom
        d = (d - c[it].pow(2)).clamp_min(0.0)

    return torch.tensor(selected, dtype=torch.long, device=L.device)


def dpp_conditional_kernel(
    L: torch.Tensor, S1_local: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    条件 DPP 核 L^{S1} = L_RR - L_RS · (L_SS)^{-1} · L_SR。
    返回 (L_cond, R_indices)。
    """
    N = L.size(0)
    L = L.float()
    if S1_local.numel() == 0:
        return L.clone(), torch.arange(N, device=L.device)

    mask = torch.ones(N, dtype=torch.bool, device=L.device)
    mask[S1_local.long()] = False
    R = torch.where(mask)[0]
    if R.numel() == 0:
        return torch.empty(0, 0, device=L.device, dtype=L.dtype), R

    S = S1_local.long()
    LSS = L[S][:, S]
    LSS = LSS + 1e-5 * torch.eye(LSS.size(0), device=L.device, dtype=L.dtype)
    LRS = L[R][:, S]
    LRR = L[R][:, R]
    sol = torch.linalg.solve(LSS, LRS.t())
    L_cond = LRR - LRS @ sol
    L_cond = 0.5 * (L_cond + L_cond.t())
    return L_cond, R


# =====================================================================
# 3. 互补两视图采样
# =====================================================================


def cluster_quotas(
    cluster_ids: torch.Tensor, view_size: int, num_clusters: int
) -> torch.Tensor:
    """按 cluster 占比分配 view quota,并保证总和等于 view_size。"""
    counts = torch.bincount(cluster_ids.long(), minlength=num_clusters).long()
    active = counts > 0
    quotas = torch.zeros(num_clusters, dtype=torch.long, device=cluster_ids.device)
    if not bool(active.any()):
        return quotas

    raw = counts.float() / counts.sum().clamp_min(1).float() * int(view_size)
    quotas[active] = torch.floor(raw[active]).long().clamp_min(1)
    quotas = torch.minimum(quotas, counts)

    while int(quotas.sum().item()) > int(view_size):
        candidates = torch.where(quotas > 1)[0]
        if candidates.numel() == 0:
            break
        ci = candidates[quotas[candidates].argmax()]
        quotas[ci] -= 1

    while int(quotas.sum().item()) < int(view_size):
        room = counts - quotas
        candidates = torch.where(room > 0)[0]
        if candidates.numel() == 0:
            break
        frac = raw[candidates] - quotas[candidates].float()
        ci = candidates[frac.argmax()]
        quotas[ci] += 1
    return quotas


def dpp_two_views(
    meta_a: dict,
    view_size: int,
    quality_topk: int = 8,
    quality_sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cluster-aware DPP two-view sampler:
      view1: 每个 cluster 内用 k-DPP greedy MAP 采 q_c 个;
      view2: 在 view1 已选下,用 Conditional k-DPP 在该 cluster 剩余实例上再采 q_c 个;
      不足时按 novelty 顺序补齐。
    """
    cluster = meta_a["cluster"].long()
    device = cluster.device
    sim = meta_a["sim"].to(device)
    num_clusters = int(meta_a["prototype"].size(0))
    n = int(cluster.numel())
    view_size = min(max(1, int(view_size)), n)
    quotas = cluster_quotas(cluster, view_size, num_clusters)

    view1: list[int] = []
    view2: list[int] = []
    leftovers: list[int] = []

    for ci in range(num_clusters):
        idx = torch.where(cluster == ci)[0]
        if idx.numel() == 0:
            continue
        q_c = int(quotas[ci].item())
        if q_c <= 0:
            leftovers.extend(idx.tolist())
            continue

        novelty = bag_novelty_scores(sim, idx, density_topk=quality_topk)
        L = build_dpp_kernel(sim, idx, novelty, sigma=quality_sigma)

        s1_local = dpp_greedy_map(L, q_c)
        chosen1 = idx[s1_local.to(idx.device)]
        view1.extend(chosen1.tolist())

        L_cond, rest_local = dpp_conditional_kernel(L, s1_local.to(L.device))
        if rest_local.numel() > 0 and L_cond.numel() > 0:
            s2_in_rest = dpp_greedy_map(L_cond, q_c)
            s2_local = rest_local[s2_in_rest.to(rest_local.device)]
            chosen2 = idx[s2_local.to(idx.device)]
            view2.extend(chosen2.tolist())
            taken = set(chosen1.tolist()) | set(chosen2.tolist())
        else:
            taken = set(chosen1.tolist())

        order = novelty.argsort(descending=True)
        for j in idx[order].tolist():
            if j not in taken:
                leftovers.append(j)

    used1 = set(view1)
    for j in leftovers:
        if len(view1) >= view_size:
            break
        if j not in used1:
            view1.append(j)
            used1.add(j)

    used2 = set(view1) | set(view2)
    for j in leftovers:
        if len(view2) >= view_size:
            break
        if j not in used2:
            view2.append(j)
            used2.add(j)

    return (
        torch.tensor(view1[:view_size], dtype=torch.long, device=device),
        torch.tensor(view2[:view_size], dtype=torch.long, device=device),
    )


def precompute_coco_views(
    meta_a: dict,
    view_ratio: float = 0.5,
    quality_topk: int = 8,
    quality_sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dataset 端调用:从 meta_a 计算并返回 (view1_idx, view2_idx),供训练时直接取用。
    view_ratio 决定每个视图的实例数 = round(N * view_ratio)。
    """
    n = int(meta_a["cluster"].numel())
    view_size = max(1, int(round(n * float(view_ratio))))
    return dpp_two_views(
        meta_a,
        view_size=view_size,
        quality_topk=int(quality_topk),
        quality_sigma=float(quality_sigma),
    )


# =====================================================================
# 4. 跨包对齐 (训练阶段每步调用,bag_b 实时变化)
# =====================================================================


def ot_align_view_to_bag(
    view_feats: torch.Tensor,
    bag_b: torch.Tensor,
    eps: float = 0.1,
    n_iters: int = 50,
    metric: str = "cosine",
) -> torch.Tensor:
    """Sinkhorn 跨包对齐:对 view 每个位置 i,从 B 中合成软 donor。"""
    M = view_feats.size(0)
    Nb = bag_b.size(0)
    device = view_feats.device
    dtype = view_feats.dtype

    if M == 0 or Nb == 0:
        return view_feats.clone()

    if metric == "cosine":
        va = F.normalize(view_feats.float(), dim=-1)
        vb = F.normalize(bag_b.float(), dim=-1)
        C = 1.0 - va @ vb.t()
    else:
        C = torch.cdist(view_feats.float(), bag_b.float(), p=2).pow(2)
        C = C / C.max().clamp_min(1e-9)

    eps = max(float(eps), 1e-3)
    K = torch.exp(-C / eps)
    mu = torch.full((M,), 1.0 / M, device=device, dtype=K.dtype)
    nu = torch.full((Nb,), 1.0 / Nb, device=device, dtype=K.dtype)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(int(max(1, n_iters))):
        u = mu / (K @ v).clamp_min(1e-12)
        v = nu / (K.t() @ u).clamp_min(1e-12)
    T = u.unsqueeze(1) * K * v.unsqueeze(0)

    row_mass = T.sum(dim=1, keepdim=True).clamp_min(1e-12)
    soft_b = (T / row_mass) @ bag_b.float()
    return soft_b.to(dtype)


def nn_align_view_to_bag(
    view_feats: torch.Tensor,
    bag_b: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """最近邻硬对齐:为 view 每个位置取 bag_b 中最相似的实例作为 donor。"""
    if view_feats.size(0) == 0 or bag_b.size(0) == 0:
        return view_feats.clone()
    if metric == "cosine":
        va = F.normalize(view_feats.float(), dim=-1)
        vb = F.normalize(bag_b.float(), dim=-1)
        sim = va @ vb.t()
        idx = sim.argmax(dim=1)
    else:
        d = torch.cdist(view_feats.float(), bag_b.float(), p=2)
        idx = d.argmin(dim=1)
    return bag_b[idx]

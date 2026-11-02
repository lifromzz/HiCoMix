from __future__ import annotations
"""Mixup strategy helpers for HiCoMix."""

import random

import torch

from coco_utils import (
    dpp_two_views,
    nn_align_view_to_bag,
    ot_align_view_to_bag,
)


@torch.no_grad()
def _kmeans_cluster(bag: torch.Tensor, k: int, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lightweight k-means on a single bag, returns (assign, prototypes)."""
    n = bag.size(0)
    k = max(1, min(int(k), n))
    x = bag.float()

    init_idx = torch.randperm(n, device=bag.device)[:k]
    centers = x[init_idx]
    assign = torch.zeros(n, dtype=torch.long, device=bag.device)

    for _ in range(max(1, int(iters))):
        dist = torch.cdist(x, centers, p=2).pow(2)
        assign = dist.argmin(dim=1)
        new_centers = []
        for ci in range(k):
            mask = assign == ci
            if mask.any():
                new_centers.append(x[mask].mean(dim=0))
            else:
                far_idx = dist.min(dim=1).values.argmax()
                new_centers.append(x[far_idx])
        centers = torch.stack(new_centers, dim=0)
    return assign, centers.to(bag.dtype)




def _match_prototype_pairs(
    dist_proto: torch.Tensor,
    mode: str,
) -> list[tuple[int, int]]:
    """
    一对一原型配对。
    nearest: 贪心选距离最小的 (ia, ib)
    farthest: 贪心选距离最大的 (ia, ib)
    random: 随机打乱后配对，尺寸不等时用随机 fallback
    """
    mode = str(mode).strip().lower()
    ka, kb = dist_proto.size(0), dist_proto.size(1)
    if ka == 0 or kb == 0:
        return []

    if mode == "random":
        ia_list = list(range(ka))
        ib_list = list(range(kb))
        random.shuffle(ia_list)
        random.shuffle(ib_list)
        pairs: list[tuple[int, int]] = []
        n_match = min(ka, kb)
        for i in range(n_match):
            pairs.append((ia_list[i], ib_list[i]))
        for ia in ia_list[n_match:]:
            pairs.append((ia, random.randrange(kb)))
        for ib in ib_list[n_match:]:
            pairs.append((random.randrange(ka), ib))
        return pairs

    ascending = mode == "nearest"
    flat_order = dist_proto.reshape(-1).argsort()
    if not ascending:
        flat_order = flat_order.flip(0)

    used_a, used_b, pairs = set(), set(), []
    for idx in flat_order.tolist():
        ia = idx // kb
        ib = idx % kb
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        pairs.append((ia, ib))
        if len(pairs) >= min(ka, kb):
            break

    pick_b = dist_proto.argmax if mode == "farthest" else dist_proto.argmin
    pick_a = dist_proto.argmax if mode == "farthest" else dist_proto.argmin
    for ia in range(ka):
        if ia in used_a:
            continue
        pairs.append((ia, pick_b(dim=1)[ia].item()))
    for ib in range(kb):
        if ib in used_b:
            continue
        pairs.append((pick_a(dim=0)[ib].item(), ib))
    return pairs


@torch.no_grad()
def _mix_matched_cluster_instances(
    a_feat: torch.Tensor,
    b_feat: torch.Tensor,
    lam: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """簇内实例混合：proto=最近邻；random=随机对齐；constant=按前 min_n 索引对齐。"""
    mode = str(mode).strip().lower()


    min_n = min(a_feat.size(0), b_feat.size(0))
    if min_n == 0:
        return a_feat, b_feat

    if mode == "instance_level_random":
        ra = torch.randperm(a_feat.size(0), device=a_feat.device)[:min_n]
        rb = torch.randperm(b_feat.size(0), device=b_feat.device)[:min_n]
        mixed_a = a_feat.clone()
        mixed_b = b_feat.clone()
        mixed_a[ra] = lam * a_feat[ra] + (1.0 - lam) * b_feat[rb]
        mixed_b[rb] = lam * b_feat[rb] + (1.0 - lam) * a_feat[ra]
        return mixed_a, mixed_b

    if mode == "instance_level_constant":
        mixed_a = a_feat.clone()
        mixed_b = b_feat.clone()
        mixed_a[:min_n] = lam * a_feat[:min_n] + (1.0 - lam) * b_feat[:min_n]
        mixed_b[:min_n] = lam * b_feat[:min_n] + (1.0 - lam) * a_feat[:min_n]
        return mixed_a, mixed_b
    
    d = torch.cdist(a_feat.float(), b_feat.float(), p=2)
    if mode == "instance_level_max":
        nn_b_for_a = d.argmax(dim=1)
        nn_a_for_b = d.argmax(dim=0)
    else:
        nn_b_for_a = d.argmin(dim=1)
        nn_a_for_b = d.argmin(dim=0)

    aligned_b = b_feat[nn_b_for_a]
    mixed_a = lam * a_feat + (1.0 - lam) * aligned_b
    aligned_a = a_feat[nn_a_for_b]
    mixed_b = lam * b_feat + (1.0 - lam) * aligned_a
    return mixed_a, mixed_b


@torch.no_grad()
def proto_bidirectional_mixup(
    bag_a: torch.Tensor,
    bag_b: torch.Tensor,
    lam: float,
    meta_a: dict | None = None,
    meta_b: dict | None = None,
    config=None,
) -> dict:
    """
    Cluster A/B into k classes, pair prototypes (nearest/farthest/random), then
    mix instances inside matched clusters to produce dual augmented bags.
    """
    if bag_a.dim() != 2 or bag_b.dim() != 2:
        raise ValueError("proto_bidirectional_mixup expects single bags shaped (N, D)")

    assign_a = meta_a["assign"].to(bag_a.device).long()
    proto_a = meta_a["prototype"].to(bag_a.device, dtype=bag_a.dtype)
    assign_b = meta_b["assign"].to(bag_b.device).long()
    proto_b = meta_b["prototype"].to(bag_b.device, dtype=bag_b.dtype)

    dist_proto = torch.cdist(proto_a.float(), proto_b.float(), p=2)
    pairs = _match_prototype_pairs(dist_proto, config.proto_pair_mode)

    mixed_a = bag_a.clone()
    mixed_b = bag_b.clone()
    for ia, ib in pairs:
        idx_a = torch.where(assign_a == ia)[0]
        idx_b = torch.where(assign_b == ib)[0]
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            continue

        a_feat = bag_a[idx_a]
        b_feat = bag_b[idx_b]
        new_a, new_b = _mix_matched_cluster_instances(a_feat, b_feat, lam, config.mix_type)
        mixed_a[idx_a] = new_a
        mixed_b[idx_b] = new_b

    return {
        "kind": "dual",
        "bag_a": mixed_a,
        "bag_b": mixed_b,
        "w_a": float(lam),
        "w_b": float(1.0 - lam),
        "w_a_b": float(1.0 - lam),
        "w_b_b": float(lam),
        "instance_mix_mode": config.mix_type,
        "proto_pair_mode": config.proto_pair_mode,
    }


@torch.no_grad()
def proto_single_mixup(
    bag_a: torch.Tensor,
    bag_b: torch.Tensor,
    lam: float,
    meta_a: dict | None = None,
    meta_b: dict | None = None,
    config=None,
    instance_mix_type: str | None = None,
) -> dict:
    """Single-branch proto mixup: update only bag_a with nearest instances from bag_b."""
    out = proto_bidirectional_mixup(
        bag_a=bag_a,
        bag_b=bag_b,
        lam=lam,
        meta_a=meta_a,
        meta_b=meta_b,
        config=config,
    )
    return {"bag": out["bag_a"], "w_a": out["w_a"], "w_b": out["w_b"]}


def nn_mixup(bag_a: torch.Tensor, bag_b: torch.Tensor, lam: float) -> dict:
    """NN strategy single-branch output with label weights."""
    with torch.no_grad():
        c = torch.cdist(bag_a, bag_b, p=2).pow(2)
        best_for_a = c.argmin(dim=1)
        aligned_b_for_a = bag_b[best_for_a]
    mixed_bag_a = lam * bag_a + (1.0 - lam) * aligned_b_for_a
    return {"bag": mixed_bag_a, "w_a": float(lam), "w_b": float(1.0 - lam)}


def nn_single_mixup(bag_a: torch.Tensor, bag_b: torch.Tensor, lam: float) -> dict:
    """Explicit alias for nn single-branch mixup."""
    return nn_mixup(bag_a, bag_b, lam)


def cat_mixup(bag_a: torch.Tensor, bag_b: torch.Tensor, lam: float) -> dict:
    """CAT strategy output with size-based label weights."""
    aug_bag_cat = torch.cat([bag_a, bag_b], dim=0)
    n_a = float(bag_a.size(0))
    n_b = float(bag_b.size(0))
    denom = max(1.0, n_a + n_b)
    return {"bag": aug_bag_cat, "w_a": n_a / denom, "w_b": n_b / denom}


def random_mixup(bag_a: torch.Tensor, bag_b: torch.Tensor, lam: float) -> dict:
    """Randomly shuffle instances in both bags, then interpolate."""
    idx_a = torch.randperm(bag_a.size(0), device=bag_a.device)
    idx_b = torch.randperm(bag_b.size(0), device=bag_b.device)
    min_n = min(bag_a.size(0), bag_b.size(0))
    mixed = lam * bag_a[idx_a[:min_n]] + (1.0 - lam) * bag_b[idx_b[:min_n]]
    return {"bag": mixed, "w_a": float(lam), "w_b": float(1.0 - lam)}


def constant_mixup(bag_a: torch.Tensor, bag_b: torch.Tensor, lam: float) -> dict:
    """Interpolate aligned-by-index instances without shuffling."""
    min_n = min(bag_a.size(0), bag_b.size(0))
    mixed = lam * bag_a[:min_n] + (1.0 - lam) * bag_b[:min_n]
    return {"bag": mixed, "w_a": float(lam), "w_b": float(1.0 - lam)}


def _resolve_two_views(
    meta_a: dict,
    bag_a: torch.Tensor,
    view_ratio: float,
    quality_topk: int,
    quality_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """优先使用 Dataset 预计算的 view 索引;缺失则现算(向后兼容)。"""
    cached_v1 = meta_a.get("view1_idx") if isinstance(meta_a, dict) else None
    cached_v2 = meta_a.get("view2_idx") if isinstance(meta_a, dict) else None
    if cached_v1 is not None and cached_v2 is not None:
        return cached_v1.to(bag_a.device).long(), cached_v2.to(bag_a.device).long()

    n_a = bag_a.size(0)
    view_size = max(1, int(round(n_a * float(view_ratio))))
    v1, v2 = dpp_two_views(
        meta_a,
        view_size=view_size,
        quality_topk=int(quality_topk),
        quality_sigma=float(quality_sigma),
    )
    return v1.to(bag_a.device), v2.to(bag_a.device)


def coco_mixup_two_views(
    bag_a: torch.Tensor,
    bag_b: torch.Tensor,
    meta_a: dict,
    meta_b: dict | None = None,  # 保留参数兼容旧 API,本实现不强制使用
    view_ratio: float = 0.5,
    lam: float = 0.5,
    cross_mode: str = "ot",
    ot_eps: float = 0.1,
    ot_iters: int = 50,
    alpha_max: float = 0.5,
    metric: str = "cosine",
    quality_sigma: float = 1.0,
    quality_topk: int = 8,
    second_view_prob: float = 1.0,
    donor_ratio: float | None = None,  # deprecated, 仅保留向后兼容
) -> dict:
    """
    CoCo-Mix v2: DPP 互补视图 + (OT | NN) 跨包混合。

    流程:
      1) view1 / view2 索引优先取自 meta_a 中预计算的 view1_idx/view2_idx,
         否则 fallback 到运行时 DPP;
      2) 对每个 view 单独做 cross-bag mixup:
         - cross_mode=ot:  Sinkhorn 软对齐,alpha = min(1-lam, alpha_max)
         - cross_mode=nn:  最近邻硬对齐,沿用 lam 混合
         - cross_mode=none: 不混 B,只做 DPP 两视图(消融用);
      3) mixed1 必定返回,mixed2 以 second_view_prob 的概率返回;
      4) 标签权重 w_a / w_b 以 list 返回(per-view),engine 端按列表消费。

    meta_a 至少需要: cluster (N,), prototype (K,D), 以及 sim (N,N)
    如已包含 view1_idx / view2_idx 则直接复用,跳过运行时 DPP。
    """
    if bag_a.dim() != 2 or bag_b.dim() != 2:
        raise ValueError("coco_mixup_two_views 只支持单包输入 (N,D)")
    if metric not in ("cosine", "l2"):
        raise ValueError(f"未知 coco metric={metric}")
    cross_mode = str(cross_mode).strip().lower()
    if cross_mode not in ("ot", "nn", "none"):
        raise ValueError(f"未知 cross_mode={cross_mode!r},应为 ot|nn|none")
    second_view_prob = min(max(float(second_view_prob), 0.0), 1.0)

    view1_idx, view2_idx = _resolve_two_views(
        meta_a, bag_a,
        view_ratio=view_ratio,
        quality_topk=quality_topk,
        quality_sigma=quality_sigma,
    )

    view1 = bag_a[view1_idx]
    view2 = bag_a[view2_idx]

    lam = min(max(float(lam), 0.0), 1.0)
    alpha = 1.0 - lam
    include_view2 = bool(
        second_view_prob >= 1.0
        or (
            second_view_prob > 0.0
            and torch.rand((), device=bag_a.device).item() < second_view_prob
        )
    )

    if cross_mode == "none" or alpha <= 1e-6:
        mixed1 = view1
        mixed2 = view2 if include_view2 else None
        w_a, w_b = 1.0, 0.0
    elif cross_mode == "ot":
        alpha_eff = min(alpha, float(alpha_max))
        soft_b1 = ot_align_view_to_bag(view1, bag_b, eps=ot_eps, n_iters=ot_iters, metric=metric)
        mixed1 = (1.0 - alpha_eff) * view1 + alpha_eff * soft_b1
        if include_view2:
            soft_b2 = ot_align_view_to_bag(view2, bag_b, eps=ot_eps, n_iters=ot_iters, metric=metric)
            mixed2 = (1.0 - alpha_eff) * view2 + alpha_eff * soft_b2
        else:
            mixed2 = None
        w_a, w_b = 1.0 - alpha_eff, alpha_eff
    else:  # nn
        donor1 = nn_align_view_to_bag(view1, bag_b, metric=metric)
        mixed1 = lam * view1 + (1.0 - lam) * donor1
        if include_view2:
            donor2 = nn_align_view_to_bag(view2, bag_b, metric=metric)
            mixed2 = lam * view2 + (1.0 - lam) * donor2
        else:
            mixed2 = None
        w_a, w_b = lam, 1.0 - lam

    views = [mixed1]
    view_indices = [view1_idx]
    if mixed2 is not None:
        views.append(mixed2)
        view_indices.append(view2_idx)

    return {
        "kind": "coco",
        "views": views,
        "view_indices": view_indices,
        "w_a": [w_a] * len(views),
        "w_b": [w_b] * len(views),
        "cross_mode": cross_mode,
        "second_view_prob": second_view_prob,
        "include_view2": include_view2,
    }

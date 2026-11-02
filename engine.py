# engine.py
"""
IG-OTMix 训练引擎。

核心流程（每个 step）：
  1. 从 DataLoader 取出包 A (bag_a, label_a, slide_name_a)
  2. 通过数据集策略采样包 B：train_loader.dataset.sample_bag_b(slide_name_a)
  3. mix_align_mode=nn：欧氏最近邻对齐实例后做 NN-Mixup
  4. 按 mix_type / mix_cat_label_mode / mix_include_original_bag 计算损失并反传

其它 mix_align_mode（cross_attn / fgw / cluster_nn）暂未接入可学习对齐模块时，
训练阶段会回退为与 nn 相同的硬 NN 对齐，并在首次打印提示。
"""

from __future__ import annotations

import json
import os
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from colorama import Fore, init
from tqdm import tqdm

from eval import evaluate
from loss_fn import SoftTargetCrossEntropy
from mix_strategies import (
    cat_mixup,
    constant_mixup,
    nn_mixup,
    nn_single_mixup,
    proto_bidirectional_mixup,
    proto_single_mixup,
    random_mixup,
)

init(autoreset=True)

print("========> HiCoMix (Prototype Mixup + OT Bag Matching) <============")


# =====================================================================
# 学生 MIL 前向（与 eval._mil_forward_probs 一致）
# =====================================================================


def _student_logits(mil_model, bag: torch.Tensor) -> torch.Tensor:
    """单包 (N, D) -> (1, C) logits（训练态下各模型 probs 键为未 softmax 的 logits）。"""
    if bag.dim() != 2:
        raise ValueError(f"期望 bag 为 (N, D)，当前 {tuple(bag.shape)}")
    n, _ = bag.shape
    batch_idx = torch.zeros(n, dtype=torch.long, device=bag.device)
    out = mil_model({"x": bag, "batch": batch_idx})
    logits = out["probs"]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return logits


# =====================================================================
# 1. Mixup 构图
# =====================================================================


def perform_nn_mixup(
    bag_a: torch.Tensor,
    bag_b: torch.Tensor,
    lam: float,
    mix_type: str,
    config=None,
    sample_meta: dict = None,
):
    """
    mix_type: 见 config --mix_type；含 single / dual / triple / quad 等。
    dual 可含 w_a_b/w_b_b（第二支路软标签与第一支不对称时使用）。
    """
    mix_type = str(mix_type).strip().lower()

    if mix_type == "random":
        out = random_mixup(bag_a, bag_b, lam)
        return {"kind": "single", **out}
    elif mix_type == "constant":
        out = constant_mixup(bag_a, bag_b, lam)
        return {"kind": "single", **out}
    elif mix_type == "cat":
        out = cat_mixup(bag_a, bag_b, lam)
        return {"kind": "single", **out}
    elif mix_type == "nn_baga":
        out = nn_single_mixup(bag_a, bag_b, lam)
        return {"kind": "single", **out}
    elif mix_type == "nn_bagb":
        out = nn_single_mixup(bag_b, bag_a, lam)
        return {"kind": "single", "bag": out["bag"], "w_a": out["w_b"], "w_b": out["w_a"]}
    elif mix_type == "nn_baga+nn_bagb":
        out_a = nn_mixup(bag_a, bag_b, lam)
        out_b = nn_mixup(bag_b, bag_a, lam)
        return {
            "kind": "dual",
            "bag_a": out_a["bag"],
            "bag_b": out_b["bag"],
            "w_a": out_a["w_a"],
            "w_b": out_a["w_b"],
            "w_a_b": out_b["w_b"],
            "w_b_b": out_b["w_a"],
        }
    elif mix_type in (
        "instance_level_random",
        "instance_level_constant",
        "instance_level_max",
    ):
        meta = (sample_meta or {}).get("proto", {})
        return proto_bidirectional_mixup(
            bag_a,
            bag_b,
            lam=lam,
            meta_a=meta.get("meta_a"),
            meta_b=meta.get("meta_b"),
            config=config,
        )
    elif mix_type == "proto_baga":
        meta = (sample_meta or {}).get("proto", {})
        out = proto_single_mixup(
            bag_a,
            bag_b,
            lam=lam,
            meta_a=meta.get("meta_a"),
            meta_b=meta.get("meta_b"),
            config=config,
        )
        return {"kind": "single", **out}
    elif mix_type == "proto_bagb":
        meta = (sample_meta or {}).get("proto", {})
        out = proto_single_mixup(
            bag_b,
            bag_a,
            lam=lam,
            meta_a=meta.get("meta_b"),
            meta_b=meta.get("meta_a"),
            config=config,
        )
        return {"kind": "single", "bag": out["bag"], "w_a": out["w_b"], "w_b": out["w_a"]}
    elif mix_type == "proto_baga+proto_bagb":
        meta = (sample_meta or {}).get("proto", {})
        proto_bagb_prob = min(
            max(float(getattr(config, "proto_bagb_prob", 1.0)), 0.0), 1.0
        )
        include_proto_bagb = bool(
            proto_bagb_prob >= 1.0
            or (
                proto_bagb_prob > 0.0
                and torch.rand((), device=bag_a.device).item() < proto_bagb_prob
            )
        )
        if include_proto_bagb:
            out = proto_bidirectional_mixup(
                bag_a,
                bag_b,
                lam=lam,
                meta_a=meta.get("meta_a"),
                meta_b=meta.get("meta_b"),
                config=config,
            )
            out["proto_bagb_prob"] = proto_bagb_prob
            out["include_proto_bagb"] = True
            return out

        out = proto_single_mixup(
            bag_a,
            bag_b,
            lam=lam,
            meta_a=meta.get("meta_a"),
            meta_b=meta.get("meta_b"),
            config=config,
        )
        return {
            "kind": "single",
            **out,
            "proto_bagb_prob": proto_bagb_prob,
            "include_proto_bagb": False,
        }
    elif mix_type == "proto_baga+nn_baga":
        # 两支均为「以 A 为主」的混合：proto 簇内 NN + 全局 NN（非 proto 双向）
        meta = (sample_meta or {}).get("proto", {})
        if meta.get("meta_a") is None or meta.get("meta_b") is None:
            raise ValueError(
                "mix_type=proto_baga+nn_baga 需要 sample_meta['proto'] 中含 meta_a / meta_b。"
            )
        p = proto_single_mixup(
            bag_a,
            bag_b,
            lam=lam,
            meta_a=meta.get("meta_a"),
            meta_b=meta.get("meta_b"),
            config=config,
        )
        n = nn_mixup(bag_a, bag_b, lam)
        return {
            "kind": "dual",
            "bag_a": p["bag"],
            "bag_b": n["bag"],
            "w_a": float(p["w_a"]),
            "w_b": float(p["w_b"]),
        }
    elif mix_type == "proto_baga+nn_bagb":
        meta = (sample_meta or {}).get("proto", {})
        if meta.get("meta_a") is None or meta.get("meta_b") is None:
            raise ValueError(
                "mix_type=proto_baga+nn_bagb 需要 sample_meta['proto'] 中含 meta_a / meta_b。"
            )
        p = proto_single_mixup(
            bag_a,
            bag_b,
            lam=lam,
            meta_a=meta.get("meta_a"),
            meta_b=meta.get("meta_b"),
            config=config,
        )
        n = nn_mixup(bag_b, bag_a, lam)
        # 第二支为 nn_bagb：首参为 B，软标签对 (label_a,label_b) 与第一支不对称，见下方 w_a_b/w_b_b
        return {
            "kind": "dual",
            "bag_a": p["bag"],
            "bag_b": n["bag"],
            "w_a": float(p["w_a"]),
            "w_b": float(p["w_b"]),
            "w_a_b": float(n["w_b"]),
            "w_b_b": float(n["w_a"]),
        }
    elif mix_type == "proto_baga+proto_bagb+nn_baga":
        meta = (sample_meta or {}).get("proto", {})
        if meta.get("meta_a") is None or meta.get("meta_b") is None:
            raise ValueError(
                "mix_type=proto_baga+proto_bagb+nn_baga 需要 sample_meta['proto'] 中含 meta_a / meta_b。"
            )
        proto_dual = proto_bidirectional_mixup(
            bag_a,
            bag_b,
            lam=lam,
            meta_a=meta.get("meta_a"),
            meta_b=meta.get("meta_b"),
            config=config,
        )
        nn_c = nn_mixup(bag_a, bag_b, lam)
        return {
            "kind": "triple",
            "bag_a": proto_dual["bag_a"],
            "bag_b": proto_dual["bag_b"],
            "bag_c": nn_c["bag"],
            "w_a": [
                float(proto_dual["w_a"]),
                float(proto_dual["w_b"]),
                float(nn_c["w_a"]),
            ],
            "w_b": [
                float(proto_dual["w_b"]),
                float(proto_dual["w_a"]),
                float(nn_c["w_b"]),
            ],
        }
    elif mix_type == "proto_baga+proto_bagb+nn_bagb":
        meta = (sample_meta or {}).get("proto", {})
        if meta.get("meta_a") is None or meta.get("meta_b") is None:
            raise ValueError(
                "mix_type=proto_baga+proto_bagb+nn_bagb 需要 sample_meta['proto'] 中含 meta_a / meta_b。"
            )
        proto_dual = proto_bidirectional_mixup(
            bag_a,
            bag_b,
            lam=lam,
            meta_a=meta.get("meta_a"),
            meta_b=meta.get("meta_b"),
            config=config,
        )
        nn_d = nn_mixup(bag_b, bag_a, lam)
        return {
            "kind": "triple",
            "bag_a": proto_dual["bag_a"],
            "bag_b": proto_dual["bag_b"],
            "bag_c": nn_d["bag"],
            "w_a": [
                float(proto_dual["w_a"]),
                float(proto_dual["w_b"]),
                float(nn_d["w_b"]),
            ],
            "w_b": [
                float(proto_dual["w_b"]),
                float(proto_dual["w_a"]),
                float(nn_d["w_a"]),
            ],
        }

    elif mix_type == "proto_baga+proto_bagb+nn_baga+nn_bagb":
        meta = (sample_meta or {}).get("proto", {})
        if meta.get("meta_a") is None or meta.get("meta_b") is None:
            raise ValueError(
                "mix_type=proto_baga+proto_bagb+nn_baga+nn_bagb 需要 sample_meta['proto'] 中含 "
                "meta_a / meta_b（与 proto_baga+proto_bagb 一致）。"
            )
        proto_dual = proto_bidirectional_mixup(
            bag_a,
            bag_b,
            lam=lam,
            meta_a=meta.get("meta_a"),
            meta_b=meta.get("meta_b"),
            config=config,
        )
        nn_c = nn_mixup(bag_a, bag_b, lam)
        nn_d = nn_mixup(bag_b, bag_a, lam)
        # 与原先两次 proto_single + 两次 nn_single 等价：proto 双向一次；NN 与 nn_baga+nn_bagb 同构
        w_a_list = [
            float(proto_dual["w_a"]),
            float(proto_dual["w_b"]),
            float(nn_c["w_a"]),
            float(nn_d["w_b"]),
        ]
        w_b_list = [
            float(proto_dual["w_b"]),
            float(proto_dual["w_a"]),
            float(nn_c["w_b"]),
            float(nn_d["w_a"]),
        ]
        return {
            "kind": "quad",
            "bag_a": proto_dual["bag_a"],
            "bag_b": proto_dual["bag_b"],
            "bag_c": nn_c["bag"],
            "bag_d": nn_d["bag"],
            "w_a": w_a_list,
            "w_b": w_b_list,
        }
    else:
        raise ValueError(f"未知 mix_type={mix_type!r}")


def _sample_bag_b_for_mix(train_loader, slide_name_a: str, config):
    """统一采样 B，并返回 Dataset 提供的可选 mix 元信息。"""
    slide_name_b, bag_b, label_b, sample_meta = train_loader.dataset.sample_bag_b(
        str(slide_name_a)
    )
    if sample_meta is None:
        sample_meta = {}
    return slide_name_b, bag_b, label_b, sample_meta




def load_student_checkpoint(ckp_path, mil_model, align_module=None, device="cuda"):
    """加载 student（及对 align_module 的同文件 state_dict 子集，若有）。"""
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckp_path}")
    payload = torch.load(ckp_path, map_location=device)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state, dict):
        raise TypeError("checkpoint 中无有效 state_dict")

    mil_model.load_state_dict(state, strict=True)
    if align_module is not None:
        pref = "align_module."
        sub = {k[len(pref) :]: v for k, v in state.items() if k.startswith(pref)}
        if sub:
            align_module.load_state_dict(sub, strict=False)
    mil_model.to(device)
    if align_module is not None:
        align_module.to(device)

# =====================================================================
# 3. 主训练循环
# =====================================================================


def train_ig_ot_mixup(mil_model, train_loader, val_loader, config):
    """
    IG-OTMix 训练主循环。

    Args:
        mil_model: MIL 模型（student），forward({"x","batch"})
        train_loader: IGOTMILDataset
        val_loader: SimpleDataset
        config: argparse.Namespace
    """

    beta_a = float(getattr(config, "beta_alpha", 1.0))
    print(f"--- 开始训练, Beta({beta_a}, {beta_a}) ---")

    optimizer = optim.Adam(
        mil_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = SoftTargetCrossEntropy()

    log_path = os.path.join(config.log_dir, f"train_log_seed{config.seed}.jsonl")
    os.makedirs(config.log_dir, exist_ok=True)

    best_val_score = 0.0

    for epoch in range(config.epochs):
        mil_model.train()
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"{config.model_name} {config.dataname} fold{config.fold} "
            f"Epoch {epoch + 1}/{config.epochs}",
        )

        for batch in pbar:
            bags_a, label_a, slide_name_a = batch

            if isinstance(slide_name_a, (list, tuple)):
                slide_name_a = slide_name_a[0]
            if isinstance(label_a, torch.Tensor) and label_a.dim() > 0:
                label_a = label_a.reshape(-1)[0]

            bags_a = bags_a.to(config.device)
            if bags_a.dim() == 3 and bags_a.size(0) == 1:
                bags_a = bags_a.squeeze(0)
            elif bags_a.dim() == 3:
                raise NotImplementedError("当前 engine 仅支持 batch_size=1 的 MIL 训练。")

            label_a = label_a.to(config.device)
            if label_a.dim() == 0:
                label_a = label_a.unsqueeze(0)

            slide_name_b, bags_b, label_b, sample_meta = _sample_bag_b_for_mix(train_loader, str(slide_name_a), config)
            bags_b = bags_b.to(config.device)
            label_b = torch.tensor(label_b, dtype=torch.long, device=config.device).view(1)

            lam = torch.distributions.Beta(beta_a, beta_a).sample().to(config.device)
            lam_val = float(lam.item())

            mix_out = perform_nn_mixup(
                bags_a,
                bags_b,
                lam=lam_val,
                mix_type=config.mix_type,
                config=config,
                sample_meta=sample_meta,
            )

            optimizer.zero_grad()

            if mix_out["kind"] == "single":
                logits = _student_logits(mil_model, mix_out["bag"])
                loss = (
                    criterion(logits.view(1, -1), label_a.long().view(-1), mix_out["w_a"])
                    + criterion(logits.view(1, -1), label_b.long().view(-1), mix_out["w_b"])
                )
            elif mix_out["kind"] == "quad":
                quad_keys = ("bag_a", "bag_b", "bag_c", "bag_d")
                logits_list = [
                    _student_logits(mil_model, mix_out[k]) for k in quad_keys
                ]
                w_a_list = mix_out["w_a"]
                w_b_list = mix_out["w_b"]
                loss = sum(
                    criterion(lg.view(1, -1), label_a.long().view(-1), float(wa))
                    + criterion(lg.view(1, -1), label_b.long().view(-1), float(wb))
                    for lg, wa, wb in zip(logits_list, w_a_list, w_b_list)
                ) / float(len(quad_keys))
            elif mix_out["kind"] == "triple":
                triple_keys = ("bag_a", "bag_b", "bag_c")
                logits_list = [
                    _student_logits(mil_model, mix_out[k]) for k in triple_keys
                ]
                w_a_list = mix_out["w_a"]
                w_b_list = mix_out["w_b"]
                loss = sum(
                    criterion(lg.view(1, -1), label_a.long().view(-1), float(wa))
                    + criterion(lg.view(1, -1), label_b.long().view(-1), float(wb))
                    for lg, wa, wb in zip(logits_list, w_a_list, w_b_list)
                ) / float(len(triple_keys))
            else:
                logits_a = _student_logits(mil_model, mix_out["bag_a"])
                logits_b = _student_logits(mil_model, mix_out["bag_b"])
                w_a = float(mix_out.get("w_a", lam_val))
                w_b = float(mix_out.get("w_b", 1.0 - lam_val))
                w_a_b = float(mix_out.get("w_a_b", w_a))
                w_b_b = float(mix_out.get("w_b_b", w_b))
                la = (
                    criterion(logits_a.view(1, -1), label_a.long().view(-1), w_a)
                    + criterion(logits_a.view(1, -1), label_b.long().view(-1), w_b)
                )
                lb = (
                    criterion(logits_b.view(1, -1), label_a.long().view(-1), w_a_b)
                    + criterion(logits_b.view(1, -1), label_b.long().view(-1), w_b_b)
                )
                loss = 0.5 * (la + lb)

            if bool(getattr(config, "mix_include_original_bag", False)):
                logits_orig = _student_logits(mil_model, bags_a)
                loss = loss + config.mix_aux_weight * criterion(logits_orig, label_a.long().view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            pbar.set_postfix(loss=total_loss / n_steps)

        val_auc, val_acc = evaluate(mil_model, val_loader, config)
        print(
            Fore.YELLOW
            + f"Train Loss: {total_loss / max(1, len(train_loader)):.4f} | "
            f"Val ACC: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
        )

        curr = (val_auc + val_acc) / 2
        if curr > best_val_score:
            best_val_score = curr
            torch.save(mil_model.state_dict(), config.val_best_ckp)
            print(f"{Fore.GREEN}保存最优模型，Score: {best_val_score:.4f}{Fore.RESET}")

        log_dict = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(1, len(train_loader)),
            "val_auc": val_auc,
            "val_acc": val_acc,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")

    torch.save(mil_model.state_dict(), config.train_last_ckp)
    print(f"最后模型保存至 {config.train_last_ckp}")

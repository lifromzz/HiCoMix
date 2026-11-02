"""Evaluate saved HiCoMix checkpoints on train / val / test splits.

Existing ``val_{dataset}_clean.csv`` is the val-best checkpoint on the test split
and is left unchanged. This script adds:

  train_last_{dataset}_clean.csv
  train_best_{dataset}_clean.csv
  val_last_{dataset}_clean.csv
  val_best_{dataset}_clean.csv
  test_last_{dataset}_clean.csv

Paths can be overridden with environment variables:
  HICOMIX_DATASET_ROOT, HICOMIX_EXCEL_ROOT, HICOMIX_RESULT_ROOT
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score, MatthewsCorrCoef, Precision, Recall

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.environ.get(
    "HICOMIX_RESULT_ROOT", os.path.join(ROOT, "train", "hicomix")
)
EXCEL_ROOT = os.environ.get("HICOMIX_EXCEL_ROOT", os.path.join(ROOT, "splits"))
CACHE_ROOT = os.path.join(RESULT_ROOT, "_split_eval_cache")
_DATASET_ROOT = os.environ.get("HICOMIX_DATASET_ROOT", os.path.join(ROOT, "dataset"))

DATA_ROOTS = {
    "TCGA-NSCLC": [os.path.join(_DATASET_ROOT, "TCGA-NSCLC", "uni_level1_256")],
    "TCGA-RCC": [os.path.join(_DATASET_ROOT, "TCGA-RCC", "uni_level1_256")],
    "CPTAC": [os.path.join(_DATASET_ROOT, "CPTAC", "uni_level1_256")],
}
NUM_CLASSES = {"TCGA-NSCLC": 2, "TCGA-RCC": 3, "CPTAC": 3}
SPLITS = ("train", "val", "test")
KINDS = ("last", "best")
FOLDS = (0, 1, 2, 3)
SUFFIXES = ("_LUAD", "_LUSC", "_KIRC", "_KIRP", "_KICH")
METRIC_KEYS = ("accuracy", "auc", "f1", "mcc", "recall", "precision")

# kind x split -> csv name stem before _{dataset}_clean.csv
OUTPUTS = (
    ("last", "train", "train_last"),
    ("best", "train", "train_best"),
    ("last", "val", "val_last"),
    ("best", "val", "val_best"),
    ("last", "test", "test_last"),
)


def long_path(path: str) -> str:
    path = os.path.abspath(path)
    if path.startswith("\\\\?\\"):
        return path
    return "\\\\?\\" + path


def load_py_module(alias: str, path: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_softmax_dim():
    orig = F.softmax

    def _softmax(input, dim=None, *args, **kwargs):
        if dim is None:
            dim = -1
        return orig(input, dim, *args, **kwargs)

    F.softmax = _softmax
    torch.nn.functional.softmax = _softmax


class MeanSAGEConv(nn.Module):
    """PyG SAGEConv subset used by GDAMIL: mean of neighbors, then lin_l + lin_r(root)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        src, dst = edge_index
        n, _ = x.shape
        neigh = torch.zeros(n, x.shape[1], device=x.device, dtype=x.dtype)
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        neigh.index_add_(0, dst, x.index_select(0, src))
        deg.index_add_(0, dst, torch.ones(src.shape[0], device=x.device, dtype=x.dtype))
        neigh = neigh / deg.clamp(min=1).unsqueeze(1)
        return self.lin_l(neigh) + self.lin_r(x)


class GlobalAttentionReadout(nn.Module):
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, x, batch=None, size=None):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        gate = self.gate_nn(x).view(-1, 1)
        gate = torch.softmax(gate, dim=0)
        return torch.sum(gate * x, dim=0, keepdim=True)


def build_gdamil(attn_cls, n_classes, hid_dim, out_dim, k_components, k_neighbors, in_dim=1024):
    class BagPartition(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_components = k_components
            self.attention = attn_cls(in_dim=hid_dim, hid_dim=hid_dim, out_dim=out_dim)
            self.cluster_logits = nn.Linear(hid_dim, k_components)

        def forward(self, x):
            x = x.squeeze(0)
            p = F.gumbel_softmax(self.cluster_logits(x), tau=0.5, hard=False)
            return p.transpose(0, 1) @ x

    class DynamicGraphBuilder(nn.Module):
        def __init__(self):
            super().__init__()
            self.W_head = nn.Linear(hid_dim, hid_dim)
            self.W_tail = nn.Linear(hid_dim, hid_dim)
            self.scale = hid_dim ** -0.5
            self.topk = k_neighbors

        def forward(self, x):
            e_h = self.W_head(x)
            e_t = self.W_tail(x)
            logits = (e_h @ e_t.transpose(0, 1)) * self.scale
            k = min(self.topk, x.size(0))
            topk_val, topk_idx = torch.topk(logits, k=k, dim=-1)
            weights = torch.softmax(topk_val, dim=-1)
            n = x.size(0)
            src = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
            dst = topk_idx.reshape(-1)
            edge_index = torch.stack([src, dst], dim=0)
            edge_weight = weights.reshape(-1)
            return edge_index, edge_weight

    class GraphAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs_list = nn.ModuleList([MeanSAGEConv(hid_dim, out_dim)])
            self.graph_builder = DynamicGraphBuilder()
            self.lin_sum = nn.Linear(out_dim, out_dim)
            self.lin_bi = nn.Linear(out_dim, out_dim)
            self.gate_U = nn.Linear(out_dim, out_dim // 2)
            self.gate_V = nn.Linear(out_dim, out_dim // 2)
            self.gate_W = nn.Linear(out_dim // 2, out_dim)
            self.norm = nn.LayerNorm(out_dim)

        def forward(self, x):
            edge_index, edge_weight = self.graph_builder(x)
            for conv in self.convs_list:
                x = F.leaky_relu(conv(x, edge_index))
            row, col = edge_index
            summed = torch.zeros_like(x).index_add_(0, row, x[col] * edge_weight.unsqueeze(1))
            sum_msg = self.lin_sum(x + summed)
            bi_msg = self.lin_bi(x * summed)
            u = self.gate_U(x)
            v = self.gate_V(summed)
            g = torch.sigmoid(self.gate_W(u + v))
            out = F.leaky_relu(g * sum_msg + (1.0 - g) * bi_msg)
            return self.norm(out + x), edge_index

    class FeatureFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())
            self.transform = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.LeakyReLU())

        def forward(self, x1, x2):
            combined = torch.cat([x1, x2], dim=1)
            gate = self.gate(combined)
            transformed = self.transform(combined)
            return gate * x1 + (1.0 - gate) * transformed

    class GDAMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.LeakyReLU())
            self.bag_partition = BagPartition()
            self.gnn = GraphAttention()
            self.basic = attn_cls(hid_dim, hid_dim, out_dim, dropout=0.1)
            self.feature_fusion = FeatureFusion()
            self.bag_embedding = GlobalAttentionReadout(
                nn.Sequential(
                    nn.Linear(out_dim, out_dim // 2),
                    nn.LeakyReLU(),
                    nn.Linear(out_dim // 2, 1),
                )
            )
            self.basic_linear = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.LeakyReLU())
            self.classifier = nn.Sequential(
                nn.Linear(out_dim, hid_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(hid_dim, n_classes),
            )

        def forward(self, data):
            x = self.encoder(data["x"])
            x_soft = self.bag_partition(x)
            x_soft, _ = self.gnn(x_soft)
            b_gnn = self.bag_embedding(x_soft)
            b_basic, _ = self.basic(x)
            b_basic = self.basic_linear(b_basic)
            b = self.feature_fusion(b_gnn, b_basic)
            logits = self.classifier(b)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            if not self.training:
                logits = torch.softmax(logits, dim=-1)
            return {"probs": logits}

    return GDAMIL()


class WiKG(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_classes, topk=6, dropout=0.3, fc_name="fc1"):
        super().__init__()
        proj = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        setattr(self, fc_name, proj)
        self._fc_name = fc_name
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)
        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)
        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)
        self.att_net = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden // 2, 1),
        )

    def forward(self, data):
        x = data["x"].unsqueeze(0)
        x = getattr(self, self._fc_name)(x)
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5
        e_h = self.W_head(x)
        e_t = self.W_tail(x)
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)
        k = min(self.topk, e_t.size(1))
        topk_weight, topk_index = torch.topk(attn_logit, k=k, dim=-1)
        topk_index = topk_index.long()
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = torch.arange(e_t.size(0), device=x.device).view(-1, 1, 1)
        nb_h = e_t[batch_indices, topk_index_expanded, :]
        topk_prob = torch.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), nb_h) + torch.matmul(
            (1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2)
        )
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, k, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum("ijkl,ijkm->ijk", nb_h, gate)
        ka_prob = torch.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_nh = torch.matmul(ka_prob, nb_h).squeeze(dim=2)
        sum_embedding = self.activation(self.linear1(e_h + e_nh))
        bi_embedding = self.activation(self.linear2(e_h * e_nh))
        embedding = sum_embedding + bi_embedding
        h = self.message_dropout(embedding).squeeze(0)
        score = self.att_net(h)
        score_softmax = torch.softmax(score, dim=0)
        h = torch.matmul(score_softmax.transpose(0, 1), h)
        h = self.norm(h)
        return {"probs": self.fc(h)}


def detect_arch(sd):
    keys = list(sd.keys())
    if any(k.startswith("gab_blocks.") for k in keys):
        return "ilra"
    if any(k.startswith("feat_proj.") for k in keys):
        return "dsmil"
    if any(k.startswith("encoder.") for k in keys):
        return "gdamil"
    if any(k.startswith("fc1.") or k.startswith("_fc1.") or k.startswith("W_head.") for k in keys):
        return "wikg"
    if any(k.startswith("attention.") for k in keys):
        return "abmil"
    raise KeyError("unrecognized checkpoint keys: " + ", ".join(keys[:8]))


def build_model(arch, sd, modules):
    if arch == "abmil":
        n_classes = sd["classifier.0.weight"].shape[0]
        feat_dim = sd["attention.0.weight"].shape[1]
        return modules["abmil"].ABMIL(n_classes=n_classes, feat_dim=feat_dim)
    if arch == "dsmil":
        n_classes = sd["i_classifier.fc.0.weight"].shape[0]
        dim_in = sd["feat_proj.projecter.0.weight"].shape[1]
        dim_emb = sd["b_classifier.q.weight"].shape[0]
        return modules["dsmil"].DSMIL(dim_in=dim_in, dim_emb=dim_emb, num_cls=n_classes)
    if arch == "ilra":
        n_classes, hidden = sd["classifier.weight"].shape
        with contextlib.redirect_stdout(io.StringIO()):
            return modules["ilra"].ILRA(
                num_layers=2, feat_dim=1024, n_classes=n_classes, hidden_feat=hidden, num_heads=8, topk=1, ln=True
            )
    if arch == "wikg":
        fc_name = "fc1" if any(k.startswith("fc1.") for k in sd) else "_fc1"
        weight = sd[f"{fc_name}.0.weight"]
        dim_hidden, dim_in = weight.shape
        n_classes = sd["fc.weight"].shape[0]
        return WiKG(dim_in=dim_in, dim_hidden=dim_hidden, n_classes=n_classes, fc_name=fc_name)
    if arch == "gdamil":
        hid_dim, in_dim = sd["encoder.0.weight"].shape
        out_dim = sd["gnn.lin_sum.weight"].shape[0]
        k_components = sd["bag_partition.cluster_logits.weight"].shape[0]
        n_classes = sd["classifier.3.weight"].shape[0]
        k_neighbors = min(10, k_components)
        return build_gdamil(modules["attn"].Attention, n_classes, hid_dim, out_dim, k_components, k_neighbors, in_dim)
    raise KeyError(arch)


def as_probs(out):
    """Return the vector the training eval consumed.

    ABMIL / DSMIL / ILRA / GDAMIL already softmax in eval mode.
    WiKG stores logits in ``probs`` and the original binary metrics threshold
    that logit directly, so this function must not softmax again.
    """
    if isinstance(out, dict):
        probs = out["probs"]
    elif isinstance(out, (tuple, list)):
        probs = out[0]
    else:
        probs = out
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    return probs


def compute_metrics(probs, labels, num_classes):
    preds = torch.tensor(probs, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.long)
    task_type = "binary" if num_classes == 2 else "multiclass"
    task_kwargs = {"num_classes": num_classes} if task_type == "multiclass" else {}
    score = preds[:, 1] if task_type == "binary" else preds
    metrics = {}
    metrics["accuracy"] = round(Accuracy(task=task_type, **task_kwargs)(score, target).item() * 100, 2)
    metrics["auc"] = round(AUROC(task=task_type, **task_kwargs)(score, target).item() * 100, 2)
    metrics["f1"] = round(F1Score(task=task_type, average="macro", **task_kwargs)(score, target).item() * 100, 2)
    metrics["mcc"] = round(MatthewsCorrCoef(task=task_type, **task_kwargs)(score, target).item() * 100, 2)
    metrics["recall"] = round(Recall(task=task_type, average="macro", **task_kwargs)(score, target).item() * 100, 2)
    metrics["precision"] = round(Precision(task=task_type, average="macro", **task_kwargs)(score, target).item() * 100, 2)
    return metrics


def write_fold_csv(path, rows):
    df = pd.DataFrame(rows, columns=list(METRIC_KEYS))
    mean_values = df.mean(numeric_only=True).round(2)
    std_values = df.std(numeric_only=True).round(2)
    df = pd.concat([df, pd.DataFrame([mean_values]), pd.DataFrame([std_values])], ignore_index=True)
    target = long_path(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    df.to_csv(target, index=False)


def csv_complete(path):
    path = long_path(path)
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return len(df) >= 6 and all(k in df.columns for k in METRIC_KEYS)


def output_path(leaf, dataset, stem):
    return os.path.join(leaf, f"{stem}_{dataset}_clean.csv")


def job_done(leaf, dataset):
    return all(csv_complete(output_path(leaf, dataset, stem)) for _, _, stem in OUTPUTS)


def index_features(dataset):
    index = {}
    for root in DATA_ROOTS[dataset]:
        if not os.path.isdir(root):
            raise FileNotFoundError(root)
        for name in os.listdir(root):
            if name.endswith(".pt"):
                index[name[:-3]] = os.path.join(root, name)
    return index


def resolve_slide(index, name):
    if name in index:
        return index[name]
    for suffix in SUFFIXES:
        if name.endswith(suffix) and name[: -len(suffix)] in index:
            return index[name[: -len(suffix)]]
    return None


def load_bag(path):
    obj = torch.load(long_path(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "l34" in obj:
            obj = obj["l34"]
        elif "l3" in obj:
            obj = obj["l3"]
        else:
            return None
    if not torch.is_tensor(obj):
        return None
    if obj.dim() != 2 or obj.size(0) <= 50:
        return None
    if obj.size(1) > 1024:
        obj = obj[:, :1024]
    return obj.float().contiguous()


def load_state(path):
    sd = torch.load(long_path(path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    cleaned = {}
    for key, value in sd.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def ckpt_path(leaf, model_name, kind, fold):
    tag = "last" if kind == "last" else "val_best"
    path = os.path.join(leaf, "checkpoint", f"{model_name}_{tag}_fold{fold}.pth")
    if os.path.exists(long_path(path)):
        return path
    return None


def iter_jobs(datasets):
    jobs = []
    for dirpath, dirnames, filenames in os.walk(long_path(RESULT_ROOT)):
        dirnames[:] = [d for d in dirnames if d not in (".ipynb_checkpoints", "_split_eval_cache", "summary_tables")]
        if os.path.basename(dirpath.rstrip("\\")) != "checkpoint":
            continue
        if not any(name.endswith(".pth") for name in filenames):
            continue
        leaf = os.path.dirname(dirpath)
        if leaf.startswith("\\\\?\\"):
            leaf = leaf[4:]
        dataset = os.path.basename(os.path.dirname(leaf))
        model_name = os.path.basename(os.path.dirname(os.path.dirname(leaf)))
        if dataset not in datasets:
            continue
        jobs.append((model_name, dataset, leaf))
    jobs.sort()
    return jobs


def read_splits(dataset):
    excel = os.path.join(EXCEL_ROOT, dataset, "Kfold_4_seed0.xlsx")
    splits = {}
    for fold in FOLDS:
        splits[fold] = {}
        for split in SPLITS:
            df = pd.read_excel(excel, sheet_name=f"{split}_fold{fold}")
            splits[fold][split] = list(zip(df["file_name"].tolist(), df["label"].tolist()))
    return splits


def cache_file(leaf, kind, fold):
    digest = hashlib.sha1(leaf.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_ROOT, digest, f"{kind}_fold{fold}.npz")


def save_cache(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)


def load_cache(path):
    data = np.load(path)
    out = {}
    for split in SPLITS:
        out[split] = (data[f"{split}_probs"], data[f"{split}_labels"])
    return out


def discover_modules():
    club = os.path.join(ROOT, "Model_Club")
    return {
        "abmil": load_py_module("nn_abmil", os.path.join(club, "abmil.py")),
        "dsmil": load_py_module("nn_dsmil", os.path.join(club, "DSMIL.py")),
        "ilra": load_py_module("nn_ilra", os.path.join(club, "ILRA.py")),
        "attn": load_py_module("ig_attn", os.path.join(club, "Attention.py")),
    }


def model_key(arch, sd):
    if arch == "abmil":
        return ("abmil", int(sd["classifier.0.weight"].shape[0]), int(sd["attention.0.weight"].shape[1]))
    if arch == "dsmil":
        return ("dsmil", int(sd["i_classifier.fc.0.weight"].shape[0]), int(sd["b_classifier.q.weight"].shape[0]))
    if arch == "ilra":
        return ("ilra", tuple(int(x) for x in sd["classifier.weight"].shape))
    if arch == "wikg":
        fc_name = "fc1" if any(k.startswith("fc1.") for k in sd) else "_fc1"
        return ("wikg", fc_name, tuple(int(x) for x in sd[f"{fc_name}.0.weight"].shape), int(sd["fc.weight"].shape[0]))
    if arch == "gdamil":
        return (
            "gdamil",
            int(sd["encoder.0.weight"].shape[0]),
            int(sd["gnn.lin_sum.weight"].shape[0]),
            int(sd["bag_partition.cluster_logits.weight"].shape[0]),
            int(sd["classifier.3.weight"].shape[0]),
        )
    return (arch,)


def _n_classes_of(arch, sd):
    if arch == "abmil":
        return int(sd["classifier.0.weight"].shape[0])
    if arch == "dsmil":
        return int(sd["i_classifier.fc.0.weight"].shape[0])
    if arch == "ilra":
        return int(sd["classifier.weight"].shape[0])
    if arch == "wikg":
        return int(sd["fc.weight"].shape[0])
    return int(sd["classifier.3.weight"].shape[0])


@torch.inference_mode()
def run_fold_kind(jobs, fold, kind, splits, feature_index, modules, device):
    prepared = []
    arch_mismatch = []
    for model_name, dataset, leaf in jobs:
        path = ckpt_path(leaf, model_name, kind, fold)
        if path is None:
            print(f"    skip missing {kind} fold{fold}: {leaf}", flush=True)
            continue
        sd = load_state(path)
        arch = detect_arch(sd)
        if arch != model_name:
            arch_mismatch.append((model_name, arch, leaf))
        model = build_model(arch, sd, modules).to(device)
        model.eval()
        model.load_state_dict(sd, strict=True)
        prepared.append({"leaf": leaf, "model": model, "n_classes": _n_classes_of(arch, sd), "arch": arch})
        del sd

    bucket = {item["leaf"]: {split: [] for split in SPLITS} for item in prepared}
    labels_bucket = {split: [] for split in SPLITS}
    missing = 0
    used = {split: 0 for split in SPLITS}
    first_probs_checked = False
    for split in SPLITS:
        for name, label in splits[fold][split]:
            path = resolve_slide(feature_index, name)
            bag = load_bag(path) if path else None
            if bag is None:
                missing += 1
                continue
            bag = bag.to(device, non_blocking=True)
            labels_bucket[split].append(int(label))
            used[split] += 1
            if used[split] % 200 == 0:
                print(f"    {split} {used[split]}", flush=True)
            for item in prepared:
                try:
                    probs = as_probs(item["model"]({"x": bag}))
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    item["model"].cpu()
                    torch.cuda.empty_cache()
                    probs = as_probs(item["model"]({"x": bag.cpu()}))
                    item["model"].to(device)
                bucket[item["leaf"]][split].append(probs.squeeze(0).detach().float().cpu())
            del bag
    result = {}
    label_arrays = {split: np.asarray(labels_bucket[split], dtype=np.int64) for split in SPLITS}
    for item in prepared:
        payload = {}
        for split in SPLITS:
            if bucket[item["leaf"]][split]:
                probs = torch.stack(bucket[item["leaf"]][split], dim=0).numpy()
            else:
                probs = np.zeros((0, item["n_classes"]), dtype=np.float32)
            payload[f"{split}_probs"] = probs
            payload[f"{split}_labels"] = label_arrays[split]
        result[item["leaf"]] = payload
    for item in prepared:
        item["model"].cpu()
        del item["model"]
    del prepared
    torch.cuda.empty_cache()
    return result, missing, used, arch_mismatch


def materialize_job(leaf, dataset, num_classes_expected):
    rows = {stem: [] for _, _, stem in OUTPUTS}
    for fold in FOLDS:
        for kind, split, stem in OUTPUTS:
            cache = cache_file(leaf, kind, fold)
            data = load_cache(cache)
            probs, labels = data[split]
            if len(labels) == 0:
                raise RuntimeError(f"empty split {split} fold {fold} for {leaf}")
            n_classes = probs.shape[1]
            rows[stem].append(compute_metrics(probs, labels, n_classes))
    for _, _, stem in OUTPUTS:
        write_fold_csv(output_path(leaf, dataset, stem), rows[stem])


def caches_ready(leaf):
    return all(os.path.exists(cache_file(leaf, kind, fold)) for kind in KINDS for fold in FOLDS)


def compare_existing(leaf, dataset, tol=0.05):
    existing = os.path.join(leaf, f"val_{dataset}_clean.csv")
    if not os.path.exists(long_path(existing)):
        return None
    got_rows = []
    for fold in FOLDS:
        data = load_cache(cache_file(leaf, "best", fold))
        probs, labels = data["test"]
        got_rows.append(compute_metrics(probs, labels, probs.shape[1]))
    ref = pd.read_csv(long_path(existing))
    diffs = []
    for i, row in enumerate(got_rows):
        for key in METRIC_KEYS:
            diffs.append(abs(float(row[key]) - float(ref.iloc[i][key])))
    return max(diffs), got_rows, ref.iloc[:4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="TCGA-NSCLC,TCGA-RCC,CPTAC")
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means all remaining jobs")
    parser.add_argument("--verify", action="store_true", help="compare recomputed test/best against existing val_*_clean.csv")
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--match", default="", help="only jobs whose path contains this text")
    args = parser.parse_args()

    patch_softmax_dim()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for this evaluation")
    print(f"device={device} {torch.cuda.get_device_name(0)}", flush=True)

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    modules = discover_modules()
    all_jobs = iter_jobs(set(datasets))
    pending = []
    for model_name, dataset, leaf in all_jobs:
        if job_done(leaf, dataset):
            continue
        if any(ckpt_path(leaf, model_name, kind, fold) is None for kind in KINDS for fold in FOLDS):
            print(f"skip incomplete checkpoints: {leaf}", flush=True)
            continue
        pending.append((model_name, dataset, leaf))
    if args.match:
        pending = [job for job in pending if args.match in job[2]]
    if args.max_jobs > 0:
        pending = pending[: args.max_jobs]
    print(f"jobs total={len(all_jobs)} pending={len(pending)}", flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return

    by_dataset = defaultdict(list)
    for job in pending:
        by_dataset[job[1]].append(job)

    mismatched = []
    for dataset in datasets:
        jobs = by_dataset.get(dataset, [])
        if not jobs:
            continue
        print(f"\n=== {dataset} jobs={len(jobs)} ===", flush=True)
        feature_index = index_features(dataset)
        splits = read_splits(dataset)
        for fold in FOLDS:
            for kind in KINDS:
                need = [job for job in jobs if not os.path.exists(cache_file(job[2], kind, fold))]
                if not need:
                    print(f"  cache hit {kind} fold{fold}", flush=True)
                    continue
                # Recompute the whole pending set for this fold so models stay grouped.
                t0 = time.time()
                print(f"  infer {kind} fold{fold} models={len(jobs)}", flush=True)
                result, missing, used, arch_mismatch = run_fold_kind(
                    jobs, fold, kind, splits, feature_index, modules, device
                )
                for model_name, arch, leaf in arch_mismatch:
                    print(f"    arch note folder={model_name} weights={arch}", flush=True)
                for leaf, payload in result.items():
                    save_cache(cache_file(leaf, kind, fold), payload)
                print(
                    f"    done {time.time()-t0:.1f}s missing_slides={missing} used={used}",
                    flush=True,
                )
        for _, _, leaf in jobs:
            if not caches_ready(leaf):
                print(f"  incomplete caches, skip csv {leaf}", flush=True)
                continue
            materialize_job(leaf, dataset, NUM_CLASSES[dataset])
            if args.verify:
                compared = compare_existing(leaf, dataset, args.tol)
                if compared is None:
                    print(f"  no existing test csv {leaf}", flush=True)
                    continue
                max_diff, got_rows, ref = compared
                rel = os.path.relpath(leaf, RESULT_ROOT)
                best_sd = load_state(ckpt_path(leaf, os.path.basename(os.path.dirname(os.path.dirname(leaf))), "best", 0))
                arch = detect_arch(best_sd)
                print(f"  verify {rel} arch={arch} max_abs_diff={max_diff:.4f}", flush=True)
                if max_diff > args.tol:
                    print(f"    got={got_rows[0]}", flush=True)
                    print(f"    ref={ref.iloc[0].to_dict()}", flush=True)
                    if arch == "gdamil":
                        print("    gdamil uses gumbel assignment, skip hard fail", flush=True)
                    else:
                        mismatched.append((rel, max_diff, arch))
    if mismatched:
        print("MISMATCH", mismatched, flush=True)
        sys.exit(2)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

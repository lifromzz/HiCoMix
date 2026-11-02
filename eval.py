from torchmetrics import AUROC, F1Score, Recall, Precision, Accuracy, MatthewsCorrCoef
import torch
import os
import csv
import pandas as pd
import numpy as np


def _mil_forward_probs(model, bag: torch.Tensor) -> torch.Tensor:
    """与 engine 一致：支持单包 (N,D) 与批量包 (B,N,D)。"""
    if bag.dim() == 2:
        n = bag.shape[0]
        x = bag
        batch_idx = torch.zeros(n, dtype=torch.long, device=bag.device)
    elif bag.dim() == 3:
        bsz, n, dim = bag.shape
        x = bag.reshape(bsz * n, dim)
        batch_idx = torch.arange(bsz, device=bag.device).repeat_interleave(n)
    else:
        raise ValueError(f"不支持的 bag 形状: {tuple(bag.shape)}")
    out = model({"x": x, "batch": batch_idx})
    p = out["probs"]
    if p.dim() == 1:
        p = p.unsqueeze(0)
    return p


def evaluate(model, loader, config, csv_path=None, mean=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bags, labels in loader:
            if isinstance(bags, tuple):
                bags = bags[0]
            bags = bags.to(config.device)
            probs = _mil_forward_probs(model, bags)
            preds = probs.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.view(-1).numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    result = get_result(all_preds, all_labels,
                        csv_path=csv_path,
                        fold=config.fold,
                        num_classes=config.num_classes,
                        mean=mean)
    return result["auc"], result["accuracy"]


def get_result(preds, target, fold, num_classes, csv_path=None, mean=False):
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)

    metrics = {}
    task_type = "binary" if num_classes == 2 else "multiclass"
    task_kwargs = {}
    if task_type == "multiclass":
        task_kwargs["num_classes"] = num_classes

    if task_type == "binary":
        preds = preds[:, 1]

    accuracy = Accuracy(task=task_type, **task_kwargs)
    metrics["accuracy"] = round(accuracy(preds, target).item() * 100, 2)

    auroc = AUROC(task=task_type, **task_kwargs)
    metrics["auc"] = round(auroc(preds, target).item() * 100, 2)

    f1 = F1Score(task=task_type, **task_kwargs, average="macro")
    metrics["f1"] = round(f1(preds, target).item() * 100, 2)

    mcc = MatthewsCorrCoef(task=task_type, **task_kwargs)
    metrics["mcc"] = round(mcc(preds, target).item() * 100, 2)

    recall = Recall(task=task_type, **task_kwargs, average="macro")
    metrics["recall"] = round(recall(preds, target).item() * 100, 2)

    precision = Precision(task=task_type, **task_kwargs, average="macro")
    metrics["precision"] = round(precision(preds, target).item() * 100, 2)

    print(metrics)

    if csv_path is not None:
        w_a = "w" if fold == 0 else "a"
        write_dict_to_csv(csv_path, metrics, w_a)
    if mean:
        compute_mean(csv_path)
    return metrics


def compute_mean(csv_path):
    df = pd.read_csv(csv_path)
    mean_values = df.mean(numeric_only=True).round(2)
    std_values = df.std(numeric_only=True).round(2)
    mean_row = pd.DataFrame([mean_values])
    std_row = pd.DataFrame([std_values])
    df = pd.concat([df, mean_row, std_row], ignore_index=True)
    df.to_csv(csv_path, index=False)


def write_dict_to_csv(filename, data_dict, w_a):
    with open(filename, w_a, newline='') as csvfile:
        fieldnames = list(data_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.stat(filename).st_size == 0:
            writer.writeheader()
        writer.writerow(data_dict)

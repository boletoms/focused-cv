"""Shared evaluation for Stage A: features, kNN top-1, the linear VOC probe and mIoU, paired bootstrap (brief section 8).

Every structure is scored the same way; `structure` is passed through to the wrapper (None = dense).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ImageList
from .model import N_PREFIX


def loader(name, size, batch, workers, with_mask=False):
    return DataLoader(ImageList(name, size, with_mask), batch_size=batch, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=False)


@torch.no_grad()
def cls_features(model, name, size, batch, workers, dtype, structure=None):
    out = []
    labels = []
    for x, y, _ in loader(name, size, batch, workers):
        f = model(x.cuda(non_blocking=True), structure=structure, dtype=dtype).final[:, 0]
        out.append(F.normalize(f, dim=-1))
        labels.append(y)
    return torch.cat(out), torch.cat(labels).cuda()


@torch.no_grad()
def knn_correct(bank, bank_labels, feats, labels, k=20, temperature=0.07, classes=100):
    """Per-image correctness of a temperature-weighted k-NN vote on cosine similarity (DINO protocol)."""
    correct = []
    for i in range(0, len(feats), 1000):
        s, idx = (feats[i:i + 1000] @ bank.T).topk(k, dim=-1)
        votes = torch.zeros(len(s), classes, device=s.device).scatter_add_(1, bank_labels[idx], (s / temperature).exp())
        correct.append(votes.argmax(-1) == labels[i:i + 1000])
    return torch.cat(correct).cpu().numpy()


@torch.no_grad()
def patch_features(model, name, size, batch, workers, dtype, structure=None):
    """Final patch tokens (after the final norm) as bf16 (n, N, D), and labels (n, size, size) uint8, on the GPU."""
    feats, masks = [], []
    for x, m, _ in loader(name, size, batch, workers, with_mask=True):
        feats.append(model(x.cuda(non_blocking=True), structure=structure, dtype=dtype).final[:, N_PREFIX:].to(torch.bfloat16))
        masks.append(m.cuda())
    return torch.cat(feats), torch.cat(masks)


def probe_logits(probe, f, size):
    n, N, _ = f.shape
    g = int(N ** 0.5)
    z = probe(f.float()).transpose(1, 2).reshape(n, -1, g, g)
    return F.interpolate(z, size=(size, size), mode="bilinear", align_corners=False)


def train_probe(feats, masks, size, cfg, seed):
    torch.manual_seed(seed)
    probe = torch.nn.Linear(feats.shape[-1], cfg["classes"]).cuda()
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(feats), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), cfg["batch"]):
            j = perm[i:i + cfg["batch"]].cuda()
            loss = F.cross_entropy(probe_logits(probe, feats[j], size), masks[j].long(), ignore_index=cfg["ignore"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        losses.append(tot / len(perm))
    return probe.eval(), losses


@torch.no_grad()
def confusion_per_image(probe, model, name, size, batch, workers, dtype, classes=21, ignore=255, structure=None):
    """(n, classes, classes) confusion counts per image (rows = truth)."""
    out = []
    for x, m, _ in loader(name, size, batch, workers, with_mask=True):
        f = model(x.cuda(non_blocking=True), structure=structure, dtype=dtype).final[:, N_PREFIX:]
        pred = probe_logits(probe, f, size).argmax(1)
        m = m.cuda().long()
        for p, t in zip(pred, m):
            keep = t != ignore
            out.append(torch.bincount(t[keep] * classes + p[keep], minlength=classes * classes).view(classes, classes).cpu())
    return torch.stack(out).numpy()


def miou(conf):
    conf = conf.astype(np.float64)
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    present = union > 0
    return float((inter[present] / union[present]).mean())


def bootstrap(stat, n_items, resamples, seed):
    """Percentile 95% interval of stat(index array) over image resamples; the same indices pair any two structures."""
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n_items, n_items)) for _ in range(resamples)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

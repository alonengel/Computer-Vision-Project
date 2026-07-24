"""Frozen encoders and feature caching (spec: `_docs/stage_1.pdf`).

All encoder parameters remain frozen; each checkpoint is used with its own
associated preprocessing. Train/validation/test features are extracted and cached
once per (dataset, encoder), and every classifier is trained and evaluated on the
cached features.

  resnet18       ImageNet-1K torchvision checkpoint, 512-d representation taken
                 before the final classification layer.
  dinov2_vits14  facebook/dinov2-small, final class-token representation (384-d).
  clip_rn50      official OpenAI CLIP RN50, frozen image and text encoders —
                 used for the zero-shot branch only, per spec.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import get_device, load_config, repo_path


def encoders_for(dataset):
    """Encoder names configured for a dataset, in a stable order."""
    cfg = load_config()["encoders"]
    return [e for e, c in cfg.items() if dataset in c["datasets"]]


def supervised_encoders_for(dataset):
    """Encoders usable by the linear probe / image-prototype heads (excludes CLIP)."""
    cfg = load_config()["encoders"]
    return [e for e in encoders_for(dataset) if cfg[e]["supervised_heads"]]


def build_encoder(name):
    """Return (extract_fn, dim). extract_fn: list[PIL RGB] -> Tensor[B, dim] (CPU, fp32)."""
    device = get_device()

    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights).to(device).eval()
        model.fc = torch.nn.Identity()  # 512-d representation before the classifier
        preprocess = weights.transforms()
        for p in model.parameters():
            p.requires_grad_(False)

        @torch.no_grad()
        def extract(pil_list):
            imgs = torch.stack([preprocess(p) for p in pil_list]).to(device)
            return model(imgs).float().cpu()

        return extract, 512

    if name == "dinov2_vits14":
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        for p in model.parameters():
            p.requires_grad_(False)

        @torch.no_grad()
        def extract(pil_list):
            inp = proc(images=pil_list, return_tensors="pt").to(device)
            out = model(**inp)
            cls = out.last_hidden_state[:, 0]  # final class token
            return cls.float().cpu()

        return extract, model.config.hidden_size

    if name == "clip_rn50":
        import clip

        model, preprocess = clip.load("RN50", device=device)
        model = model.float().eval()  # fp32: fp16 weights misbehave on ROCm-Windows
        for p in model.parameters():
            p.requires_grad_(False)

        @torch.no_grad()
        def extract(pil_list):
            imgs = torch.stack([preprocess(p) for p in pil_list]).to(device)
            return model.encode_image(imgs).float().cpu()

        return extract, model.visual.output_dim

    raise ValueError(f"unknown encoder {name}")


class _PoolDataset(Dataset):
    def __init__(self, pool):
        self.pool = pool

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool.get_image(i), int(self.pool.labels[i])


def _collate(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels)


@torch.no_grad()
def extract_features(pool, extract_fn, batch_size=128):
    loader = DataLoader(_PoolDataset(pool), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    feats, labels = [], []
    for pil_list, lab in tqdm(loader, desc=f"  {pool.name}/{pool.split}", leave=False):
        feats.append(extract_fn(pil_list))
        labels.append(lab)
    return torch.cat(feats).contiguous(), torch.cat(labels)


def feat_path(dataset, split, encoder):
    d = repo_path(load_config()["paths"]["features_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dataset}_{split}_{encoder}.pt"


def cache_features(pool, encoder_name, extract_fn, dim):
    """Extract + save unless the cache exists. Returns the cache path.

    The cache carries the split's fingerprint (labels + image identities) so that
    downstream code can check the committed training subsets against the split the
    features actually came from.
    """
    from .data import pool_fingerprint

    path = feat_path(pool.name, pool.split, encoder_name)
    if path.exists():
        return path
    feats, labels = extract_features(pool, extract_fn)
    assert (labels.numpy() == pool.labels).all(), "extraction reordered labels"
    torch.save({"features": feats, "labels": labels, "dim": dim,
                "class_names": pool.class_names, "dataset": pool.name,
                "split": pool.split, "encoder": encoder_name,
                "pool_fingerprint": pool_fingerprint(pool)}, path)
    return path


def backfill_fingerprints(pools_by_key):
    """Add `pool_fingerprint` to feature caches written before it was recorded.

    pools_by_key: {(dataset, split): Pool}. Only the metadata field is written;
    the cached tensors are untouched, so no result changes.
    """
    from .data import pool_fingerprint

    updated = []
    for (ds, split), pool in pools_by_key.items():
        fp = pool_fingerprint(pool)
        for enc in encoders_for(ds):
            path = feat_path(ds, split, enc)
            if not path.exists():
                continue
            d = torch.load(path, weights_only=True)
            if d.get("pool_fingerprint") == fp:
                continue
            assert (d["labels"].numpy() == pool.labels).all(), \
                f"{path.name}: cached labels do not match the current split"
            d["pool_fingerprint"] = fp
            torch.save(d, path)
            updated.append(path.name)
    return updated


def load_features(dataset, split, encoder):
    return torch.load(feat_path(dataset, split, encoder), weights_only=True)


# --------------------------------------------------------------------------- #
# CLIP text prototypes (zero-shot branch)
# --------------------------------------------------------------------------- #

def clip_text_path(dataset):
    d = repo_path(load_config()["paths"]["artifacts_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"clip_text_{dataset}.pt"


@torch.no_grad()
def cache_clip_text_prototypes(dataset, class_names):
    """One L2-normalized text prototype per class from the dataset's spec prompt."""
    import clip

    path = clip_text_path(dataset)
    prompt = load_config()["datasets"][dataset]["prompt"]
    if path.exists():
        d = torch.load(path, weights_only=True)
        if d.get("prompt") == prompt and d.get("class_names") == list(class_names):
            return path

    device = get_device()
    model, _ = clip.load("RN50", device=device)
    model = model.float().eval()
    tokens = clip.tokenize([prompt.format(c) for c in class_names]).to(device)
    emb = model.encode_text(tokens).float().cpu()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    torch.save({"text_prototypes": emb, "prompt": prompt,
                "class_names": list(class_names)}, path)
    return path

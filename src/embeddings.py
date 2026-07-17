"""Frozen-backbone feature extraction with on-disk caching, plus CLIP text embeddings.

Backbones (ADR 0001): clip_vitb32 (primary), dinov2_vits14, resnet50 — all frozen.
Caches: results/features/{dataset}_{split}_{backbone}.pt with
{features, labels, dim, class_names}. CLIP text embeddings (stage-2 FM targets):
results/artifacts/clip_text_{dataset}.pt.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import get_device, load_config, repo_path

BACKBONES = ["clip_vitb32", "dinov2_vits14", "resnet50"]

# Prompt templates per dataset. The first template is the "primary" single prompt;
# the full list is the prompt-ensemble ablation (averaged, then re-normalized).
PROMPT_TEMPLATES = {
    "mnist": [
        'a photo of the number: "{}".',
        "a photo of the handwritten digit {}.",
        "a low-resolution photo of the digit {}.",
    ],
    "cifar10": [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a low-resolution photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}.",
    ],
    "mini_imagenet": [
        "a photo of a {}.",
        "a bad photo of a {}.",
        "a photo of one {}.",
        "a close-up photo of a {}.",
        "a bright photo of a {}.",
    ],
}
# Validation-class text embeddings (selection only) use the same templates.
PROMPT_TEMPLATES["mini_imagenet_val"] = PROMPT_TEMPLATES["mini_imagenet"]


def build_backbone(name):
    """Return (extract_fn, dim). extract_fn: list[PIL RGB] -> Tensor[B, dim] (CPU, fp32, no grad).

    All backbones are frozen and used purely as feature extractors.
    """
    device = get_device()

    if name == "clip_vitb32":
        import clip

        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.float().eval()  # fp32: fp16 weights misbehave on ROCm-Windows
        dim = model.visual.output_dim  # 512

        @torch.no_grad()
        def extract(pil_list):
            imgs = torch.stack([preprocess(p) for p in pil_list]).to(device)
            return model.encode_image(imgs).float().cpu()

        return extract, dim

    if name == "dinov2_vits14":
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        dim = model.config.hidden_size  # 384

        @torch.no_grad()
        def extract(pil_list):
            inp = proc(images=pil_list, return_tensors="pt").to(device)
            out = model(**inp)
            feat = out.pooler_output if out.pooler_output is not None \
                else out.last_hidden_state[:, 0]
            return feat.float().cpu()

        return extract, dim

    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights).to(device).eval()
        model.fc = torch.nn.Identity()  # penultimate (2048-d) features
        preprocess = weights.transforms()
        dim = 2048

        @torch.no_grad()
        def extract(pil_list):
            imgs = torch.stack([preprocess(p) for p in pil_list]).to(device)
            return model(imgs).float().cpu()

        return extract, dim

    raise ValueError(f"unknown backbone {name}")


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


def feat_path(dataset, split, backbone):
    d = repo_path(load_config()["paths"]["features_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dataset}_{split}_{backbone}.pt"


def cache_features(pool, backbone_name, extract_fn, dim):
    """Extract + save unless the cache already exists. Returns the cache path."""
    path = feat_path(pool.name, pool.split, backbone_name)
    if path.exists():
        return path
    feats, labels = extract_features(pool, extract_fn)
    assert (labels.numpy() == pool.labels).all(), "extraction shuffled labels"
    torch.save({"features": feats, "labels": labels, "dim": dim,
                "class_names": pool.class_names}, path)
    return path


def load_features(dataset, split, backbone):
    return torch.load(feat_path(dataset, split, backbone), weights_only=True)


def clip_text_path(dataset):
    d = repo_path(load_config()["paths"]["artifacts_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"clip_text_{dataset}.pt"


@torch.no_grad()
def cache_clip_text_embeddings(dataset, class_names):
    """Encode class prompts with CLIP ViT-B/32. Saves both the primary single-prompt
    embeddings and the ensemble average (each L2-normalized) — stage-2 FM targets."""
    import clip

    path = clip_text_path(dataset)
    if path.exists():
        return path
    device = get_device()
    model, _ = clip.load("ViT-B/32", device=device)
    model = model.float().eval()
    templates = PROMPT_TEMPLATES[dataset]
    per_template = []
    for t in templates:
        tokens = clip.tokenize([t.format(c) for c in class_names]).to(device)
        emb = model.encode_text(tokens).float().cpu()
        per_template.append(emb / emb.norm(dim=-1, keepdim=True))
    stack = torch.stack(per_template)  # [T, C, D]
    ensemble = stack.mean(0)
    ensemble = ensemble / ensemble.norm(dim=-1, keepdim=True)
    torch.save({"primary": stack[0], "ensemble": ensemble, "templates": templates,
                "class_names": list(class_names)}, path)
    return path

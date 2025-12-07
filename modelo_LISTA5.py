# corrected_modelo_LISTA5.py
"""
Optimized ConvNeXt-Tiny + handcrafted features + OPTUNA TUNING (Fast mode for Optuna)

Principais mudanças:
 - Light augmentations durante Optuna (rápido)
 - Full augmentations no treino final
 - DataLoader otimizado: num_workers, pin_memory, persistent_workers, prefetch_factor
 - Mixed precision (AMP) quando GPU disponível
 - Precompute features robusto
 - Freeze/unfreeze por stage no ConvNeXt (mais controlado)
 - Melhor cálculo de weighted sampler e reutilização
"""

import os
import random
import math
from glob import glob
import time
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets
from torchvision.models import convnext_tiny

from skimage.feature import local_binary_pattern
import albumentations as A
from albumentations.pytorch import ToTensorV2

import optuna
from optuna.trial import TrialState

# -------------------------
# USER CONFIG
# -------------------------
TRAIN_DIR = r"C:/Users/José Eduardo/OneDrive/Desktop/pyton/dataset/train"
TEST_DIR  = r"C:/Users/José Eduardo/OneDrive/Desktop/pyton/dataset/test"
OUTPUT_CSV = "predicoes_convnext_optuna.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações Fixas
BATCH_SIZE = 32
# Sugestão prática: use min(4, os.cpu_count()//2) para evitar sobrecarga no Windows
NUM_WORKERS = 0
SEED = 1910
TARGET_SIZE = 320  # use 256 se quiser mais velocidade (teste)

# Configurações do Optuna
N_TRIALS = 20          # aumentar se quiser mais busca
OPTUNA_EPOCHS = 6     # reduzido para busca (pruning ajuda)
FINAL_EPOCHS = 45      # treino final com melhores params

# Caching
PRECOMPUTE_FEATURES = True
FEATURES_CACHE = "features_cache.npy"
PATHS_CACHE = "paths_cache.npy"

# Optuna light augmentation flag
USE_LIGHT_AUG_FOR_OPTUNA = True

# Reprodutibilidade
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # allow CUDNN optimizations (faster)
    torch.backends.cudnn.benchmark = True
    # allow TF32 on Ampere+ (trade-off speed/precision; usually OK for convnets)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

seed_everything()

# -------------------------
# Utility classes (top-level so multiprocessing can pickle them)
# -------------------------
class WrappedSubset(Subset):
    """
    Subset wrapper that applies a different transform than the underlying dataset.
    This is top-level (pickleable) so DataLoader workers on Windows can work.
    """
    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform
        self.base_dataset = subset.dataset

    def __getitem__(self, idx):
        # idx is index within the subset
        real_idx = self.indices[idx]
        pil_img, label = self.base_dataset.imgfolder[real_idx]
        if getattr(self.base_dataset, "precompute", False) and getattr(self.base_dataset, "cache", None) is not None:
            feats = self.base_dataset.cache[real_idx].astype(np.float32)
        else:
            feats = extract_handcrafted_features_pil(pil_img)
        img_t = self.transform(pil_img)
        return img_t, torch.from_numpy(feats), label

class FeatureWrapper(nn.Module):
    """
    Top-level feature wrapper for ConvNeXt base to produce a flattened feature vector.
    """
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# -------------------------
# Handcrafted feature helpers
# -------------------------
def apply_clahe_gray_pil(pil_img, clipLimit=2.0, grid=(8,8)):
    arr = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=grid)
    return clahe.apply(arr)

def compute_lbp_hist(gray, P=8, R=1, n_bins=59):
    # ensure uint8 to reduce conversions
    gray = gray.astype(np.uint8)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

def build_gabor_kernels(ksize=21):
    kernels = []
    for theta in np.arange(0, np.pi, np.pi/4):
        for sigma in (3,5):
            lam = 8.0; gamma = 0.5
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
            kernels.append(kern)
    return kernels

GABOR_KERNELS = build_gabor_kernels()

def compute_gabor_energy(gray, kernels=GABOR_KERNELS):
    energies = []
    for k in kernels:
        f = cv2.filter2D(gray, cv2.CV_32F, k)
        energies.append(np.mean(np.abs(f)))
    energies = np.array(energies, dtype=np.float32)
    s = energies.sum()
    if s > 0: energies /= (s + 1e-9)
    return energies

def compute_sobel_orientation_stats(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2) + 1e-9
    ori = np.arctan2(gy, gx)
    mean_ori = math.atan2(np.sum(np.sin(ori)*mag), np.sum(np.cos(ori)*mag))
    mean_ori = float(mean_ori) / math.pi
    std_ori = float(np.std(ori))
    return np.array([mean_ori, std_ori], dtype=np.float32)

def compute_contrast_variance(gray):
    v = float(np.var(gray))
    s = float(np.std(gray))
    mean = float(np.mean(gray))
    contrast = s / (mean + 1e-9) if mean > 0 else s
    return np.array([contrast, v], dtype=np.float32)

def compute_fft_bands(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = np.log(mag + 1e-9)
    h, w = gray.shape
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.max(r) if np.max(r) > 0 else 1.0
    low_mask = r <= (0.08 * max_r)
    mid_mask = (r > (0.08 * max_r)) & (r <= (0.3 * max_r))
    high_mask = (r > (0.3 * max_r))
    low_e = mag[low_mask].mean() if np.any(low_mask) else 0.0
    mid_e = mag[mid_mask].mean() if np.any(mid_mask) else 0.0
    high_e = mag[high_mask].mean() if np.any(high_mask) else 0.0
    arr = np.array([low_e, mid_e, high_e], dtype=np.float32)
    s = arr.sum()
    if s > 0: arr = arr / (s + 1e-9)
    return arr

def extract_handcrafted_features_pil(pil_img):
    gray = apply_clahe_gray_pil(pil_img)
    lbp = compute_lbp_hist(gray)
    gabor = compute_gabor_energy(gray)
    sobel = compute_sobel_orientation_stats(gray)
    cvf = compute_contrast_variance(gray)
    fft = compute_fft_bands(gray)
    feats = np.concatenate([lbp, gabor, sobel, cvf, fft]).astype(np.float32)
    return feats

# -------------------------
# Augmentations (light for Optuna, full for final)
# -------------------------
def get_train_albumentations(light=True):
    if light:
        return A.Compose([
            A.LongestMaxSize(max_size=TARGET_SIZE),
            A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, pad_value=0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=8, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
    else:
        # Full augmentations for final training
        # simplified CoarseDropout args to avoid version-specific warnings
        return A.Compose([
            A.LongestMaxSize(max_size=TARGET_SIZE),
            A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, pad_value=0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=12, p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=40, p=0.25),
                A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.12),
            ], p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.35),
            A.GaussianBlur(blur_limit=(1,3), p=0.12),
            A.CoarseDropout(max_holes=3, max_height=24, max_width=24, p=0.18),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])

class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug
    def __call__(self, pil_img):
        arr = np.array(pil_img)
        res = self.aug(image=arr)
        return res['image']

# -------------------------
# Datasets
# -------------------------
class HybridTrainDataset(Dataset):
    def __init__(self, root_dir, albumentations_transform, precompute=False, cache_features_path=FEATURES_CACHE, cache_paths_path=PATHS_CACHE, force_recompute=False):
        self.imgfolder = datasets.ImageFolder(root_dir)
        self.transform = albumentations_transform
        self.precompute = precompute
        self.cache_features_path = cache_features_path
        self.cache_paths_path = cache_paths_path
        self.cache = None

        if precompute:
            if (not force_recompute) and os.path.exists(self.cache_features_path) and os.path.exists(self.cache_paths_path):
                try:
                    paths = np.load(self.cache_paths_path, allow_pickle=True)
                    if len(paths) == len(self.imgfolder.samples):
                        print("Loading handcrafted features cache...")
                        self.cache = np.load(self.cache_features_path, allow_pickle=False)
                    else:
                        print("Cache mismatch. Recomputing...")
                        self._create_cache()
                except Exception as e:
                    print("Cache load error:", e)
                    self._create_cache()
            else:
                print("No cache found or force_recompute. Creating...")
                self._create_cache()

    def _create_cache(self):
        feats_list = []
        paths = []
        for path, _ in tqdm(self.imgfolder.samples, desc="Precompute features"):
            pil = Image.open(path).convert("RGB")
            feats = extract_handcrafted_features_pil(pil)
            feats_list.append(feats)
            paths.append(path)
        feats_arr = np.stack(feats_list, axis=0).astype(np.float32)
        np.save(self.cache_features_path, feats_arr)
        np.save(self.cache_paths_path, np.array(paths, dtype=object))
        self.cache = feats_arr

    def __len__(self):
        return len(self.imgfolder)

    def __getitem__(self, idx):
        pil_img, label = self.imgfolder[idx]
        if self.precompute and self.cache is not None:
            feats = self.cache[idx].astype(np.float32)
        else:
            feats = extract_handcrafted_features_pil(pil_img)
        img_t = self.transform(pil_img)
        return img_t, torch.from_numpy(feats), label

class HybridTestDataset(Dataset):
    def __init__(self, root_dir, albumentations_transform, exts=(".bmp",".png",".jpg",".jpeg")):
        self.paths = []
        for ext in exts:
            self.paths += sorted(glob(os.path.join(root_dir, f"*{ext}")))
        seen = set(); uniq = []
        for p in self.paths:
            if p not in seen:
                uniq.append(p); seen.add(p)
        self.paths = uniq
        self.transform = albumentations_transform

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(p).convert("RGB")
        feats = extract_handcrafted_features_pil(pil)
        img_t = self.transform(pil)
        return img_t, torch.from_numpy(feats), p

# -------------------------
# Model
# -------------------------
class HybridModel(nn.Module):
    def __init__(self, cnn_backbone, cnn_feat_dim, handcrafted_dim, head_hidden=512, dropout=0.3, n_classes=2):
        super().__init__()
        self.cnn = cnn_backbone
        self.feat_norm = nn.LayerNorm(handcrafted_dim)
        self.head = nn.Sequential(
            nn.Linear(cnn_feat_dim + handcrafted_dim, head_hidden),
            nn.GELU(),
            nn.BatchNorm1d(head_hidden),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes)
        )

    def forward(self, x_img, x_feats):
        cnn_out = self.cnn(x_img)
        x_feats = self.feat_norm(x_feats)
        if x_feats.dtype != cnn_out.dtype:
            x_feats = x_feats.to(cnn_out.dtype)
        x = torch.cat([cnn_out, x_feats], dim=1)
        return self.head(x)

def build_convnext_feature_extractor(unfreeze_last_stages=1):
    """
    Build ConvNeXt tiny and return a feature extractor that outputs a 1D vector per image.
    unfreeze_last_stages: number of stage blocks to unfreeze from the end (0 = fully frozen)
    """
    base = convnext_tiny(weights="IMAGENET1K_V1")
    # find classifier in convnext_tiny and replace to produce flatten output
    feat_dim = base.classifier[-1].in_features
    # Keep LN + Flatten (classifier usually: [LayerNorm, Linear])
    base.classifier = nn.Sequential(base.classifier[0], base.classifier[1])

    # Freeze everything first
    for param in base.parameters():
        param.requires_grad = False

    # Unfreeze last N stages (convnext.features is a Sequential of stages)
    if unfreeze_last_stages > 0:
        children = list(base.features.children())
        cnt = 0
        for child in reversed(children):
            for p in child.parameters():
                p.requires_grad = True
            cnt += 1
            if cnt >= unfreeze_last_stages:
                break

    wrapper = FeatureWrapper(base)
    return wrapper, feat_dim

# -------------------------
# Training Routines
# -------------------------
def get_grad_scaler(device):
    # prefer torch.amp.GradScaler if available, fallback to cuda.amp
    if device.type == "cuda":
        try:
            return torch.amp.GradScaler()
        except Exception:
            return torch.cuda.amp.GradScaler()
    return None

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for imgs, feats, labels in loader:
        imgs = imgs.to(device)
        feats = feats.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):

            outputs = model(imgs, feats)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader) if len(loader)>0 else 0.0

def evaluate(model, loader, device):
    model.eval()
    total = 0; correct = 0; loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, feats, labels in loader:
            imgs = imgs.to(device)
            feats = feats.to(device).float()
            labels = labels.to(device)
            outputs = model(imgs, feats)
            loss = crit(outputs, labels)
            loss_sum += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total if total>0 else 0.0
    return acc, (loss_sum / len(loader)) if len(loader)>0 else 0.0

def predict_test(model, loader, device, label_map=None):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, feats, paths in loader:
            imgs = imgs.to(device)
            feats = feats.to(device).float()
            outputs = model(imgs, feats)
            preds = outputs.argmax(dim=1).cpu().numpy()
            for p, path in zip(preds, paths):
                label = label_map[p] if label_map else ("F" if p==0 else "M")
                results.append((os.path.basename(path), label))
    return results

def stratified_split(imgfolder, val_frac=0.15):
    from sklearn.model_selection import StratifiedShuffleSplit
    targets = [s[1] for s in imgfolder.samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
    return train_idx, val_idx

# -------------------------
# OPTUNA OBJECTIVE
# -------------------------
def objective(trial, train_loader, val_loader, handcrafted_dim, device):
    # Hiperparâmetros
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    head_hidden = trial.suggest_int("head_hidden", 128, 1024, step=128)
    # limitar unfreeze stages (0-2) para trials rápidos
    unfreeze_n = trial.suggest_int("unfreeze_n", 0, 2)

    # Model build
    cnn_base, cnn_feat_dim = build_convnext_feature_extractor(unfreeze_last_stages=unfreeze_n)
    model = HybridModel(cnn_base, cnn_feat_dim, handcrafted_dim, head_hidden=head_hidden, dropout=dropout, n_classes=2)
    model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = get_grad_scaler(device)

    best_val_acc = 0.0
    for epoch in range(OPTUNA_EPOCHS):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_acc, _ = evaluate(model, val_loader, device)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

# -------------------------
# MAIN
# -------------------------
def main():
    print("Device:", DEVICE)
    print(f"NUM_WORKERS={NUM_WORKERS} | BATCH_SIZE={BATCH_SIZE} | TARGET_SIZE={TARGET_SIZE}")

    # 1. Data transforms
    train_aug_light = AlbumentationsTransform(get_train_albumentations(light=True))
    train_aug_full = AlbumentationsTransform(get_train_albumentations(light=False))
    valid_aug = AlbumentationsTransform(get_train_albumentations(light=False))  # validation uses normalized & resize

    # 2. Build dataset (precompute features if enabled)
    full_dataset = HybridTrainDataset(TRAIN_DIR, albumentations_transform=train_aug_full, precompute=PRECOMPUTE_FEATURES)

    # Handcrafted dim
    dummy_feat = full_dataset[0][1]
    handcrafted_dim = dummy_feat.shape[0]
    print(f"Handcrafted dim: {handcrafted_dim}")

    # Stratified split
    train_idx, val_idx = stratified_split(full_dataset.imgfolder, val_frac=0.15)
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    # Weighted sampler (compute once)
    targets_all = [s[1] for s in full_dataset.imgfolder.samples]
    class_counts = np.bincount(targets_all, minlength=2)
    class_weights = 1.0 / (class_counts + 1e-9)
    sample_weights = [class_weights[targets_all[i]] for i in train_idx]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_idx), replacement=True)

    # DataLoader tuning depending on NUM_WORKERS
    if NUM_WORKERS > 0:
        pin_memory = True if (DEVICE.type == 'cuda') else False
        persistent_workers = True
        prefetch_factor = 2
    else:
        # Windows / Optuna safe defaults
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None

    # Create Wrapped subsets (so we can use different augmentations)
    train_ds_light = WrappedSubset(train_ds, train_aug_light)
    train_ds_full = WrappedSubset(train_ds, train_aug_full)
    val_ds_wrapped = WrappedSubset(val_ds, valid_aug)

    # DataLoaders for Optuna (light aug) and validation
    train_loader_optuna = DataLoader(
        train_ds_light,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    val_loader = DataLoader(
        val_ds_wrapped,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    # 3. RUN OPTUNA
    print("\n--- Starting Optuna Study (light aug, frozen conv stages) ---")
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
)


    func = lambda trial: objective(trial, train_loader_optuna, val_loader, handcrafted_dim, DEVICE)

    study.optimize(func, n_trials=N_TRIALS, show_progress_bar=True)

    print("\nBest params found:")
    print(study.best_params)
    print(f"Best Val Acc during search: {study.best_value:.4f}")

    # save study
    try:
        import joblib
        joblib_fname = "optuna_study.pkl"
        joblib.dump(study, joblib_fname)
        print("Saved study to", joblib_fname)
    except Exception as e:
        print("Could not save study:", e)

    # 4. Final training with best params (use full augmentations and possibly unfreeze more if desired)
    print("\n--- Retraining with Best Params (full augmentations) ---")
    best = study.best_params
    # ensure keys exist
    best_unfreeze = best.get("unfreeze_n", 0)
    cnn_base, cnn_feat_dim = build_convnext_feature_extractor(unfreeze_last_stages=best_unfreeze)
    final_model = HybridModel(cnn_base, cnn_feat_dim, handcrafted_dim,
                              head_hidden=best.get("head_hidden", 512),
                              dropout=best.get("dropout", 0.3),
                              n_classes=2)
    final_model.to(DEVICE)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, final_model.parameters()),
                            lr=best.get("lr", 1e-4), weight_decay=best.get("weight_decay", 1e-4))
    criterion = nn.CrossEntropyLoss()
    scaler = get_grad_scaler(DEVICE)

    # use DataLoader with full augmentations for training (reuse sampler)
    train_loader_final = DataLoader(
        train_ds_full,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    val_loader_final = val_loader  # reuse validation loader

    best_final_val = 0.0
    for epoch in range(FINAL_EPOCHS):
        t_loss = train_one_epoch(final_model, train_loader_final, criterion, optimizer, DEVICE, scaler)
        v_acc, v_loss = evaluate(final_model, val_loader_final, DEVICE)
        print(f"Final Epoch {epoch+1}/{FINAL_EPOCHS} | Train L: {t_loss:.4f} | Val L: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

        if v_acc > best_final_val:
            best_final_val = v_acc
            torch.save(final_model.state_dict(), "best_optuna_model.pth")

    print(f"Finished. Best Final Val Acc: {best_final_val}")

    # 5. Predict Test
    if os.path.exists("best_optuna_model.pth"):
        final_model.load_state_dict(torch.load("best_optuna_model.pth", map_location=DEVICE))

    test_ds = HybridTestDataset(TEST_DIR, albumentations_transform=valid_aug)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory,
                             persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    results = predict_test(final_model, test_loader, DEVICE, label_map={0:"F", 1:"M"})
    df = pd.DataFrame(results, columns=["image_path","prediction"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

import os, math, sys, csv, argparse, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR     = "./att_faces"          # s1..s40, 每个文件夹 1.pgm..10.pgm
IMG_H, IMG_W = 112, 92                
N_CLASSES    = 40
N_FEATURES   = IMG_H * IMG_W

TRAIN_PER_CLASS = 5                   # 每类训练张数 → 200 训练 / 200 测试
K_LIST          = [5, 7, 9, 11, 13]   
SEED            = 1

BATCH_ORIG, BATCH_AUG   = 32, 32
EPOCHS_ORIG, EPOCHS_AUG = 80, 200      # 原始集 200 张 / 增强集 5×200=1000 张
LR, WEIGHT_DECAY        = 1e-3, 5e-4
LABEL_SMOOTH            = 0.1         

ROT_ANGLES = (45, 135, 225, 315)      # 左上 / 左下 / 右下 / 右上
ANGLE_NAME = {0: "原始(0°)", 45: "左上(45°)", 135: "左下(135°)",
              225: "右下(225°)", 315: "右上(315°)"}
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESAMPLE   = getattr(Image, "Resampling", Image).BILINEAR
SPLIT_FILE = "orl_split.npz"

def load_data(data_dir=DATA_DIR):
    X, y = [], []
    for i in range(1, N_CLASSES + 1):
        d = os.path.join(data_dir, f"s{i}")
        files = sorted(glob.glob(os.path.join(d, "*.pgm")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        for f in files:
            img = np.asarray(Image.open(f), dtype=np.float64) / 255.0
            X.append(img.ravel()); y.append(i - 1)
    X, y = np.asarray(X), np.asarray(y, dtype=np.int64)
    return X, y

def make_split(y, train_per_class=TRAIN_PER_CLASS):
    tr, te = [], []
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        assert len(idx) == 10
        tr += idx[:train_per_class].tolist()
        te += idx[train_per_class:].tolist()
    return np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)


def rotate_images(X, angle):
    out = np.empty_like(X)
    for i in range(len(X)):
        img = Image.fromarray((X[i].reshape(IMG_H, IMG_W) * 255.0).round().astype(np.uint8))
        out[i] = np.asarray(img.rotate(angle, resample=RESAMPLE, fillcolor=0),
                            dtype=np.float64).ravel() / 255.0
    return out

def build_augmented_train(X_train, y_train, angles=ROT_ANGLES):
    Xs, ys = [X_train], [y_train]
    for a in angles:
        Xs.append(rotate_images(X_train, a)); ys.append(y_train)
    X_aug, y_aug = np.concatenate(Xs, 0), np.concatenate(ys, 0)
    print(f"[增强] 训练集 {X_train.shape[0]} → {X_aug.shape[0]} 张（旋转角 {angles}）")
    return X_aug, y_aug

class FaceDataset(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = (X.reshape(-1, 1, IMG_H, IMG_W).astype(np.float32) - mean) / std
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), self.y[i]

class ORLCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES, p_drop=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),  # 56×46
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),  # 28×23
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),  # 14×11
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256), nn.ReLU(True), nn.Dropout(p_drop),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_cnn(model, loader, epochs, lr, tag):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    for ep in range(1, epochs + 1):
        tot = corr = 0; lsum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()
            lsum += loss.item() * len(yb)
            corr += (out.argmax(1) == yb).sum().item(); tot += len(yb)
        sch.step()
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"[{tag}] epoch {ep:3d}/{epochs}  loss={lsum/tot:.4f}  train_acc={corr/tot:.4f}")
    return model

@torch.no_grad()
def cnn_predict(model, X, mean, std, batch=256):
    model.eval()
    dl = DataLoader(FaceDataset(X, np.zeros(len(X), dtype=np.int64), mean, std),
                    batch_size=batch, shuffle=False)
    return np.concatenate([model(xb.to(DEVICE)).argmax(1).cpu().numpy() for xb, _ in dl])

def pairwise_sqdist_ab(A, B):
    sa = np.einsum("ij,ij->i", A, A)
    sb = np.einsum("ij,ij->i", B, B)
    d2 = sa[:, None] + sb[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2

def majority_vote(neighbor_idx, y, k, n_classes=N_CLASSES):
    n = neighbor_idx.shape[0]
    labels = y[neighbor_idx[:, :k]]
    flat = labels + np.arange(n)[:, None] * n_classes
    counts = np.bincount(flat.ravel(), minlength=n * n_classes).reshape(n, n_classes)
    return counts.argmax(axis=1).astype(np.int64)   

def confusion(y_true, y_pred, n_classes=N_CLASSES):
    cm = np.zeros((n_classes, n_classes), np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def nmi_from_cm(cm, average="arithmetic"):
    cm = np.asarray(cm, dtype=np.float64); n = cm.sum()
    pxy = cm / n; px, py = pxy.sum(1), pxy.sum(0)
    mask = pxy > 0
    mi = float((pxy[mask] * np.log2(pxy[mask] / np.outer(px, py)[mask])).sum())
    def ent(p):
        p = p[p > 0]; return float(-(p * np.log2(p)).sum())
    hx, hy = ent(px), ent(py)
    denom = {"arithmetic": (hx + hy) / 2.0, "geometric": math.sqrt(hx * hy),
             "min": min(hx, hy), "max": max(hx, hy)}[average]
    return mi / denom if denom > 0 else 0.0

def cen_from_cm(cm, modified=False):
    C = np.asarray(cm, dtype=np.float64); N = C.shape[0]; S = C.sum()
    D = C.sum(1) + C.sum(0)
    base = 2.0 * (N - 1)                  
    if not modified:
        Dj, w = D, D / (2.0 * S)
    else:
        Dj = D - np.diag(C); w = Dj / (2.0 * S - np.trace(C))
    total = 0.0
    for j in range(N):
        if Dj[j] <= 0: continue
        h = 0.0
        for k in range(N):
            if k == j: continue
            for c in (C[j, k], C[k, j]):
                if c > 0:
                    p = c / Dj[j]
                    h -= p * math.log(p, base)
        total += w[j] * h
    return float(total)

def knn_predict(X_train, y_train, X_test, k_list=K_LIST, n_classes=N_CLASSES):
    nb = np.argsort(pairwise_sqdist_ab(X_test, X_train), axis=1, kind="stable")[:, :max(k_list)]
    return {k: majority_vote(nb, y_train, k, n_classes) for k in k_list}

def evaluate_preds(y_true, y_pred, n_classes=N_CLASSES):
    cm = confusion(y_true, y_pred, n_classes)
    return float(np.trace(cm) / cm.sum()), nmi_from_cm(cm), \
           nmi_from_cm(cm, "geometric"), cen_from_cm(cm), cm

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, obj):
        for s in self.streams:
            s.write(obj); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def parse_args():
    ap = argparse.ArgumentParser(description="ORL 上的 CNN + 与 kNN 对比（多 seed）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="随机种子列表")
    ap.add_argument("--epochs-orig", type=int, default=EPOCHS_ORIG)
    ap.add_argument("--epochs-aug",  type=int, default=EPOCHS_AUG)
    ap.add_argument("--angles", type=int, nargs="+", default=list(ROT_ANGLES))
    ap.add_argument("--no-knn",    action="store_true")
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--log", type=str, default="cnn.log", help="日志文件名")
    return ap.parse_args()

def run_once(seed, args):
    set_seed(seed)
    print("\n" + "=" * 78)
    print(f"  RUN  seed = {seed}")
    print("=" * 78)

    X, y = load_data(DATA_DIR)
    tr_idx, te_idx = make_split(y)          
    X_tr, y_tr, X_te, y_te = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
    print(f"Loaded ORL: X={X.shape}, device={DEVICE}; "
          f"train={len(tr_idx)}/test={len(te_idx)}")

    mean, std = float(X_tr.mean()), float(X_tr.std())
    if std == 0: std = 1.0

    results, robust, knn_orig_acc = [], [], {}

    # ---- 1) CNN-orig ----
    print(f"\n----- [seed={seed}] 1) CNN-orig 训练 → 原数据集测试 -----")
    cnn_orig = ORLCNN().to(DEVICE)
    dl = DataLoader(FaceDataset(X_tr, y_tr, mean, std), batch_size=BATCH_ORIG, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    train_cnn(cnn_orig, dl, args.epochs_orig, LR, f"CNN-orig(s{seed})")
    acc, na, ng, ce, cm = evaluate_preds(y_te, cnn_predict(cnn_orig, X_te, mean, std))
    print(f"[CNN-orig → 原始测试集] ACC={acc:.4f}  NMI={na:.4f}  CEN={ce:.4f}")
    np.savetxt(f"cnn_cm_orig_s{seed}.csv", cm, fmt="%d", delimiter=",")
    torch.save(cnn_orig.state_dict(), f"cnn_orig_s{seed}.pt")
    results.append(("CNN-orig", "-", acc, na, ng, ce))

    print(f"\n----- [seed={seed}] 2) 数据增强（旋转 {args.angles}）-----")
    X_aug, y_aug = build_augmented_train(X_tr, y_tr, tuple(args.angles))
    cnn_aug = ORLCNN().to(DEVICE)
    dl = DataLoader(FaceDataset(X_aug, y_aug, mean, std), batch_size=BATCH_AUG, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    train_cnn(cnn_aug, dl, args.epochs_aug, LR, f"CNN-aug(s{seed})")
    acc, na, ng, ce, cm = evaluate_preds(y_te, cnn_predict(cnn_aug, X_te, mean, std))
    print(f"[CNN-aug → 原始测试集] ACC={acc:.4f}  NMI={na:.4f}  CEN={ce:.4f}")
    np.savetxt(f"cnn_cm_aug_s{seed}.csv", cm, fmt="%d", delimiter=",")
    torch.save(cnn_aug.state_dict(), f"cnn_aug_s{seed}.pt")
    results.append(("CNN-aug(增强5x)", "-", acc, na, ng, ce))

    if not args.no_knn:
        print(f"\n----- [seed={seed}] 3) kNN 对比 -----")
        for tag, Xtr_, ytr_ in (("kNN-orig", X_tr, y_tr), ("kNN-aug", X_aug, y_aug)):
            preds = knn_predict(Xtr_, ytr_, X_te)
            for k in K_LIST:
                a, na, ng, ce, _ = evaluate_preds(y_te, preds[k])
                results.append((tag, str(k), a, na, ng, ce))
                if tag == "kNN-orig":
                    knn_orig_acc[k] = a
                    print(f"kNN-orig  k={k:<3d} ACC={a:.4f}  NMI={na:.4f}  CEN={ce:.4f}")

    if not args.no_robust:
        print(f"\n----- [seed={seed}] 4) 鲁棒性（旋转测试集）-----")
        best_k = max(knn_orig_acc, key=knn_orig_acc.get) if knn_orig_acc else None
        for a in [0] + list(args.angles):
            Xt = X_te if a == 0 else rotate_images(X_te, a)
            name = ANGLE_NAME.get(a, f"rot{a}°")
            ao = evaluate_preds(y_te, cnn_predict(cnn_orig, Xt, mean, std))[0]
            aa = evaluate_preds(y_te, cnn_predict(cnn_aug, Xt, mean, std))[0]
            ak = float("nan")
            if best_k is not None:
                nb = np.argsort(pairwise_sqdist_ab(Xt, X_tr), axis=1, kind="stable")[:, :best_k]
                ak = float((majority_vote(nb, y_tr, best_k) == y_te).mean())
            print(f"{name:<10} CNN-orig={ao:.4f}  CNN-aug={aa:.4f}"
                  + (f"  kNN(k={best_k})={ak:.4f}" if best_k is not None else ""))
            robust.append((name, ao, aa, ak))
    return results, robust

def summarize(all_results, all_robust, seeds):
    from collections import defaultdict
    groups = defaultdict(list)
    for results in all_results:
        for r in results:
            groups[(r[0], r[1])].append(r[2:])

    print("\n" + "=" * 78)
    print(f"跨 seed 汇总（seeds={seeds}，mean±std）")
    print("=" * 78)
    print(f"{'方法':<18}{'k':>4}  {'ACC':>16}  {'NMI':>16}  {'CEN':>16}")
    rows = []
    for (m, k), vals in groups.items():
        a = np.asarray(vals, float)
        mu = a.mean(0)
        sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1])
        rows.append((m, k, mu, sd, len(a)))
        print(f"{m:<18}{k:>4}  {mu[0]:.4f}±{sd[0]:.4f}   "
              f"{mu[1]:.4f}±{sd[1]:.4f}   {mu[3]:.4f}±{sd[3]:.4f}")

    with open("cnn_results_mean_std.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "k", "ACC_mean", "NMI_mean", "NMIgeo_mean", "CEN_mean",
                    "ACC_std", "NMI_std", "NMIgeo_std", "CEN_std", "n_seeds"])
        for m, k, mu, sd, n in rows:
            w.writerow([m, k] + [f"{x:.6f}" for x in mu] + [f"{x:.6f}" for x in sd] + [n])

    with open("cnn_results_all_seeds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "method", "k", "ACC", "NMI_arith", "NMI_geo", "CEN"])
        for seed, results in zip(seeds, all_results):
            for r in results:
                w.writerow([seed, r[0], r[1]] + [f"{x:.6f}" for x in r[2:]])

    rg = defaultdict(list)
    for robust in all_robust:
        for name, ao, aa, ak in robust:
            rg[name].append((ao, aa, ak))
    print("\n鲁棒性 mean±std：")
    print(f"{'测试集':<12}{'CNN-orig':>18}{'CNN-aug':>18}{'kNN最佳':>18}")
    with open("cnn_robust_mean_std.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["testset", "CNN_orig_mean", "CNN_aug_mean", "kNN_mean",
                    "CNN_orig_std", "CNN_aug_std", "kNN_std"])
        for name, vals in rg.items():
            a = np.asarray(vals, float)
            mu = a.mean(0)
            sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1])
            print(f"{name:<12}" + "".join(f"{m:.4f}±{s:.4f}".rjust(18)
                                          for m, s in zip(mu, sd)))
            w.writerow([name] + [f"{x:.6f}" for x in mu] + [f"{x:.6f}" for x in sd])

if __name__ == "__main__":
    args = parse_args()
    log_f = open(args.log, "w", encoding="utf-8")   
    sys.stdout = Tee(sys.__stdout__, log_f)         

    all_results, all_robust = [], []
    for s in args.seeds:
        results, robust = run_once(s, args)
        all_results.append(results)
        all_robust.append(robust)

    summarize(all_results, all_robust, args.seeds)
    log_f.close()
    print(f"\n完成：完整输出在 {args.log}；汇总 cnn_results_mean_std.csv / "
          f"cnn_robust_mean_std.csv；明细 cnn_results_all_seeds.csv")

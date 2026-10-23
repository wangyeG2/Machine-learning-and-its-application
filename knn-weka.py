import os, math, sys, csv, glob
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "./att_faces"
IMG_H, IMG_W = 112, 92
N_CLASSES = 40
N_FEATURES = IMG_H * IMG_W
K_LIST = [5, 7, 9, 11, 13]
N_SPLITS = 10
SEED = 1

# ---------- 数据加载 ----------
def load_data(data_dir=DATA_DIR):
    X, y = [], []
    for i in range(1, N_CLASSES + 1):
        files = sorted(glob.glob(os.path.join(data_dir, f"s{i}", "*.pgm")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        for f in files:
            X.append(np.asarray(Image.open(f), dtype=np.float64).ravel() / 255.0)
            y.append(i - 1)
    return np.asarray(X), np.asarray(y, dtype=np.int64)

# ---------- 手写 kNN 核心 ----------
def pairwise_sqdist(X):
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return d2

def pairwise_sqdist_ab(A, B):
    sa = np.einsum("ij,ij->i", A, A)
    sb = np.einsum("ij,ij->i", B, B)
    d2 = sa[:, None] + sb[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2

def topk_neighbor_idx(sq_dist, k_max):
    d = sq_dist.copy(); np.fill_diagonal(d, np.inf)   # LOO: 排除自身
    return np.argsort(d, axis=1, kind="stable")[:, :k_max]

def majority_vote(neighbor_idx, y, k, n_classes=N_CLASSES):
    n = neighbor_idx.shape[0]
    labels = y[neighbor_idx[:, :k]]
    flat = labels + np.arange(n)[:, None] * n_classes
    counts = np.bincount(flat.ravel(), minlength=n * n_classes).reshape(n, n_classes)
    return counts.argmax(axis=1).astype(np.int64)     # 平票取最小标签

def confusion(y_true, y_pred, n_classes=N_CLASSES):
    cm = np.zeros((n_classes, n_classes), np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def knn_loo(X, y, k_list=K_LIST, n_classes=N_CLASSES):
    nb = topk_neighbor_idx(pairwise_sqdist(X), max(k_list))
    acc, cm = {}, {}
    for k in k_list:
        pred = majority_vote(nb, y, k, n_classes)
        acc[k] = float((pred == y).mean())
        cm[k] = confusion(y, pred, n_classes)
    return acc, cm

def knn_cv(X, y, k_list=K_LIST, n_classes=N_CLASSES, n_splits=N_SPLITS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cm = {k: np.zeros((n_classes, n_classes), np.int64) for k in k_list}
    pred_all = {k: np.empty(len(y), np.int64) for k in k_list}
    for tr, te in skf.split(X, y):
        nb = np.argsort(pairwise_sqdist_ab(X[te], X[tr]), axis=1, kind="stable")[:, :max(k_list)]
        for k in k_list:
            pred = majority_vote(nb, y[tr], k, n_classes)
            pred_all[k][te] = pred
            for t, p in zip(y[te], pred):
                cm[k][t, p] += 1
    acc = {k: float(np.trace(m) / m.sum()) for k, m in cm.items()}
    return acc, cm, pred_all

# ---------- ARFF 导出 ----------
def to_arff(X, y, path="orl.arff"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("@relation orl\n\n")
        for i in range(N_FEATURES):
            f.write(f"@attribute p{i} numeric\n")
        f.write("@attribute class {" + ",".join(map(str, range(N_CLASSES))) + "}\n\n@data\n")
        for xi, yi in zip(X, y):
            f.write(",".join(f"{v:.4f}" for v in xi) + f",{yi}\n")

# ---------- 指标：NMI / CEN ----------
def nmi_from_cm(cm, average="arithmetic"):
    cm = np.asarray(cm, dtype=np.float64); n = cm.sum()
    pxy = cm / n
    px, py = pxy.sum(1), pxy.sum(0)
    mask = pxy > 0
    mi = float((pxy[mask] * np.log2(pxy[mask] / np.outer(px, py)[mask])).sum())
    def ent(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
    hx, hy = ent(px), ent(py)
    denom = {"arithmetic": (hx + hy) / 2.0, "geometric": math.sqrt(hx * hy),
             "min": min(hx, hy), "max": max(hx, hy)}[average]
    return mi / denom if denom > 0 else 0.0

def cen_from_cm(cm, modified=False):
    C = np.asarray(cm, dtype=np.float64)
    N = C.shape[0]; S = C.sum()
    D = C.sum(1) + C.sum(0)
    base = 2.0 * (N - 1)
    if not modified:
        Dj, w = D, D / (2.0 * S)
    else:
        Dj = D - np.diag(C)
        w = Dj / (2.0 * S - np.trace(C))
    total = 0.0
    for j in range(N):
        if Dj[j] <= 0:
            continue
        h = 0.0
        for k in range(N):
            if k == j:
                continue
            for c in (C[j, k], C[k, j]):
                if c > 0:
                    p = c / Dj[j]
                    h -= p * math.log(p, base)
        total += w[j] * h
    return float(total)

# ---------- Weka 结果解析 ----------
def parse_weka_confusion(txt_path):
    rows = []
    for line in open(txt_path, encoding="utf-8", errors="ignore"):
        if "|" not in line:
            continue
        nums = [int(t) for t in line.split("|")[0].split()]
        if nums:
            rows.append(nums)
        if len(rows) == len(rows[0]):
            break
    return np.array(rows, dtype=np.int64)

def analyze_weka(txt_path):
    cm = parse_weka_confusion(txt_path)
    n = cm.sum()
    print(f"[Weka] {txt_path}: N={n}, ACC={np.trace(cm)/n:.4f}, "
          f"NMI={nmi_from_cm(cm):.4f}, CEN={cen_from_cm(cm):.4f}")

# ---------- 主程序 ----------
if __name__ == "__main__":
    X, y = load_data(DATA_DIR)
    to_arff(X, y)

    acc_loo, cm_loo = knn_loo(X, y)
    acc_cv, cm_cv, _ = knn_cv(X, y)

    print("\nk\t| LOO: ACC\tNMI\tCEN\t| CV10: ACC\tNMI\tCEN")
    print("-" * 78)
    for k in K_LIST:
        print(f"{k}\t| {acc_loo[k]:.4f}\t{nmi_from_cm(cm_loo[k]):.4f}"
              f"\t{cen_from_cm(cm_loo[k]):.4f}"
              f"\t| {acc_cv[k]:.4f}\t{nmi_from_cm(cm_cv[k]):.4f}"
              f"\t{cen_from_cm(cm_cv[k]):.4f}")

    with open("orl_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "k", "ACC", "NMI_arith", "NMI_geo", "CEN"])
        for mode, accd, cmd in (("LOO", acc_loo, cm_loo), ("CV10", acc_cv, cm_cv)):
            for k in K_LIST:
                w.writerow([mode, k, f"{accd[k]:.6f}",
                            f"{nmi_from_cm(cmd[k]):.6f}",
                            f"{nmi_from_cm(cmd[k], 'geometric'):.6f}",
                            f"{cen_from_cm(cmd[k]):.6f}"])
    for k in K_LIST:
        np.savetxt(f"orl_cm_cv_k{k}.csv", cm_cv[k], fmt="%d", delimiter=",")

    for p in sys.argv[1:]:
        analyze_weka(p)

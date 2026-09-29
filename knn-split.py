# knn_split.py —— knn-weka.py 改造版：数据划分变为与 cnn.py 一致的固定 训练集+测试集
#
# [改动1] 划分: 原 StratifiedKFold 10 折 CV（及 LOO）→ 固定划分：每类前 5 张训练、后 5 张测试；
#         make_split 与 cnn.py 逐行一致，并通过 orl_split.npz 与 cnn.py 互相校验，
#         保证 kNN 与 CNN 用的是同一份训练/测试数据（同实验配置对比的前提）。
# [改动2] ARFF: --arff 时分别导出 orl_train.arff / orl_test.arff，
#         Weka 中用 train 建模、test 评测，与手写 kNN 同协议。
# [保留]  kNN 算法、ACC、NMI、CEN、Weka 混淆矩阵解析 —— 与 knn-weka.py 逐行一致。
#
# 用法:
#   python knn_split.py                          # 手写 kNN（固定划分）ACC/NMI/CEN
#   python knn_split.py --arff                   # 另导出 Weka 用 ARFF
#   python knn_split.py weka_k5.txt weka_k9.txt  # 追加解析 Weka 输出（IBk 固定划分的结果）

import os, math, csv, argparse, glob
import numpy as np
from PIL import Image

# ============ 配置区（与 knn-weka.py / cnn.py 一致） ============
DATA_DIR   = "./att_faces"
IMG_H, IMG_W = 112, 92
N_CLASSES  = 40
N_FEATURES = IMG_H * IMG_W
TRAIN_PER_CLASS = 5                  # [改动1] 每类训练张数（与 cnn.py 一致）
K_LIST     = [5, 7, 9, 11, 13]
SPLIT_FILE = "orl_split.npz"

def load_data(data_dir=DATA_DIR):
    """遍历 s{1..40}/{1..10}.pgm，展平 + 归一化到 [0,1]。标签: s1→0 ... s40→39（与 knn-weka.py 一致）"""
    if not os.path.isdir(data_dir):
        raise SystemExit(f"找不到数据目录: {data_dir}\n当前工作目录: {os.getcwd()}")
    X, y = [], []
    for i in range(1, N_CLASSES + 1):
        d = os.path.join(data_dir, f"s{i}")
        files = sorted(glob.glob(os.path.join(d, "*.pgm")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        assert len(files) == 10, f"s{i} 下应有 10 张 PGM，实际 {len(files)}"
        for f in files:
            img = np.asarray(Image.open(f), dtype=np.float64) / 255.0
            assert img.shape == (IMG_H, IMG_W), f"尺寸异常 {f}: {img.shape}"
            X.append(img.ravel()); y.append(i - 1)
    X, y = np.asarray(X), np.asarray(y, dtype=np.int64)
    assert X.shape == (N_CLASSES * 10, N_FEATURES), f"总数异常: {X.shape}"
    return X, y

# ---------- [改动1] 与 cnn.py 完全一致的固定划分 ----------
def make_split(y, train_per_class=TRAIN_PER_CLASS):
    tr, te = [], []
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        assert len(idx) == 10
        tr += idx[:train_per_class].tolist()
        te += idx[train_per_class:].tolist()
    return np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)

def sync_split(tr_idx, te_idx, path=SPLIT_FILE):
    if os.path.exists(path):
        z = np.load(path)
        assert np.array_equal(z["train_idx"], tr_idx) and np.array_equal(z["test_idx"], te_idx), \
            f"{path} 与当前划分不一致！请删除后重跑 cnn.py / knn_split.py"
        print(f"[划分] 已校验 {path} 与 cnn.py 一致")
    else:
        np.savez(path, train_idx=tr_idx, test_idx=te_idx)
        print(f"[划分] 已写出 {path}")

# ---------- 核心算法（与 knn-weka.py 完全一致，未改动） ----------
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
    return counts.argmax(axis=1).astype(np.int64)   # 平票取最小标签（与 Weka 一致）

def confusion(y_true, y_pred, n_classes=N_CLASSES):
    cm = np.zeros((n_classes, n_classes), np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def knn_split_eval(X_train, y_train, X_test, y_test, k_list=K_LIST, n_classes=N_CLASSES):
    """[改动1] 原 10 折 CV → 固定训练/测试上的 kNN（算法与 knn-weka.py 完全一致）"""
    nb = np.argsort(pairwise_sqdist_ab(X_test, X_train), axis=1, kind="stable")[:, :max(k_list)]
    acc, cm, preds = {}, {}, {}
    for k in k_list:
        preds[k] = majority_vote(nb, y_train, k, n_classes)
        cm[k] = confusion(y_test, preds[k], n_classes)
        acc[k] = float(np.trace(cm[k]) / cm[k].sum())
    return acc, cm, preds

# ---------- ARFF 导出（[改动2] 训练/测试分别导出；类别枚举 0..39） ----------
def to_arff(X, y, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("@relation orl\n\n")
        for i in range(N_FEATURES):
            f.write(f"@attribute p{i} numeric\n")
        f.write("@attribute class {" + ",".join(map(str, range(N_CLASSES))) + "}\n\n@data\n")
        for xi, yi in zip(X, y):
            f.write(",".join(f"{v:.4f}" for v in xi) + f",{yi}\n")

# ---------- 指标（NMI / CEN，与 knn-weka.py 未改动，40 类自动生效） ----------
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
    base = 2.0 * (N - 1)                 # 40 类时底数为 78
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

# ---------- Weka 结果解析（与 knn-weka.py 未改动） ----------
def parse_weka_confusion(txt_path):
    rows = []
    for line in open(txt_path, encoding="utf-8", errors="ignore"):
        if "|" not in line: continue
        nums = [int(t) for t in line.split("|")[0].split()]
        if nums: rows.append(nums)
        if len(rows) == len(rows[0]): break
    cm = np.array(rows, dtype=np.int64)
    assert cm.shape == (N_CLASSES, N_CLASSES), "解析出的混淆矩阵不是 40×40"
    return cm

def analyze_weka(txt_path):
    cm = parse_weka_confusion(txt_path)
    n = cm.sum()
    print(f"[Weka] {txt_path}: N={n}, ACC={np.trace(cm)/n:.4f}, "
          f"NMI={nmi_from_cm(cm):.4f}, CEN={cen_from_cm(cm):.4f}")

# ---------- 主程序 ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="knn-weka.py 改造版：固定训练/测试划分（与 cnn.py 一致）")
    ap.add_argument("--arff", action="store_true", help="导出 orl_train.arff / orl_test.arff 供 Weka")
    ap.add_argument("weka", nargs="*", help="Weka 输出 txt（可选，追加解析其混淆矩阵）")
    args = ap.parse_args()

    X, y = load_data(DATA_DIR)
    print(f"Loaded ORL: X={X.shape}, y classes={np.unique(y).size}")

    tr_idx, te_idx = make_split(y)           # [改动1] 与 cnn.py 完全一致的固定划分
    sync_split(tr_idx, te_idx)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_test,  y_test  = X[te_idx], y[te_idx]
    print(f"划分: train={len(tr_idx)}（每类 {TRAIN_PER_CLASS}）, test={len(te_idx)} —— 与 cnn.py 同一份数据")

    acc, cm, preds = knn_split_eval(X_train, y_train, X_test, y_test)

    # 与 sklearn 交叉核验 NMI 实现正确性（与 knn-weka.py 的做法一致）
    try:
        from sklearn.metrics import normalized_mutual_info_score
        for k in K_LIST:
            ref = normalized_mutual_info_score(y_test, preds[k], average_method="arithmetic")
            assert abs(ref - nmi_from_cm(cm[k])) < 1e-9, f"NMI 核验失败 k={k}"
        print("[核验] NMI 实现与 sklearn 一致")
    except ImportError:
        pass

    print("\nk\t| ACC\tNMI\tCEN   （固定划分: 每类 5 训练 / 5 测试，与 cnn.py 一致）")
    print("-" * 60)
    for k in K_LIST:
        print(f"{k}\t| {acc[k]:.4f}\t{nmi_from_cm(cm[k]):.4f}\t{cen_from_cm(cm[k]):.4f}")

    with open("knn_split_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "k", "ACC", "NMI_arith", "NMI_geo", "CEN"])
        for k in K_LIST:
            w.writerow(["SPLIT", k, f"{acc[k]:.6f}", f"{nmi_from_cm(cm[k]):.6f}",
                        f"{nmi_from_cm(cm[k], 'geometric'):.6f}", f"{cen_from_cm(cm[k]):.6f}"])
    for k in K_LIST:
        np.savetxt(f"knn_cm_test_k{k}.csv", cm[k], fmt="%d", delimiter=",")
    print("\n已导出 knn_split_results.csv 与 knn_cm_test_k*.csv")

    if args.arff:
        to_arff(X_train, y_train, "orl_train.arff")
        to_arff(X_test,  y_test,  "orl_test.arff")
        print("已导出 orl_train.arff / orl_test.arff（Weka: 用 train 建模、test 评测，与手写 kNN 同协议）")

    for p in args.weka:
        analyze_weka(p)

# knn-weka.py —— ORL 数据集：手写 kNN vs Weka vs sklearn 三方对比（ACC / NMI / CEN）
# 用法:
#   python knn-weka.py                    # 完整流程：手写 LOO/CV10 + sklearn 仲裁 + 导出 orl.arff
#   python knn-weka.py orl_weka_k5.txt orl_weka_k7.txt ...
#                                         # 仅解析 Weka 结果文本，计算其 NMI/CEN（几秒完成）

import os, math, sys, csv, glob, time
import numpy as np
from PIL import Image                                   # pip install pillow
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score

# ============ 配置区 ============
DATA_DIR   = "./att_faces"      # att_faces 下是 s1..s40，每个文件夹 1.pgm..10.pgm
IMG_H, IMG_W = 112, 92          # ORL 原始尺寸 → N_FEATURES = 10304
N_CLASSES  = 40
K_LIST     = [5, 7, 9, 11, 13]
N_SPLITS   = 10
SEED       = 1                  # 与 Weka 默认 seed 对齐
# =================================
N_FEATURES = IMG_H * IMG_W

# ---------- 数据加载 ----------
def load_data(data_dir=DATA_DIR):
    """遍历 s{1..40}/{1..10}.pgm，展平 + 归一化到 [0,1]。标签: s1→0 ... s40→39"""
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
            X.append(img.ravel())
            y.append(i - 1)
    X, y = np.asarray(X), np.asarray(y, dtype=np.int64)
    assert X.shape == (N_CLASSES * 10, N_FEATURES), f"总数异常: {X.shape}"
    return X, y

# ---------- 手写 kNN 核心 ----------
def knn_loo_naive(X, y, k_list):
    """朴素版，仅用于正确性对照。"""
    n = len(X)
    preds = {k: np.empty(n, dtype=np.int64) for k in k_list}
    for k in k_list:
        for i in range(n):
            cand = sorted(((math.dist(X[i], X[j]), y[j]) for j in range(n) if j != i),
                          key=lambda t: t[0])
            votes = {}
            for _, lb in cand[:k]:
                votes[lb] = votes.get(lb, 0) + 1
            preds[k][i] = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return preds

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
    d = sq_dist.copy(); np.fill_diagonal(d, np.inf)      # LOO: 排除自身
    return np.argsort(d, axis=1, kind="stable")[:, :k_max]

def majority_vote(neighbor_idx, y, k, n_classes=N_CLASSES):
    n = neighbor_idx.shape[0]
    labels = y[neighbor_idx[:, :k]]
    flat = labels + np.arange(n)[:, None] * n_classes
    counts = np.bincount(flat.ravel(), minlength=n * n_classes).reshape(n, n_classes)
    return counts.argmax(axis=1).astype(np.int64)        # 平票取最小标签（与 Weka 一致）

def confusion(y_true, y_pred, n_classes=N_CLASSES):
    cm = np.zeros((n_classes, n_classes), np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def knn_loo(X, y, k_list=K_LIST, n_classes=N_CLASSES):
    """留一法（初级要求）。返回各 k 的 (ACC, 混淆矩阵, 预测向量)。"""
    nb = topk_neighbor_idx(pairwise_sqdist(X), max(k_list))
    acc, cm, preds = {}, {}, {}
    for k in k_list:
        p = majority_vote(nb, y, k, n_classes)
        preds[k] = p
        acc[k] = float((p == y).mean())
        cm[k] = confusion(y, p, n_classes)
    return acc, cm, preds

def knn_cv(X, y, k_list=K_LIST, n_classes=N_CLASSES, n_splits=N_SPLITS,
           seed=SEED, folds=None):
    """十折分层 CV（中级要求，与 Weka 同协议）。folds 可传入外部共用折划分。"""
    if folds is None:
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed).split(X, y))
    cm = {k: np.zeros((n_classes, n_classes), np.int64) for k in k_list}
    pred_all = {k: np.empty(len(y), np.int64) for k in k_list}
    for tr, te in folds:
        nb = np.argsort(pairwise_sqdist_ab(X[te], X[tr]),
                        axis=1, kind="stable")[:, :max(k_list)]
        for k in k_list:
            pred = majority_vote(nb, y[tr], k, n_classes)
            pred_all[k][te] = pred
            for t, p in zip(y[te], pred):
                cm[k][t, p] += 1
    acc = {k: float(np.trace(m) / m.sum()) for k, m in cm.items()}
    return acc, cm, pred_all

# ---------- sklearn 仲裁（参数严格对齐） ----------
def _make_knn(k):
    """严格对齐的 sklearn 分类器：等权投票、暴力搜索、欧氏距离。
    algorithm='brute' 必须显式指定，防止 auto 选树结构导致平距处理差异。"""
    return KNeighborsClassifier(n_neighbors=k, weights="uniform",
                                algorithm="brute", metric="euclidean")

def knn_cv_sklearn(X, y, k_list=K_LIST, n_classes=N_CLASSES, folds=None):
    """与手写 knn_cv 同协议（共用 folds）；sklearn 不做特征缩放，与手写 ÷255 一致。"""
    if folds is None:
        folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                     random_state=SEED).split(X, y))
    cm = {k: np.zeros((n_classes, n_classes), np.int64) for k in k_list}
    pred_all = {k: np.empty(len(y), np.int64) for k in k_list}
    for tr, te in folds:
        for k in k_list:
            pred = _make_knn(k).fit(X[tr], y[tr]).predict(X[te])
            pred_all[k][te] = pred
            for t, p in zip(y[te], pred):
                cm[k][t, p] += 1
    acc = {k: float(np.trace(m) / m.sum()) for k, m in cm.items()}
    return acc, cm, pred_all

def knn_loo_sklearn(X, y, k_list=K_LIST, n_classes=N_CLASSES):
    """sklearn 留一法（LOO 无随机性，与手写 LOO 天然同协议）。"""
    loo = list(LeaveOneOut().split(X, y))
    acc, cm, pred_all = {}, {}, {}
    for k in k_list:
        pred = cross_val_predict(_make_knn(k), X, y, cv=loo)
        pred_all[k] = pred
        acc[k] = float((pred == y).mean())
        cm[k] = confusion(y, pred, n_classes)
    return acc, cm, pred_all

def verify(X, y, n_check=150, k_list=K_LIST, seed=0, n_classes=N_CLASSES):
    """随机子集上用朴素版核对高速版，不一致则终止。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_check, len(X)), replace=False)
    Xs, ys = X[idx], y[idx]
    preds_naive = knn_loo_naive(Xs, ys, k_list)
    nb = topk_neighbor_idx(pairwise_sqdist(Xs), max(k_list))
    for k in k_list:
        if not np.array_equal(majority_vote(nb, ys, k, n_classes), preds_naive[k]):
            raise SystemExit("高速版与朴素版结果不一致，终止！")

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
    """由混淆矩阵计算归一化互信息。average: arithmetic(=sklearn默认)/geometric/min/max。"""
    cm = np.asarray(cm, dtype=np.float64); n = cm.sum()
    pxy = cm / n
    px, py = pxy.sum(1), pxy.sum(0)
    mask = pxy > 0
    mi = float((pxy[mask] * np.log2(pxy[mask] / np.outer(px, py)[mask])).sum())
    def ent(p):
        p = p[p > 0]; return float(-(p * np.log2(p)).sum())
    hx, hy = ent(px), ent(py)
    denom = {"arithmetic": (hx + hy) / 2.0, "geometric": math.sqrt(hx * hy),
             "min": min(hx, hy), "max": max(hx, hy)}[average]
    return mi / denom if denom > 0 else 0.0

def cen_from_cm(cm, modified=False):
    """混淆熵 CEN（Wang et al. 2008 原始定义），越小越好，40 类时取值 [0,1]。
    原始版分母 D_j = 行和 + 列和（对角元计两次）；modified=True 切换 MCEN。"""
    C = np.asarray(cm, dtype=np.float64)
    N = C.shape[0]; S = C.sum()
    D = C.sum(1) + C.sum(0)
    base = 2.0 * (N - 1)                       # 40 类时底数为 78
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
    """提取 Weka 结果文本中的 40×40 混淆矩阵。只接受每行恰好 40 个整数的行。"""
    rows = []
    for line in open(txt_path, encoding="utf-8", errors="ignore"):
        if "|" not in line:
            continue
        nums = [int(t) for t in line.split("|")[0].split()]
        if len(nums) == N_CLASSES:
            rows.append(nums)
            if len(rows) == N_CLASSES:
                break
    cm = np.array(rows, dtype=np.int64)
    assert cm.shape == (N_CLASSES, N_CLASSES), \
        f"解析失败：得到 {cm.shape}，请检查 txt 是否完整（需含 40 行混淆矩阵）"
    return cm

def analyze_weka(txt_path, writer=None):
    cm = parse_weka_confusion(txt_path)
    n = cm.sum()
    acc, nmi, cen = np.trace(cm) / n, nmi_from_cm(cm), cen_from_cm(cm)
    print(f"[Weka] {txt_path}: N={n}, ACC={acc:.4f}, NMI={nmi:.4f}, CEN={cen:.4f}")
    if writer is not None:
        writer.writerow([txt_path, f"{acc:.6f}", f"{nmi:.6f}", f"{cen:.6f}"])
    return cm

# ---------- CSV 导出（被 Excel 占用时自动改名） ----------
def write_csv(rows, name="orl_results.csv"):
    try:
        with open(name, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"已导出 {name}")
    except PermissionError:
        alt = f"orl_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(alt, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"⚠ {name} 被占用（多半开着 Excel），已改存为 {alt}")

# ---------- 主程序 ----------
if __name__ == "__main__":
    # 分支一：带参数 → 仅解析 Weka 结果文本，跳过全流程
    if len(sys.argv) > 1:
        with open("weka_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["file", "ACC", "NMI", "CEN"])
            for p in sys.argv[1:]:
                analyze_weka(p, w)
        print("已导出 weka_metrics.csv")
        sys.exit(0)

    # 分支二：无参数 → 完整三方实验
    X, y = load_data(DATA_DIR)
    print(f"Loaded ORL: X={X.shape}, y classes={np.unique(y).size}")
    verify(X, y)                       # 朴素版自检
    to_arff(X, y)                      # 生成 orl.arff 给 Weka

    # ★ 折划分只生成一次，手写与 sklearn 共用（仲裁前提）；Weka 同协议不同实现
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=SEED).split(X, y))

    acc_loo, cm_loo, pred_loo   = knn_loo(X, y)
    acc_cv,  cm_cv,  pred_cv    = knn_cv(X, y, folds=folds)
    acc_sk,  cm_sk,  pred_sk    = knn_cv_sklearn(X, y, folds=folds)
    acc_sloo, cm_sloo, pred_sloo = knn_loo_sklearn(X, y)

    # NMI 公式仲裁：sklearn 官方实现 vs 自算（三种预测全部核验）
    for pr, cmx in ((pred_loo, cm_loo), (pred_cv, cm_cv), (pred_sk, cm_sk),
                    (pred_sloo, cm_sloo)):
        for k in K_LIST:
            ref = normalized_mutual_info_score(y, pr[k], average_method="arithmetic")
            assert abs(ref - nmi_from_cm(cmx[k])) < 1e-9, f"NMI 核验失败 k={k}"

    # ★ 逐样本仲裁：同一折划分下 sklearn 与手写预测应完全一致
    print("\n[仲裁] sklearn vs 手写 CV10（共用 folds）")
    for k in K_LIST:
        agree = float((pred_cv[k] == pred_sk[k]).mean())
        print(f"  k={k}: 逐样本一致率={agree:.4f}  "
              f"dACC={acc_sk[k]-acc_cv[k]:+.4f}  "
              f"dNMI={nmi_from_cm(cm_sk[k])-nmi_from_cm(cm_cv[k]):+.4f}  "
              f"dCEN={cen_from_cm(cm_sk[k])-cen_from_cm(cm_cv[k]):+.4f}")

    print("\nk\t| LOO: ACC\tNMI\tCEN\t| CV10: ACC\tNMI\tCEN\t| SK10: ACC\tNMI\tCEN")
    print("-" * 112)
    for k in K_LIST:
        print(f"{k}\t| {acc_loo[k]:.4f}\t{nmi_from_cm(cm_loo[k]):.4f}"
              f"\t{cen_from_cm(cm_loo[k]):.4f}"
              f"\t| {acc_cv[k]:.4f}\t{nmi_from_cm(cm_cv[k]):.4f}"
              f"\t{cen_from_cm(cm_cv[k]):.4f}"
              f"\t| {acc_sk[k]:.4f}\t{nmi_from_cm(cm_sk[k]):.4f}"
              f"\t{cen_from_cm(cm_sk[k]):.4f}")

    # 导出三方结果（SK10 为 sklearn 仲裁行）
    rows = [["mode", "k", "ACC", "NMI_arith", "NMI_geo", "CEN"]]
    for mode, accd, cmd in (("LOO", acc_loo, cm_loo),
                            ("CV10", acc_cv, cm_cv),
                            ("SK10", acc_sk, cm_sk),
                            ("SK_LOO", acc_sloo, cm_sloo)):
        for k in K_LIST:
            rows.append([mode, k, f"{accd[k]:.6f}",
                         f"{nmi_from_cm(cmd[k]):.6f}",
                         f"{nmi_from_cm(cmd[k], 'geometric'):.6f}",
                         f"{cen_from_cm(cmd[k]):.6f}"])
    write_csv(rows, "orl_results.csv")

    for k in K_LIST:
        np.savetxt(f"orl_cm_cv_k{k}.csv", cm_cv[k], fmt="%d", delimiter=",")
    print("已导出 orl_cm_cv_k*.csv")

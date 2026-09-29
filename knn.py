import math
import numpy as np

DATA_PATH = "semeion.data"
N_FEATURES = 256      
N_CLASSES = 10
K_LIST = [5,7,9,11,13]   


def load_data(path):  #数据加载
    data = np.loadtxt(path)
    X = data[:, :N_FEATURES].astype(np.float64)
    y = data[:, N_FEATURES:].argmax(axis=1).astype(np.int64)
    return X, y


def knn_loo_naive(X, y, k_list): # 朴素留一法K近邻
    n = len(X)
    preds = {k: np.empty(n, dtype=np.int64) for k in k_list}
    for k in k_list:
        for i in range(n):
            cand = sorted(((math.dist(X[i], X[j]), y[j])
                           for j in range(n) if j != i), key=lambda t: t[0])
            votes = {}
            for _, label in cand[:k]:
                votes[label] = votes.get(label, 0) + 1
            preds[k][i] = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return preds


def pairwise_sqdist(X): #计算样本间的平方欧氏距离
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def topk_neighbor_idx(sq_dist, k_max): # 获取k个最近邻的索引
    d = sq_dist.copy()
    np.fill_diagonal(d, np.inf)   
    return np.argsort(d, axis=1, kind="stable")[:, :k_max]


def majority_vote(neighbor_idx, y, k, n_classes): # 多数投票
    n = neighbor_idx.shape[0]
    labels = y[neighbor_idx[:, :k]]
    flat = labels + np.arange(n)[:, None] * n_classes
    counts = np.bincount(flat.ravel(), minlength=n * n_classes).reshape(n, n_classes)
    return counts.argmax(axis=1).astype(np.int64)   # 平票取最小标签


def knn_loo(X, y, k_list=K_LIST, n_classes=N_CLASSES): # 留一法K近邻
    neighbor_idx = topk_neighbor_idx(pairwise_sqdist(X), max(k_list))
    acc = {}
    for k in k_list:
        preds = majority_vote(neighbor_idx, y, k, n_classes)
        acc[k] = float((preds == y).mean())
    return acc


def verify(X, y, n_check=150, k_list=K_LIST, seed=0, n_classes=N_CLASSES): #验证函数
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_check, len(X)), replace=False)
    Xs, ys = X[idx], y[idx]
    preds_naive = knn_loo_naive(Xs, ys, k_list)
    neighbor_idx = topk_neighbor_idx(pairwise_sqdist(Xs), max(k_list))
    for k in k_list:
        if not np.array_equal(majority_vote(neighbor_idx, ys, k, n_classes),
                              preds_naive[k]):
            raise SystemExit("高速版与朴素版结果不一致，终止！")


if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    verify(X, y)                    
    acc = knn_loo(X, y)             
    print("k值\t识别精度")
    for k in K_LIST:
        print(f"{k}\t{acc[k]:.4f}")

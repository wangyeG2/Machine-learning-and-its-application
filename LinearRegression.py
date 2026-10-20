import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.seterr(over='ignore', invalid='ignore')

DATA_FILE = 'winequality-white.csv'
SEED = 42
BGD_LRS = [0.01, 0.05, 0.1, 0.5]                 # BGD 候选学习率
SGD_LRS = [0.001, 0.005, 0.01, 0.05, 0.1]        # SGD 候选学习率
SGD_EPOCHS = 100   
N_EPOCHS = 300     # BGD 和 SGD 的公平对比轮数
LAMS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]        # 岭回归 λ 扫描网格
LAMS_PATH = np.logspace(-4, 1, 40)               # 系数路径的 λ 网格
DIVERGE_TOL = 10.0                               # MSE 超过初始值 10 倍视为发散


def mse(X, y, th):
    v = float(np.mean((X @ th - y) ** 2))
    return v if np.isfinite(v) else np.inf

def bgd(X, y, Xte, yte, lr, n_epochs=N_EPOCHS):
    th = np.zeros(X.shape[1])
    h_tr, h_te = [mse(X, y, th)], [mse(Xte, yte, th)]
    for _ in range(n_epochs):
        th -= lr * (2.0 / len(y)) * X.T @ (X @ th - y)
        v = mse(X, y, th)
        if not np.isfinite(v):                          # 发散
            h_tr.append(np.inf); h_te.append(np.inf)
            break
        h_tr.append(v); h_te.append(mse(Xte, yte, th))
    return th, h_tr, h_te


def sgd(X, y, Xte, yte, lr, n_epochs=SGD_EPOCHS, record='epoch'):
    th = np.zeros(X.shape[1])
    h_tr, h_te = [mse(X, y, th)], [mse(Xte, yte, th)]
    rng = np.random.default_rng(SEED)
    for _ in range(n_epochs):
        for i in rng.permutation(len(y)):
            xi, yi = X[i], y[i]
            th -= lr * 2.0 * xi * (xi @ th - yi)
            if record == 'update':
                h_tr.append(mse(X, y, th)); h_te.append(mse(Xte, yte, th))
        if record == 'epoch':
            h_tr.append(mse(X, y, th)); h_te.append(mse(Xte, yte, th))
        if not np.isfinite(h_tr[-1]):                   # 发散
            break
    return th, h_tr, h_te


def ridge_analytic(X, y, lam):
    n = X.shape[1]
    I_reg = np.eye(n); I_reg[0, 0] = 0.0                # 偏置项不参与惩罚
    return np.linalg.solve(X.T @ X + X.shape[0] * lam * I_reg, X.T @ y)


def ridge_bgd(X, y, lam, lr, n_epochs=10000, tol=1e-12):
    th = np.zeros(X.shape[1])
    reg = np.ones_like(th); reg[0] = 0.0                # 偏置项不参与惩罚
    for _ in range(n_epochs):
        delta = lr * ((2.0 / len(y)) * X.T @ (X @ th - y) + 2.0 * lam * reg * th)
        th -= delta
        if np.max(np.abs(delta)) < tol:
            break
    return th


def ridge_report(name, th, Xa, ya, Xb, yb):
    tr = float(np.mean((Xa @ th - ya) ** 2))
    te = float(np.mean((Xb @ th - yb) ** 2))
    print(f'{name:<30} 训练MSE={tr:.4f} 测试MSE={te:.4f}')
    return tr, te


def analyze_lr(name, res):
    print(f'\n===== {name} 不同学习率分析（验证集协议） =====')
    print(f'{"lr":>8} | {"最终训练MSE":>12} | {"最终验证MSE":>12} | {"最低验证MSE":>12} | 结果')
    best_lr, best_fin = None, np.inf
    for lr, (h_tr, h_va) in res.items():
        va = np.asarray(h_va, float)
        if not np.isfinite(va[-1]) or va[-1] > DIVERGE_TOL * va[0]:
            print(f'{lr:>8} | {"—":>12} | {"—":>12} | {"—":>12} | 发散')
            continue
        print(f'{lr:>8} | {float(h_tr[-1]):>12.4f} | {va[-1]:>12.4f} | '
              f'{va.min():>12.4f} | 收敛')
        if va[-1] < best_fin:
            best_lr, best_fin = lr, float(va[-1])
    print(f'>>> {name} 选定 lr={best_lr}（验证MSE最低 {best_fin:.4f}）')
    return best_lr

def first_reach(h, frac=1.05):
    a = np.asarray(h, float)
    hit = np.nonzero(a <= a[-1] * frac)[0]
    return int(hit[0]) if hit.size else None


def fit_eval(cols):
    A, B = Xtr[:, cols], Xte[:, cols]
    th = ridge_analytic(A, y_train, 0.0)
    return th, float(np.mean((A @ th - y_train) ** 2)), \
        float(np.mean((B @ th - y_test) ** 2))


# ================= 数据读取与预处理 =================
df = pd.read_csv(DATA_FILE, sep=';')
df_raw = df.copy()                                      
print(f'原始数据: {df.shape}')
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)        # 完全重复样本会造成训练/测试泄漏
print(f'剔除完全重复样本 {n_dup} 条，剩余 {len(df)} 条')

feat_names = [c for c in df.columns if c != 'quality']
X = df[feat_names].to_numpy(dtype=float)
y = df['quality'].to_numpy(dtype=float)

# 分层 4:1 划分，保持各评分占比一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
diff = (pd.Series(y_train).value_counts(normalize=True).sort_index()
        - pd.Series(y_test).value_counts(normalize=True).sort_index()).abs()
print(f'训练/测试各评分占比最大绝对差 = {diff.max():.4f}')
print(f'[修改6] 训练集 {len(y_train)} 条 / 测试集 {len(y_test)} 条')

# 不去重直接划分，量化重复样本泄漏的影响
Xr = df_raw[feat_names].to_numpy(dtype=float)
yr = df_raw['quality'].to_numpy(dtype=float)
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
    Xr, yr, test_size=0.2, stratify=yr, random_state=SEED)
sc_r = StandardScaler().fit(Xtr_r)
Ar = np.hstack([np.ones((len(Xtr_r), 1)), sc_r.transform(Xtr_r)])
Br = np.hstack([np.ones((len(Xte_r), 1)), sc_r.transform(Xte_r)])
th_r = ridge_analytic(Ar, ytr_r, 0.0)
tr_keys = set(map(tuple, Xtr_r))
leak = sum(tuple(row) in tr_keys for row in Xte_r)
print(f'[消融] 不去重: 测试MSE={float(np.mean((Br @ th_r - yte_r) ** 2)):.4f} '
      f'泄漏测试样本 {leak}/{len(yte_r)} ({leak / len(yte_r):.1%})')

Xs_tr, Xs_va, ys_tr, ys_va = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=0)
sc_lr = StandardScaler().fit(Xs_tr)                     # 标准化参数只来自子训练集
Xl_tr = np.hstack([np.ones((len(Xs_tr), 1)), sc_lr.transform(Xs_tr)])
Xl_va = np.hstack([np.ones((len(Xs_va), 1)), sc_lr.transform(Xs_va)])

# 主训练/测试标准化（参数仅由训练集确定），偏置并入第 0 列
scaler = StandardScaler().fit(X_train)
Xtr = np.hstack([np.ones((len(X_train), 1)), scaler.transform(X_train)])
Xte = np.hstack([np.ones((len(X_test), 1)), scaler.transform(X_test)])
th_ols = ridge_analytic(Xtr, y_train, 0.0)              # 精确 OLS 系数

# ================= 学习率扫描与选择（验证集协议） =================
bgd_res = {lr: bgd(Xl_tr, ys_tr, Xl_va, ys_va, lr)[1:] for lr in BGD_LRS}
sgd_res = {lr: sgd(Xl_tr, ys_tr, Xl_va, ys_va, lr)[1:] for lr in SGD_LRS}
lr_b = analyze_lr('BGD', bgd_res)
lr_s = analyze_lr('SGD', sgd_res)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, res, nm in [(axes[0], bgd_res, 'BGD'), (axes[1], sgd_res, 'SGD')]:
    for lr, (_, h_va) in res.items():                   
        ax.plot(h_va, label=f'{nm} lr={lr}')
    ax.set(title=f'{nm} 不同学习率（验证MSE）',          
           xlabel='epoch', ylabel='MSE')
    ax.set_yscale('log'); ax.set_ylim(0.3, 1e4)
    ax.legend(); ax.grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig('lr_compare.png', dpi=150); plt.show()

# ================= 用选定学习率在全训练集上训练最终模型 =================
th_b, tr_b, te_b = bgd(Xtr, y_train, Xte, y_test, lr=lr_b)
th_s, tr_s, te_s = sgd(Xtr, y_train, Xte, y_test, lr=lr_s)
print(f'\nBGD(lr={lr_b}): 训练MSE={tr_b[-1]:.4f} 测试MSE={te_b[-1]:.4f}')
print(f'SGD(lr={lr_s}): 训练MSE={tr_s[-1]:.4f} 测试MSE={te_s[-1]:.4f}')

print('\nBGD 最终模型系数:')
for nm, c in zip(['bias'] + feat_names, th_b):
    print(f'  {nm:<26}{c: .4f}')

# 收敛曲线
_, u_tr, u_te = sgd(Xtr, y_train, Xte, y_test, lr=lr_s, n_epochs=1, record='update')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.plot(tr_b, label='BGD 训练'); ax.plot(te_b, '--', label='BGD 测试')
ax.plot(tr_s, label='SGD 训练'); ax.plot(te_s, '--', label='SGD 测试')
ax.set(xlabel='epoch', ylabel='MSE',
       title=f'BGD(lr={lr_b}) vs SGD(lr={lr_s}) 收敛曲线')
ax.legend(); ax.grid(alpha=.3)
ax = axes[1]
ax.plot(u_tr, label='SGD 训练（逐次更新）')
ax.plot(u_te, '--', label='SGD 测试（逐次更新）')
ax.set(xlabel='参数更新次数', ylabel='MSE', title=f'SGD(lr={lr_s}) 逐次更新曲线')
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig('mse_curve.png', dpi=150); plt.show()

# ================= BGD vs SGD =================
print('\n===== BGD vs SGD 比较 =====')
th_s2, tr_s2, te_s2 = sgd(Xtr, y_train, Xte, y_test, lr=lr_s, n_epochs=N_EPOCHS)
rows = []
for nm, tr, te in [('BGD', tr_b, te_b), ('SGD', tr_s2, te_s2)]:
    rows.append([nm, f'{tr[-1]:.4f}', f'{te[-1]:.4f}', str(first_reach(tr))])
print(pd.DataFrame(rows, columns=['方法', '最终训练MSE', '最终测试MSE',
                                  '训练MSE进入终值×1.05所需epoch']).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(te_b, label=f'BGD(lr={lr_b})'); ax.plot(te_s2, '--', label=f'SGD(lr={lr_s})')
ax.set(xlabel='epoch', ylabel='测试MSE', title='BGD 与 SGD 前 50 epoch 收敛速度对比')  # [修改12]
ax.set_xlim(0, 50); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig('mid1_speed.png', dpi=150); plt.show()

# ================= 特征影响与扰动实验 =================
print('\n===== 特征影响分析 =====')
coef_s = pd.Series(th_ols[1:], index=feat_names)        # 特征已标准化
print('\n(a) 标准化模型系数:')
print(coef_s.sort_values(key=np.abs, ascending=False).to_string(float_format='%.4f'))
top3 = coef_s.abs().nlargest(3).index.tolist()
print(f'\n选取特征: {top3}')

print('\n(b) 扰动实验：单特征注入 1 个标准差高斯噪声后的预测变化')
rng = np.random.default_rng(0)
pred0 = Xte @ th_ols
avgs = []
for nm in top3:
    Xp = Xte.copy()
    Xp[:, feat_names.index(nm) + 1] += rng.normal(0, 1, len(Xp))
    avgs.append(float(np.mean(np.abs(Xp @ th_ols - pred0))))
    print(f'  {nm:<24} 系数={coef_s[nm]:+.4f} 平均|Δ预测|={avgs[-1]:.4f}')

plt.figure(figsize=(6, 4))
plt.bar(top3, avgs, color=['#4C72B0', '#DD8452', '#55A868'])
for i, v in enumerate(avgs):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
plt.ylabel('平均|预测变化|'); plt.title('扰动实验：1σ 特征噪声引起的预测变化')
plt.grid(axis='y', alpha=.3); plt.tight_layout()
plt.savefig('viz_F8_perturb.png', dpi=150); plt.show()

# ================= 最相关特征对实验 =================
print('\n===== 最相关特征对实验 =====')
corr = pd.DataFrame(X_train, columns=feat_names).corr()
c_abs = corr.abs().mask(np.eye(len(feat_names), dtype=bool))
f1, f2 = c_abs.stack().idxmax()
c1, c2 = feat_names.index(f1) + 1, feat_names.index(f2) + 1
print(f'最相关特征对: {f1} ↔ {f2}, Pearson r = {corr.loc[f1, f2]:.4f}')

plt.figure(figsize=(8, 6.5))
im = plt.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
plt.xticks(range(11), feat_names, rotation=45, ha='right')
plt.yticks(range(11), feat_names)
for i in range(11):
    for j in range(11):
        plt.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center',
                 fontsize=7, color='white' if abs(corr.values[i, j]) > .55 else 'black')
plt.colorbar(im, label='Pearson r')
plt.title('特征 Pearson 相关系数矩阵')
plt.tight_layout(); plt.savefig('viz_F1_heatmap.png', dpi=150); plt.show()

schemes = {'全模型(保留两者)': list(range(12)),
           f'移除{f2}': [c for c in range(12) if c != c2],
           f'仅用{f1}+{f2}': [0, c1, c2],
           f'仅用{f1}': [0, c1]}
results = {}
print(f'\n{"方案":<24}{"训练MSE":>10}{"测试MSE":>10} 相关特征系数')
for nm, cols in schemes.items():
    th, m_tr, m_te = fit_eval(cols)
    results[nm] = (th, cols, m_tr, m_te)
    info = ', '.join(f'{feat_names[c-1]}={t:+.4f}' for c, t in zip(cols, th) if c in (c1, c2))
    print(f'{nm:<24}{m_tr:>10.4f}{m_te:>10.4f} {info}')

th_f, cols_f = results['全模型(保留两者)'][:2]
th_r2, cols_r = results[f'移除{f2}'][:2]
mse_keep, mse_drop = results['全模型(保留两者)'][3], results[f'移除{f2}'][3]
print(f'>>> {f1} 系数: 全模型 {th_f[cols_f.index(c1)]:+.4f} → 移除{f2}后 {th_r2[cols_r.index(c1)]:+.4f}')
print(f'>>> 测试MSE: 保留两者 {mse_keep:.4f} vs 移除后 {mse_drop:.4f}')

names = list(schemes); tes = [results[n][3] for n in names]
plt.figure(figsize=(7, 4))
plt.bar(names, tes, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
plt.ylim(top=0.9)
for i, v in enumerate(tes):
    plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
plt.ylabel('测试MSE'); plt.title('特征相关性实验：各方案测试 MSE')
plt.xticks(rotation=10); plt.grid(axis='y', alpha=.3); plt.tight_layout()
plt.savefig('mid3_corr.png', dpi=150); plt.show()

th_wo = ridge_analytic(np.delete(Xtr, c2, axis=1), y_train, 0.0)
coef_wo = np.insert(th_wo[1:], c2 - 1, np.nan)
yy = np.arange(11)
plt.figure(figsize=(7.5, 5.5))
plt.barh(yy - .2, th_ols[1:], .4, label='全模型（保留两者）')
plt.barh(yy + .2, coef_wo, .4, label=f'移除 {f2}')
plt.yticks(yy, feat_names); plt.axvline(0, c='k', lw=.8)
plt.xlabel('标准化系数')
plt.title(f'共线性对系数的影响：{f1} {th_ols[c1]:+.3f} → {th_wo[c1]:+.3f}')
plt.legend(); plt.grid(axis='x', alpha=.3); plt.tight_layout()
plt.savefig('viz_F3_coef_shift.png', dpi=150); plt.show()

# ================= 岭回归 =================
print('\n===== 岭回归 =====')
rows = []
for lam in LAMS:
    th = ridge_analytic(Xtr, y_train, lam)
    rows.append([lam, float(np.mean((Xtr @ th - y_train) ** 2)),
                 float(np.mean((Xte @ th - y_test) ** 2)), th[c1], th[c2]])
df_r = pd.DataFrame(rows, columns=['λ', '训练MSE', '测试MSE', f'{f1}系数', f'{f2}系数'])
print(df_r.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
# 对照验证：解析法 vs 梯度法
lam_demo = 1e-2
th_a = ridge_analytic(Xtr, y_train, lam_demo)
th_g = ridge_bgd(Xtr, y_train, lam_demo, lr=0.1)
ridge_report(f'BGD λ={lam_demo:g}', th_g, Xtr, y_train, Xte, y_test)
print(f'解析法 vs BGD 系数最大差 = {np.max(np.abs(th_a - th_g)):.2e}')

# λ 对训练/测试 MSE 的影响（展示曲线）
lams_arr = df_r['λ'].to_numpy()
tr_arr, te_arr = df_r['训练MSE'].to_numpy(), df_r['测试MSE'].to_numpy()
best_i = int(np.argmin(te_arr))                         # 仅用于图上标参考线
plt.figure(figsize=(7, 4.5))
plt.semilogx(lams_arr + 1e-12, tr_arr, 'o-', label='训练MSE')
plt.semilogx(lams_arr + 1e-12, te_arr, 's--', label='测试MSE')
plt.axvline(max(lams_arr[best_i], 1e-12), ls=':', color='gray', alpha=.6)
pad = 0.1 * (max(tr_arr.max(), te_arr.max()) - min(tr_arr.min(), te_arr.min()))
plt.ylim(min(tr_arr.min(), te_arr.min()) - pad, max(tr_arr.max(), te_arr.max()) + pad)
plt.xlabel('λ'); plt.ylabel('MSE')
plt.title('岭回归：λ 对训练/测试 MSE 的影响'); plt.legend(); plt.grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig('adv_ridge_lambda.png', dpi=150); plt.show()

# 系数路径
paths = np.array([ridge_analytic(Xtr, y_train, l)[1:] for l in LAMS_PATH])
plt.figure(figsize=(7, 4.5))
for j, nm in enumerate(feat_names):
    if nm not in (f1, f2):
        plt.semilogx(LAMS_PATH, paths[:, j], color='gray', alpha=.35, lw=1)
plt.semilogx(LAMS_PATH, paths[:, feat_names.index(f1)], lw=2.5, label=f1)
plt.semilogx(LAMS_PATH, paths[:, feat_names.index(f2)], lw=2.5, label=f2)
plt.xlabel('λ（对数轴）'); plt.ylabel('系数')
plt.title(f'岭回归系数路径（最相关对 r={corr.loc[f1, f2]:.2f}）')
plt.legend(); plt.grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig('adv_ridge_path.png', dpi=150); plt.show()

# ================= 验证集选 λ =================
val_mse = [float(np.mean((Xl_va @ ridge_analytic(Xl_tr, ys_tr, l) - ys_va) ** 2))
           for l in LAMS]
lam_val = LAMS[int(np.argmin(val_mse))]
print('\n验证集选 λ：' + ', '.join(f'λ={l:g}:{v:.4f}' for l, v in zip(LAMS, val_mse)))
print(f'>>> 验证集选定 λ={lam_val:g}')
tr_fin, te_fin = ridge_report(f'最终模型(验证集选λ={lam_val:g})',
                              ridge_analytic(Xtr, y_train, lam_val),
                              Xtr, y_train, Xte, y_test)

plt.figure(figsize=(7, 4))
plt.semilogx(np.array(LAMS) + 1e-12, val_mse, 'o-')
plt.xlabel('λ（对数轴）'); plt.ylabel('验证MSE')
plt.title('验证集选 λ')
plt.grid(alpha=.3, which='both'); plt.tight_layout()
plt.savefig('viz_F9_val_curve.png', dpi=150); plt.show()

# ================= 系数稳定性量化 =================
lams_chk = [0.0, 1e-2, 1e-1]
mus, sds = [], []
print(f'\n 系数稳定性：5 个不同划分下 "{f1}" 系数的波动')
for lam in lams_chk:
    coefs = []
    for rs in range(5):
        Xa, _, ya, _ = train_test_split(X_train, y_train, test_size=0.2,
                                        stratify=y_train, random_state=rs)
        sc3 = StandardScaler().fit(Xa)
        Aa = np.hstack([np.ones((len(Xa), 1)), sc3.transform(Xa)])
        coefs.append(ridge_analytic(Aa, ya, lam)[c1])
    mus.append(np.mean(coefs)); sds.append(np.std(coefs))
    print(f'  λ={lam:<5g} 系数均值={mus[-1]:+.4f} 标准差={sds[-1]:.4f}')

plt.figure(figsize=(6, 4))
plt.errorbar(range(len(lams_chk)), mus, yerr=sds, fmt='o-', capsize=6, lw=1.5)
plt.xticks(range(len(lams_chk)), [f'λ={l:g}' for l in lams_chk])
plt.ylabel(f'{f1} 系数（5 个划分 ±1std）')
plt.title('L2 使跨划分系数波动收窄：' + ' → '.join(f'{s:.4f}' for s in sds))
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('viz_F10_stability.png', dpi=150); plt.show()

# ================= 各模型测试 MSE 汇总 =================
baseline = float(np.mean((y_test - y_train.mean()) ** 2))
vals = [baseline, te_b[-1], te_fin, te_arr[0]]
labels = ['均值基线', f'BGD 300轮\n(lr={lr_b})',
          f'岭回归\n(验证集选λ={lam_val:g})', 'OLS λ=0']
plt.figure(figsize=(7.5, 4.5))
bars = plt.bar(labels, vals, color=['#999999', '#4C72B0', '#DD8452', '#55A868'])
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v, f'{v:.4f}', ha='center', va='bottom')
plt.ylabel('测试MSE'); plt.ylim(0, max(vals) * 1.12)
plt.title('各模型测试 MSE')
plt.grid(axis='y', alpha=.3); plt.tight_layout()
plt.savefig('viz_F6_model_cmp.png', dpi=150); plt.show()

print(f'>>> [仅诊断对照，不作选型] 测试集扫描最优λ={lams_arr[best_i]:g}'
      f'(测试MSE={te_arr[best_i]:.4f}) '
      f'λ=0(OLS)测试MSE={te_arr[0]:.4f} '
      f'验证集选定λ={lam_val:g}(最终测试MSE={te_fin:.4f})')

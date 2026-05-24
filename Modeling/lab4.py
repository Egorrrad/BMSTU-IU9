from collections import Counter
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

W = 72
SEP2 = "-" * W

print("  ЗАГРУЗКА И ОПИСАТЕЛЬНЫЙ АНАЛИЗ ДАННЫХ")
df = pd.read_csv("diabetes.csv")
TARGET = "Glucose"
EXCLUDED = ["Outcome", "Insulin"]
CANDIDATES = [c for c in df.columns if c != TARGET and c not in EXCLUDED]

print(f"Объектов: {df.shape[0]},  Столбцов: {df.shape[1]}")
print(f"Целевая переменная : {TARGET}")
print(f"Исключены          : {EXCLUDED}")
print(f"Признаки-кандидаты : {CANDIDATES}")
print()
print("Описательная статистика:")

desc_cols = [TARGET] + CANDIDATES
desc = df[desc_cols].describe().T[["mean", "std", "min", "50%", "max"]]
desc.columns = ["среднее", "std", "min", "медиана", "max"]


def skewness(arr):
    a = np.asarray(arr, dtype=float)
    mu = a.mean()
    n = len(a)
    sigma = a.std()
    if sigma == 0:
        return 0.0
    return ((a - mu) ** 3).mean() / sigma ** 3


desc["асимметрия"] = [skewness(df[c]) for c in desc_cols]
print(desc.round(3).to_string())

print("  РАЗБИЕНИЕ ВЫБОРКИ")
df_sorted = df.sort_values(TARGET).reset_index(drop=True)
train_mask = (df_sorted.index % 2 == 0)
test_mask = (df_sorted.index % 2 == 1)

df_train = df_sorted[train_mask].reset_index(drop=True)
df_test = df_sorted[test_mask].reset_index(drop=True)

print(f"Train: {len(df_train)}  |  Test: {len(df_test)}")
print()
print("Проверка близости распределений Glucose:")
print(f"{'Показатель':<14} {'Train':>10} {'Test':>10}")
print(SEP2[:36])
for label, fn in [("среднее", np.mean), ("std", np.std), ("медиана", np.median), ("min", np.min), ("max", np.max)]:
    tv = fn(df_train[TARGET].values)
    ev = fn(df_test[TARGET].values)
    print(f"  {label:<12} {tv:>10.3f} {ev:>10.3f}")
print()

print("  СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ")
X_train_raw = df_train[CANDIDATES].values.astype(float)
X_test_raw = df_test[CANDIDATES].values.astype(float)
Y_train = df_train[TARGET].values.astype(float)
Y_test = df_test[TARGET].values.astype(float)

mu_train = X_train_raw.mean(axis=0)
std_train = X_train_raw.std(axis=0, ddof=1)

X_train_std = (X_train_raw - mu_train) / std_train
X_test_std = (X_test_raw - mu_train) / std_train

print()
print(f"{'Признак':<28} {'μ (train)':>12} {'σ (train)':>12}")
print(SEP2[:54])
for j, col in enumerate(CANDIDATES):
    print(f"  {col:<26} {mu_train[j]:>12.4f} {std_train[j]:>12.4f}")


def kendall_tau(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    n0 = n * (n - 1) // 2
    nc = nd = 0
    tx = ty = 0
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        sign = np.sign(dx) * np.sign(dy)
        nc += int((sign > 0).sum())
        nd += int((sign < 0).sum())
        tx += int((dx == 0).sum())
        ty += int((dy == 0).sum())
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    return (nc - nd) / denom if denom > 0 else 0.0


def spearman_rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    def rank(arr):
        order = np.argsort(arr)
        r = np.empty(n, dtype=float)
        r[order] = np.arange(1, n + 1, dtype=float)
        sorted_arr = arr[order]
        i = 0
        while i < n:
            j = i
            while j < n and sorted_arr[j] == sorted_arr[i]:
                j += 1
            avg = (i + 1 + j) / 2.0
            r[order[i:j]] = avg
            i = j
        return r

    rx = rank(x)
    ry = rank(y)
    d2 = ((rx - ry) ** 2).sum()
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return num / den if den > 0 else 0.0


print()

print("  КОРРЕЛЯЦИОННЫЙ АНАЛИЗ (на рабочей выборке)")
TAU_THRESHOLD = 0.2
T_CRIT = 1.96
n_train = len(Y_train)
K_DOF = n_train - 2

print(f"  n_train = {n_train},  K = n - 2 = {K_DOF},  t_крит = {T_CRIT:.2f} (α=0.05)")
print()
print(f"{'Признак':<24} {'τ':>9} {'ρ':>9} {'r':>9} {'|t(τ)|':>9} {'L1':>4} {'L2':>4} {'Отбор':>6}")
print(SEP2[:80])


def t_stat(coef, n):
    if abs(coef) >= 1.0:
        return float('inf')
    K = n - 2
    return abs(coef) * np.sqrt(K / (1 - coef ** 2))


tau_vals = {}
rho_vals = {}
r_vals = {}
selected = []

for j, col in enumerate(CANDIDATES):
    x = X_train_raw[:, j]
    tau = kendall_tau(x, Y_train)
    rho = spearman_rho(x, Y_train)
    r = pearson_r(x, Y_train)
    tau_vals[col] = tau
    rho_vals[col] = rho
    r_vals[col] = r

    t_tau = t_stat(tau, n_train)
    level1 = t_tau > T_CRIT
    level2 = abs(tau) >= TAU_THRESHOLD
    passed = level1 and level2
    if passed:
        selected.append(col)

    l1 = "+" if level1 else "-"
    l2 = "+" if level2 else "-"
    mark = " +" if passed else " -"
    print(f"  {col:<22} {tau:>+9.4f} {rho:>+9.4f} {r:>+9.4f} {t_tau:>9.2f}  {l1:>3}  {l2:>3}  {mark}")

print()
print("Согласие знаков τ Кендалла и ρ Спирмена:")
all_agree = True
for col in CANDIDATES:
    agree = (np.sign(tau_vals[col]) == np.sign(rho_vals[col]))
    status = "ОК" if agree else "РАСХОЖДЕНИЕ"
    if not agree:
        all_agree = False
    print(f"  {col:<26}: τ={tau_vals[col]:+.4f}, ρ={rho_vals[col]:+.4f}  [{status}]")
if all_agree:
    print("  => Знаки совпадают у всех пар.")

if len(selected) == 0:
    significant = [c for c in CANDIDATES if t_stat(tau_vals[c], n_train) > T_CRIT]
    if len(significant) == 0:
        selected = sorted(CANDIDATES, key=lambda c: abs(tau_vals[c]), reverse=True)[:5]
    else:
        selected = sorted(significant, key=lambda c: abs(tau_vals[c]), reverse=True)[:5]

print()
print(f"Отобранные признаки ({len(selected)}): {selected}")
print()

for col in selected:
    if abs(tau_vals[col]) >= TAU_THRESHOLD and abs(r_vals[col]) < 0.15:
        print(f"         {col} — τ={tau_vals[col]:+.4f}, r={r_vals[col]:+.4f} => нелинейная связь")
    else:
        print(f"  {col}: τ={tau_vals[col]:+.4f}, r={r_vals[col]:+.4f} — линейная составляющая присутствует")

print()

print("  МАТРИЦА τ КЕНДАЛЛА МЕЖДУ ОТОБРАННЫМИ ПРИЗНАКАМИ")
p = len(selected)
tau_matrix = np.zeros((p, p))
for i, ci in enumerate(selected):
    xi = X_train_raw[:, CANDIDATES.index(ci)]
    for j, cj in enumerate(selected):
        xj = X_train_raw[:, CANDIDATES.index(cj)]
        tau_matrix[i, j] = kendall_tau(xi, xj)

header = f"{'':28}" + "".join(f"{c[:10]:>12}" for c in selected)
print(header)
print(SEP2[:len(header)])
for i, ci in enumerate(selected):
    row = f"  {ci:<26}" + "".join(f"{tau_matrix[i, j]:>+12.4f}" for j in range(p))
    print(row)

print()
high_corr_pairs = []
for i in range(p):
    for j in range(i + 1, p):
        if abs(tau_matrix[i, j]) > 0.5:
            high_corr_pairs.append((selected[i], selected[j], tau_matrix[i, j]))

if high_corr_pairs:
    print("Пары с |τ| > 0.5 (сильная корреляция — риск мультиколлинеарности):")
    for ci, cj, v in high_corr_pairs:
        print(f"  {ci} & {cj}: τ={v:+.4f}")
else:
    print("Пар с |τ| > 0.5 не обнаружено — мультиколлинеарность не критична.")
print()


def mae(y_true, y_pred):
    return float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / len(y_true))


def mse(y_true, y_pred):
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float((diff ** 2).mean())


def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def estimate_beta_power(x_arr, y_arr, eps=1.0):
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    lnX = np.log(x + eps)
    lnY = np.log(y + eps)
    mean_lnX = lnX.mean()
    mean_lnY = lnY.mean()
    cov_xy = ((lnX - mean_lnX) * (lnY - mean_lnY)).mean()
    var_x = ((lnX - mean_lnX) ** 2).mean()
    beta = cov_xy / var_x if var_x > 0 else 0.0
    alpha = np.exp(mean_lnY - beta * mean_lnX)
    ss_res = ((lnY - (mean_lnY + beta * (lnX - mean_lnX))) ** 2).sum()
    ss_tot = ((lnY - mean_lnY) ** 2).sum()
    r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, alpha, r2_log


def choose_transform(beta):
    if 0.75 <= abs(beta) <= 1.25:
        return 'linear'
    if 1.5 <= beta <= 2.5:
        return 'sqrt'
    if 0.25 < beta < 0.75:
        return 'square'
    if abs(beta) > 2.5:
        return 'log'
    if beta <= -0.5:
        return 'inverse'
    return 'linear'


def apply_transform(arr, label, eps=1.0):
    arr = np.asarray(arr, dtype=float)
    if label == 'linear':
        return arr
    if label == 'sqrt':
        return np.sqrt(arr + eps)
    if label == 'log':
        return np.log(arr + eps)
    if label == 'square':
        return arr ** 2
    if label == 'inverse':
        return 1.0 / (arr + eps)
    return arr


def feature_feature_beta(df_train_data, features, eps=1.0):
    result = {}
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            if i < j:
                xi = df_train_data[fi].values.astype(float)
                xj = df_train_data[fj].values.astype(float)
                beta, _, _ = estimate_beta_power(xi, xj, eps=eps)
                result[(fi, fj)] = beta
    return result


def build_polynomial_features(X_std, feature_names):
    n, p = X_std.shape
    features_list = list(feature_names)
    X_poly = [X_std.copy()]
    names_poly = list(feature_names)

    for j in range(p):
        X_sq = X_std[:, j:j + 1] ** 2
        X_poly.append(X_sq)
        names_poly.append(f"{feature_names[j]}^2")

    for i in range(p):
        for j in range(i + 1, p):
            X_prod = X_std[:, i:i + 1] * X_std[:, j:j + 1]
            X_poly.append(X_prod)
            names_poly.append(f"{feature_names[i]}*{feature_names[j]}")

    X_result = np.hstack(X_poly)
    return X_result, names_poly


def polynomial_features_arbitrary(X, degree, feature_names=None):
    n, p = X.shape
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(p)]
    cols = []
    names = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(p), d):
            col = np.ones(n, dtype=float)
            for idx in combo:
                col = col * X[:, idx]
            cols.append(col)
            counts = Counter(combo)
            parts = []
            for idx in sorted(counts.keys()):
                exp = counts[idx]
                parts.append(feature_names[idx] if exp == 1 else f"{feature_names[idx]}^{exp}")
            names.append("·".join(parts))
    return np.column_stack(cols), names


def fit_poly_lstsq(X_train_poly, Y_train, X_test_poly, Y_test):
    A_tr = np.hstack([np.ones((X_train_poly.shape[0], 1)), X_train_poly])
    A_te = np.hstack([np.ones((X_test_poly.shape[0], 1)), X_test_poly])
    beta, _, _, _ = np.linalg.lstsq(A_tr, Y_train, rcond=None)
    y_hat_tr = A_tr @ beta
    y_hat_te = A_te @ beta
    mae_tr = float(np.mean(np.abs(Y_train - y_hat_tr)))
    mae_te = float(np.mean(np.abs(Y_test - y_hat_te)))
    gap_pct = 100.0 * (mae_te - mae_tr) / mae_tr if mae_tr > 0 else float("inf")
    try:
        cond = float(np.linalg.cond(A_tr.T @ A_tr))
    except np.linalg.LinAlgError:
        cond = float("inf")
    return mae_tr, mae_te, gap_pct, cond, beta


def sweep_polynomial_degrees(X_tr_base, X_te_base, Y_tr, Y_te, d_max=10, gap_limit=20.0):
    results = []
    rises = 0
    prev_mae_te = float("inf")
    for d in range(1, d_max + 1):
        X_tr_poly, names_poly = polynomial_features_arbitrary(X_tr_base, d)
        X_te_poly, _ = polynomial_features_arbitrary(X_te_base, d)
        mae_tr, mae_te, gap, cond, _ = fit_poly_lstsq(X_tr_poly, Y_tr, X_te_poly, Y_te)
        status = "ОК" if gap <= gap_limit else "переобучение"
        results.append(
            {"d": d, "n_feat": X_tr_poly.shape[1], "mae_tr": mae_tr, "mae_te": mae_te, "gap": gap, "cond": cond,
             "status": status})
        if mae_te > prev_mae_te:
            rises += 1
        else:
            rises = 0
        prev_mae_te = mae_te
        if rises >= 2:
            results[-1]["early_stop"] = True
            break
    return results


def pick_best_degree(results, gap_limit=20.0):
    ok = [r for r in results if r["gap"] <= gap_limit]
    if ok:
        best = min(ok, key=lambda r: r["mae_te"])
        return best, "удовлетворяет ограничению переобучения"
    best = min(results, key=lambda r: r["gap"])
    return best, "ни одна степень не прошла gap<=20%, взята с min gap"


def greedy_backward_poly(X_tr, X_te, Y_tr, Y_te, degree, feature_names, gap_limit=20.0):
    active = list(range(len(feature_names)))
    history = []
    prev_mae_te = float("inf")

    def _eval(idx_list):
        Xtr = X_tr[:, idx_list]
        Xte = X_te[:, idx_list]
        Xtr_p, _ = polynomial_features_arbitrary(Xtr, degree)
        Xte_p, _ = polynomial_features_arbitrary(Xte, degree)
        return fit_poly_lstsq(Xtr_p, Y_tr, Xte_p, Y_te)

    mae_tr_full, mae_te_full, gap_full, cond_full, _ = _eval(active)
    prev_mae_te = mae_te_full
    history.append({"step": 0, "removed": None, "active": [feature_names[i] for i in active], "n": len(active),
                    "mae_tr": mae_tr_full, "mae_te": mae_te_full, "gap": gap_full, })

    while len(active) > 1:
        best_mae = float("inf")
        best_j = None
        best_metrics = None
        for j in active:
            candidate = [i for i in active if i != j]
            mae_tr, mae_te, gap, cond, _ = _eval(candidate)
            if mae_te < best_mae:
                best_mae = mae_te
                best_j = j
                best_metrics = (mae_tr, mae_te, gap)

        if best_mae <= prev_mae_te:
            active = [i for i in active if i != best_j]
            prev_mae_te = best_mae
            history.append(
                {"step": len(history), "removed": feature_names[best_j], "active": [feature_names[i] for i in active],
                 "n": len(active), "mae_tr": best_metrics[0], "mae_te": best_metrics[1], "gap": best_metrics[2], })
        else:
            break

    return history


def compute_pca(X_std_train):
    n = X_std_train.shape[0]
    C = (X_std_train.T @ X_std_train) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / eigenvalues.sum()
    return eigenvectors, eigenvalues, explained


def apply_pca(X_std, V, k=None):
    if k is None:
        k = V.shape[1]
    return X_std @ V[:, :k]


print("Модель M0: Линейная")
print()

print("  МНК: ОЦЕНКА ПАРАМЕТРОВ МОДЕЛИ (рабочая выборка)")
sel_idx = [CANDIDATES.index(c) for c in selected]
X_tr = np.hstack([np.ones((len(X_train_std), 1)), X_train_std[:, sel_idx]])
X_te = np.hstack([np.ones((len(X_test_std), 1)), X_test_std[:, sel_idx]])

cond = np.linalg.cond(X_tr.T @ X_tr)
print(f"Число обусловленности XᵀX: {cond:.4e}")
if cond > 1e10:
    print("      высокая обусловленность — возможна мультиколлинеарность")
else:
    print("  Обусловленность приемлема.")

beta_m0 = np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ Y_train)

print()
print("Модель: Y = β0 + " + " + ".join(f"β{j + 1}·{c}_std" for j, c in enumerate(selected)))
print()
print(f"  {'Параметр':<12} {'Оценка':>12}")
print(SEP2[:26])
print(f"  {'β0 (intercept)':<12} {beta_m0[0]:>+12.4f}")
for j, col in enumerate(selected):
    print(f"  {'β' + str(j + 1) + '(' + col[:10] + ')':<12} {beta_m0[j + 1]:>+12.4f}")
print()

print("  ОЦЕНКА КАЧЕСТВА НА КОНТРОЛЬНОЙ ВЫБОРКЕ")
Y_hat_train_m0 = X_tr @ beta_m0
Y_hat_test_m0 = X_te @ beta_m0

Y_hat_naive = np.full(len(Y_test), Y_train.mean())

mae_train_m0 = mae(Y_train, Y_hat_train_m0)
mae_test_m0 = mae(Y_test, Y_hat_test_m0)
mae_naive = mae(Y_test, Y_hat_naive)
mse_test_m0 = mse(Y_test, Y_hat_test_m0)
rmse_test_m0 = float(np.sqrt(mse_test_m0))
r2_test_m0 = r_squared(Y_test, Y_hat_test_m0)

print(f"  {'Метрика':<30} {'Значение':>12}")
print(SEP2[:44])
print(f"  {'MAE train (рабочая)':<30} {mae_train_m0:>12.4f}  мг/дл  [основная]")
print(f"  {'MAE test (контроль)':<30} {mae_test_m0:>12.4f}  мг/дл  [основная]")
print(f"  {'MAE naive (прогноз Ȳ_train)':<30} {mae_naive:>12.4f}  мг/дл")
print(f"  {'MSE test':<30} {mse_test_m0:>12.4f}        [справочно]")
print(f"  {'RMSE test':<30} {rmse_test_m0:>12.4f}        [справочно]")
print(f"  {'R² test':<30} {r2_test_m0:>12.4f}        [справочно]")
print()

print("Критерии адекватности:")
ok1 = mae_test_m0 < mae_naive
ok2 = r2_test_m0 > 0
ok3 = abs(mae_test_m0 - mae_train_m0) / mae_train_m0 < 0.15


def check(cond, text):
    mark = "ОК" if cond else "НАРУШЕН"
    print(f"  [{mark}] {text}")


check(ok1, f"MAE_model ({mae_test_m0:.2f}) < MAE_naive ({mae_naive:.2f})")
check(ok2, f"R² > 0  ({r2_test_m0:.4f})")
check(ok3, f"|MAE_test - MAE_train| / MAE_train < 15% ({100 * abs(mae_test_m0 - mae_train_m0) / mae_train_m0:.1f}%)")
print()

print("  АНАЛИЗ ОСТАТКОВ (рабочая выборка)")
residuals = Y_train - Y_hat_train_m0
print(f"  Среднее остатков  : {residuals.mean():>+.6f}  (должно быть ~0)")
print(f"  Std остатков      : {residuals.std():>10.4f}")
print(f"  Min / Max         : {residuals.min():>+.4f} / {residuals.max():>+.4f}")
if abs(residuals.mean()) < 1e-6:
    print("  => Среднее ~0: предпосылка E[ε]=0 выполнена.")
else:
    print("      среднее остатков ненулевое")
print()

print("  КС-КРИТЕРИЙ (справочно)")


def ks_two_sample(a, b):
    combined = np.sort(np.concatenate([a, b]))
    n1, n2 = len(a), len(b)
    cdf_a = np.searchsorted(np.sort(a), combined, side='right') / n1
    cdf_b = np.searchsorted(np.sort(b), combined, side='right') / n2
    d = np.max(np.abs(cdf_a - cdf_b))
    d_crit = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))
    return d, d_crit


ks_d_m0, ks_crit_m0 = ks_two_sample(Y_hat_test_m0, Y_test)
print(f"  D         = {ks_d_m0:.6f}")
print(f"  D_крит    = {ks_crit_m0:.6f}  (α = 0.05)")
if ks_d_m0 < ks_crit_m0:
    print("  => H0 не отвергается: распределения прогнозов и факта схожи.")
else:
    print("  => H0 отвергается: распределения различаются.")
print()

print("  АНАЛИЗ УСТОЙЧИВОСТИ (смена ролей train↔test)")
mu2 = X_test_raw[:, sel_idx].mean(axis=0)
std2 = X_test_raw[:, sel_idx].std(axis=0, ddof=1)
X_tr2_re = (X_test_raw[:, sel_idx] - mu2) / std2
X_te2_re = (X_train_raw[:, sel_idx] - mu2) / std2
X_tr2 = np.hstack([np.ones((len(X_tr2_re), 1)), X_tr2_re])
X_te2 = np.hstack([np.ones((len(X_te2_re), 1)), X_te2_re])

beta2_m0 = np.linalg.solve(X_tr2.T @ X_tr2, X_tr2.T @ Y_test)
Y_hat_te2_m0 = X_te2 @ beta2_m0
mae_test2_m0 = mae(Y_train, Y_hat_te2_m0)

print(f"  Оригинальный fold: MAE_test = {mae_test_m0:.4f},  R² = {r2_test_m0:.4f}")
print(f"  Обратный fold    : MAE_test = {mae_test2_m0:.4f}")
diff_pct_m0 = 100 * abs(mae_test_m0 - mae_test2_m0) / max(mae_test_m0, mae_test2_m0)
print(f"  Расхождение MAE  : {diff_pct_m0:.1f}%")
if diff_pct_m0 < 15:
    print("  => Модель устойчива (расхождение < 15%).")
else:
    print("  => Модель неустойчива. Рассмотреть полиномиальное расширение.")
print()

print("  АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
print("  (MAE при удалении каждого признака по очереди)")
print(f"  Базовая MAE_test = {mae_test_m0:.4f} (полная модель с {len(selected)} признаками)")
print()
print(f"  {'Исключён':<28} {'MAE_test':>10} {'ΔMAE':>10}")
print(SEP2[:50])

for k, col in enumerate(selected):
    rem_idx = [i for i in range(len(sel_idx)) if i != k]
    if len(rem_idx) == 0:
        print(f"  {col:<28}   нет признаков — пропуск")
        continue
    X_tr_k = np.hstack([np.ones((len(X_train_std), 1)), X_train_std[:, [sel_idx[i] for i in rem_idx]]])
    X_te_k = np.hstack([np.ones((len(X_test_std), 1)), X_test_std[:, [sel_idx[i] for i in rem_idx]]])
    beta_k = np.linalg.solve(X_tr_k.T @ X_tr_k, X_tr_k.T @ Y_train)
    mae_k = mae(Y_test, X_te_k @ beta_k)
    delta = mae_k - mae_test_m0
    print(f"  {col:<28} {mae_k:>10.4f} {delta:>+10.4f}")

rejected = [c for c in CANDIDATES if c not in selected]
if rejected:
    print()
    print("  Добавление отброшенного признака к полной модели:")
    print(f"  {'Добавлен':<28} {'MAE_test':>10} {'ΔMAE':>10}")
    print(SEP2[:50])
    for col in rejected:
        j = CANDIDATES.index(col)
        ext_idx = sel_idx + [j]
        X_tr_e = np.hstack([np.ones((len(X_train_std), 1)), X_train_std[:, ext_idx]])
        X_te_e = np.hstack([np.ones((len(X_test_std), 1)), X_test_std[:, ext_idx]])
        beta_e = np.linalg.solve(X_tr_e.T @ X_tr_e, X_tr_e.T @ Y_train)
        mae_e = mae(Y_test, X_te_e @ beta_e)
        delta = mae_e - mae_test_m0
        print(f"  {col:<28} {mae_e:>10.4f} {delta:>+10.4f}")

print()
print()

print("Диагностика нелинейных связей между признаками (feature↔feature β)")
print()

ff_betas = feature_feature_beta(df_train, CANDIDATES)
print(f"{'Пара признаков':<40} {'β':>10} {'|β-1|>0.5':>12}")
print(SEP2[:64])

candidates_for_product = []
for (fi, fj), beta in sorted(ff_betas.items()):
    is_candidate = abs(beta - 1.0) > 0.5
    mark = "* кандидат" if is_candidate else ""
    print(f"  {fi} × {fj:<28} {beta:>+10.4f}  {mark}")
    if is_candidate:
        candidates_for_product.append((fi, fj))

print()
if candidates_for_product:
    print(f"Выявлено {len(candidates_for_product)} пар-кандидатов на произведение (|β-1|>0.5)")
else:
    print("Кандидатов на произведение не выявлено.")
print()
print()

print("Модель M1: β-преобразованная")
print()

print("   Оценка показателя степени β и выбор преобразования")
print()
print(f"{'Признак':<22} {'β':>10} {'Преобразование':>20}")
print(SEP2[:54])

transform_map_m1 = {}
X_train_transform = np.zeros_like(X_train_raw)
X_test_transform = np.zeros_like(X_test_raw)

for j, col in enumerate(CANDIDATES):
    x = X_train_raw[:, j]
    beta, _, _ = estimate_beta_power(x, Y_train)
    transform_type = choose_transform(beta)
    transform_map_m1[col] = (beta, transform_type)
    print(f"  {col:<22} {beta:>+10.4f} {transform_type:>20}")

    X_train_transform[:, j] = apply_transform(x, transform_type)
    X_test_transform[:, j] = apply_transform(X_test_raw[:, j], transform_type)

print()
print("   Стандартизация преобразованных признаков")
mu_transform = X_train_transform.mean(axis=0)
std_transform = X_train_transform.std(axis=0, ddof=1)
X_train_std_m1 = (X_train_transform - mu_transform) / std_transform
X_test_std_m1 = (X_test_transform - mu_transform) / std_transform

print()
print("   Двухуровневый фильтр на стандартизованных преобразованных признаках")
print()
print(f"{'Признак':<24} {'τ':>9} {'ρ':>9} {'r':>9} {'|t(τ)|':>9} {'L1':>4} {'L2':>4} {'Отбор':>6}")
print(SEP2[:80])

tau_vals_m1 = {}
selected_m1 = []

for j, col in enumerate(CANDIDATES):
    x = X_train_std_m1[:, j]
    tau = kendall_tau(x, Y_train)
    rho = spearman_rho(x, Y_train)
    r = pearson_r(x, Y_train)
    tau_vals_m1[col] = tau

    t_tau = t_stat(tau, n_train)
    level1 = t_tau > T_CRIT
    level2 = abs(tau) >= TAU_THRESHOLD
    passed = level1 and level2
    if passed:
        selected_m1.append(col)

    l1 = "+" if level1 else "-"
    l2 = "+" if level2 else "-"
    mark = " +" if passed else " -"
    print(f"  {col:<22} {tau:>+9.4f} {rho:>+9.4f} {r:>+9.4f} {t_tau:>9.2f}  {l1:>3}  {l2:>3}  {mark}")

if len(selected_m1) == 0:
    significant_m1 = [c for c in CANDIDATES if t_stat(tau_vals_m1[c], n_train) > T_CRIT]
    if len(significant_m1) == 0:
        selected_m1 = sorted(CANDIDATES, key=lambda c: abs(tau_vals_m1[c]), reverse=True)[:3]
    else:
        selected_m1 = sorted(significant_m1, key=lambda c: abs(tau_vals_m1[c]), reverse=True)[:3]

print()
print(f"Отобранные признаки ({len(selected_m1)}): {selected_m1}")
print()

print("   МНК на отобранных преобразованных признаках")
sel_idx_m1 = [CANDIDATES.index(c) for c in selected_m1]
X_tr_m1 = np.hstack([np.ones((len(X_train_std_m1), 1)), X_train_std_m1[:, sel_idx_m1]])
X_te_m1 = np.hstack([np.ones((len(X_test_std_m1), 1)), X_test_std_m1[:, sel_idx_m1]])

beta_m1 = np.linalg.solve(X_tr_m1.T @ X_tr_m1, X_tr_m1.T @ Y_train)

print()
print("Модель M1: Y = β0 + " + " + ".join(f"β{j + 1}·{c}_transform" for j, c in enumerate(selected_m1)))
print()
print(f"  {'Параметр':<12} {'Оценка':>12}")
print(SEP2[:26])
print(f"  {'β0 (intercept)':<12} {beta_m1[0]:>+12.4f}")
for j, col in enumerate(selected_m1):
    print(f"  {'β' + str(j + 1) + '(' + col[:10] + ')':<12} {beta_m1[j + 1]:>+12.4f}")
print()

print("  ОЦЕНКА КАЧЕСТВА НА КОНТРОЛЬНОЙ ВЫБОРКЕ")
Y_hat_train_m1 = X_tr_m1 @ beta_m1
Y_hat_test_m1 = X_te_m1 @ beta_m1

mae_train_m1 = mae(Y_train, Y_hat_train_m1)
mae_test_m1 = mae(Y_test, Y_hat_test_m1)
mse_test_m1 = mse(Y_test, Y_hat_test_m1)
rmse_test_m1 = float(np.sqrt(mse_test_m1))
r2_test_m1 = r_squared(Y_test, Y_hat_test_m1)

print(f"  {'Метрика':<30} {'Значение':>12}")
print(SEP2[:44])
print(f"  {'MAE train (рабочая)':<30} {mae_train_m1:>12.4f}  мг/дл")
print(f"  {'MAE test (контроль)':<30} {mae_test_m1:>12.4f}  мг/дл")
print(f"  {'MSE test':<30} {mse_test_m1:>12.4f}")
print(f"  {'RMSE test':<30} {rmse_test_m1:>12.4f}")
print(f"  {'R² test':<30} {r2_test_m1:>12.4f}")
print()

print("  АНАЛИЗ УСТОЙЧИВОСТИ M1: смена ролей выборок (train↔test)")
mu_transform_test = X_test_transform[:, sel_idx_m1].mean(axis=0)
std_transform_test = X_test_transform[:, sel_idx_m1].std(axis=0, ddof=1)
X_tr_m1_swap = (X_test_transform[:, sel_idx_m1] - mu_transform_test) / std_transform_test
X_te_m1_swap = (X_train_transform[:, sel_idx_m1] - mu_transform_test) / std_transform_test
X_tr_m1_swap = np.hstack([np.ones((len(X_tr_m1_swap), 1)), X_tr_m1_swap])
X_te_m1_swap = np.hstack([np.ones((len(X_te_m1_swap), 1)), X_te_m1_swap])

beta_m1_swap = np.linalg.solve(X_tr_m1_swap.T @ X_tr_m1_swap, X_tr_m1_swap.T @ Y_test)
Y_hat_m1_swap = X_te_m1_swap @ beta_m1_swap
mae_test_m1_swap = mae(Y_train, Y_hat_m1_swap)

print(f"  Оригинальный fold: MAE_test = {mae_test_m1:.4f}")
print(f"  Обратный fold    : MAE_test = {mae_test_m1_swap:.4f}")
print()

print()

print("Модель M2: Полиномиальное расширение признаков, перебор степеней 1..10")
D_MAX = 10
GAP_LIMIT = 20.0


def print_sweep_table(results, title):
    print(f"  {title}")
    print(f"  {'deg':>3} {'n_feat':>7} {'MAE_train':>12} {'MAE_test':>12} {'gap%':>8} {'cond(XᵀX)':>12} {'статус':>10}")
    print(SEP2[:72])
    for r in results:
        es = "  <- ранняя остановка" if r.get("early_stop", False) else ""
        print(f"  {r['d']:>3} {r['n_feat']:>7} {r['mae_tr']:>12.4f} {r['mae_te']:>12.4f} "
              f"{r['gap']:>+8.2f} {r['cond']:>12.2e} {r['status']:>10}{es}")
    print()


print("--- Сценарий А: полином на исходных (стандартизованных) признаках ---")
print()
results_A = sweep_polynomial_degrees(X_train_std, X_test_std, Y_train, Y_test, d_max=D_MAX, gap_limit=GAP_LIMIT)
print_sweep_table(results_A, f"Перебор (degrees 1..{results_A[-1]['d']}):")
best_A, reason_A = pick_best_degree(results_A, gap_limit=GAP_LIMIT)
print(f"  Наилучшая конфигурация сценария А: degree={best_A['d']}, MAE_test={best_A['mae_te']:.4f}")
print(f"  Обоснование: {reason_A}")
print()

print("--- Сценарий Б: полином на β-преобразованных стандартизованных признаках ---")
print()
results_B = sweep_polynomial_degrees(X_train_std_m1, X_test_std_m1, Y_train, Y_test, d_max=D_MAX, gap_limit=GAP_LIMIT)
print_sweep_table(results_B, f"Перебор (degrees 1..{results_B[-1]['d']}):")
best_B, reason_B = pick_best_degree(results_B, gap_limit=GAP_LIMIT)
print(f"  Наилучшая конфигурация сценария Б: degree={best_B['d']}, MAE_test={best_B['mae_te']:.4f}")
print(f"  Обоснование: {reason_B}")
print()

if best_A["mae_te"] <= best_B["mae_te"]:
    best_m2 = best_A
    best_m2_scenario = "А (сырые)"
    X_tr_m2_base, X_te_m2_base = X_train_std, X_test_std
else:
    best_m2 = best_B
    best_m2_scenario = "Б (β-преобр.)"
    X_tr_m2_base, X_te_m2_base = X_train_std_m1, X_test_std_m1

print(f"Итог M2: сценарий {best_m2_scenario}, degree={best_m2['d']}, "
      f"MAE_test={best_m2['mae_te']:.4f}, gap={best_m2['gap']:+.2f}%")
print()

X_tr_poly_best, _ = polynomial_features_arbitrary(X_tr_m2_base, best_m2["d"])
X_te_poly_best, _ = polynomial_features_arbitrary(X_te_m2_base, best_m2["d"])
mae_train_m2, mae_test_m2, gap_m2, cond_m2, beta_m2 = fit_poly_lstsq(X_tr_poly_best, Y_train, X_te_poly_best, Y_test)

A_te_m2 = np.hstack([np.ones((X_te_poly_best.shape[0], 1)), X_te_poly_best])
Y_hat_test_m2 = A_te_m2 @ beta_m2
mse_test_m2 = mse(Y_test, Y_hat_test_m2)
rmse_test_m2 = float(np.sqrt(mse_test_m2))
r2_test_m2 = r_squared(Y_test, Y_hat_test_m2)

print("  АНАЛИЗ УСТОЙЧИВОСТИ M2: смена ролей выборок (train↔test)")
if best_m2_scenario.startswith("А"):
    mu_sw = X_test_raw.mean(axis=0)
    std_sw = X_test_raw.std(axis=0, ddof=1)
    X_tr_base_sw = (X_test_raw - mu_sw) / std_sw
    X_te_base_sw = (X_train_raw - mu_sw) / std_sw
else:
    mu_sw = X_test_transform.mean(axis=0)
    std_sw = X_test_transform.std(axis=0, ddof=1)
    X_tr_base_sw = (X_test_transform - mu_sw) / std_sw
    X_te_base_sw = (X_train_transform - mu_sw) / std_sw

X_tr_sw_poly, _ = polynomial_features_arbitrary(X_tr_base_sw, best_m2["d"])
X_te_sw_poly, _ = polynomial_features_arbitrary(X_te_base_sw, best_m2["d"])
mae_tr_sw, mae_te_sw, _, _, _ = fit_poly_lstsq(X_tr_sw_poly, Y_test, X_te_sw_poly, Y_train)
print(f"  Оригинальный fold: MAE_test = {mae_test_m2:.4f}")
print(f"  Обратный fold    : MAE_test = {mae_te_sw:.4f}")
swap_gap = 100.0 * abs(mae_te_sw - mae_test_m2) / mae_test_m2
print(f"  Расхождение между фолдами: {swap_gap:.2f}%")
print()

print("Модель M3: полиномиальная регрессия с пошаговым исключением признаков (backward elimination)")
print()
print("τ-фильтр не применяется. Начальный набор: все 6 признаков-кандидатов.")
print("На каждом шаге исключается признак, удаление которого минимизирует MAE_test.")
print("Остановка: удаление любого признака из текущего набора не снижает MAE_test.")
print()

M3_DEGREES = [1, 2, 3]
best_m3_overall = None
best_m3_degree = None
best_m3_mae_te = float("inf")

for d_m3 in M3_DEGREES:
    print(f"  --- Степень d={d_m3} ---")
    history_m3 = greedy_backward_poly(X_train_std, X_test_std, Y_train, Y_test, degree=d_m3, feature_names=CANDIDATES,
                                      gap_limit=GAP_LIMIT)

    print(f"  {'шаг':>4} {'удалён':>28} {'n':>3} {'MAE_tr':>10} {'MAE_te':>10} {'gap%':>7}")
    print(f"  " + "-" * 65)
    for h in history_m3:
        removed = h["removed"] if h["removed"] else "(нет)"
        print(f"  {h['step']:>4} {removed:>28} {h['n']:>3} "
              f"{h['mae_tr']:>10.4f} {h['mae_te']:>10.4f} {h['gap']:>+7.2f}")
    print()

    best_step = min(history_m3, key=lambda x: x["mae_te"])
    gap_ok = best_step["gap"] <= GAP_LIMIT
    note = "" if gap_ok else " [gap>20%, исключено из сравнения]"
    print(f"  Лучший шаг d={d_m3}: признаки={best_step['active']}, "
          f"MAE_test={best_step['mae_te']:.4f}, gap={best_step['gap']:+.2f}%{note}")
    print()

    if gap_ok and best_step["mae_te"] < best_m3_mae_te:
        best_m3_mae_te = best_step["mae_te"]
        best_m3_overall = best_step
        best_m3_degree = d_m3

if best_m3_overall is None:
    candidates_m3 = []
    for d_m3 in M3_DEGREES:
        h = greedy_backward_poly(X_train_std, X_test_std, Y_train, Y_test, degree=d_m3, feature_names=CANDIDATES)
        best_step = min(h, key=lambda x: x["mae_te"])
        best_step["_d"] = d_m3
        candidates_m3.append(best_step)
    best_m3_overall = min(candidates_m3, key=lambda x: x["mae_te"])
    best_m3_degree = best_m3_overall["_d"]

print(f"Итог M3: степень d={best_m3_degree}, признаки={best_m3_overall['active']},")
print(f"  MAE_test={best_m3_overall['mae_te']:.4f}, gap={best_m3_overall['gap']:+.2f}%")

m3_sel_idx = [CANDIDATES.index(c) for c in best_m3_overall["active"]]
X_tr_m3_sel = X_train_std[:, m3_sel_idx]
X_te_m3_sel = X_test_std[:, m3_sel_idx]
X_tr_m3_p, _ = polynomial_features_arbitrary(X_tr_m3_sel, best_m3_degree)
X_te_m3_p, _ = polynomial_features_arbitrary(X_te_m3_sel, best_m3_degree)
mae_train_m3, mae_test_m3, gap_m3, _, beta_m3_coef = fit_poly_lstsq(X_tr_m3_p, Y_train, X_te_m3_p, Y_test)
A_te_m3 = np.hstack([np.ones((X_te_m3_p.shape[0], 1)), X_te_m3_p])
Y_hat_test_m3 = A_te_m3 @ beta_m3_coef
mse_test_m3 = mse(Y_test, Y_hat_test_m3)
rmse_test_m3 = float(np.sqrt(mse_test_m3))
r2_test_m3 = r_squared(Y_test, Y_hat_test_m3)

mu_sw_m3 = X_test_raw[:, m3_sel_idx].mean(axis=0)
std_sw_m3 = X_test_raw[:, m3_sel_idx].std(axis=0, ddof=1)
X_tr_m3_sw = (X_test_raw[:, m3_sel_idx] - mu_sw_m3) / std_sw_m3
X_te_m3_sw = (X_train_raw[:, m3_sel_idx] - mu_sw_m3) / std_sw_m3
X_tr_m3_sw_p, _ = polynomial_features_arbitrary(X_tr_m3_sw, best_m3_degree)
X_te_m3_sw_p, _ = polynomial_features_arbitrary(X_te_m3_sw, best_m3_degree)
_, mae_te_m3_sw, _, _, _ = fit_poly_lstsq(X_tr_m3_sw_p, Y_test, X_te_m3_sw_p, Y_train)
print(f"  Stability swap M3: ориг={mae_test_m3:.4f}, обр={mae_te_m3_sw:.4f}, "
      f"расх={100 * abs(mae_te_m3_sw - mae_test_m3) / mae_test_m3:.1f}%")
print()

print("Снижение размерности методом главных компонент (PCA)")
print()

V_pca, eigenvalues_pca, explained_pca = compute_pca(X_train_std)
cumulative_pca = np.cumsum(explained_pca)

print(f"  {'Компонента':>10} {'lambda':>10} {'Доля%':>9} {'Накопл.%':>10}")
print("  " + "-" * 43)
for k_i in range(len(eigenvalues_pca)):
    print(f"  {'PC' + str(k_i + 1):>10} {eigenvalues_pca[k_i]:>10.4f} "
          f"{100 * explained_pca[k_i]:>9.2f} {100 * cumulative_pca[k_i]:>10.2f}")
print()

print("Матрица нагрузок V (строки=признаки, столбцы=компоненты):")
hdr_pca = f"  {'Признак':<28}" + "".join(f"{'PC' + str(k_i + 1):>9}" for k_i in range(len(eigenvalues_pca)))
print(hdr_pca)
print("  " + "-" * (len(hdr_pca) - 2))
for j, col in enumerate(CANDIDATES):
    row = f"  {col:<28}" + "".join(f"{V_pca[j, k_i]:>+9.4f}" for k_i in range(len(eigenvalues_pca)))
    print(row)
print()

print("Модель M4: полиномиальное расширение в базисе главных компонент")
print()

Z_train_full = apply_pca(X_train_std, V_pca)
Z_test_full = apply_pca(X_test_std, V_pca)

print("--- M4a: перебор степеней d=1..3, полный базис главных компонент (6 компонент) ---")
print()
results_m4a = sweep_polynomial_degrees(Z_train_full, Z_test_full, Y_train, Y_test, d_max=3, gap_limit=GAP_LIMIT)
print_sweep_table(results_m4a, "Полный PCA-базис:")
best_m4a, reason_m4a = pick_best_degree(results_m4a, gap_limit=GAP_LIMIT)
print(f"  Наилучшая конфигурация M4a: degree={best_m4a['d']}, MAE_test={best_m4a['mae_te']:.4f} ({reason_m4a})")
print()

print("--- M4b: d=2, варьируется k первых компонент ---")
print()
print(f"  {'k':>3} {'MAE_tr':>10} {'MAE_te':>10} {'gap%':>7} {'cond':>12}")
print("  " + "-" * 45)
best_m4b_mae = float("inf")
best_m4b_k = 1
for k_pca in range(1, len(CANDIDATES) + 1):
    Z_tr_k = apply_pca(X_train_std, V_pca, k=k_pca)
    Z_te_k = apply_pca(X_test_std, V_pca, k=k_pca)
    X_tr_k_p, _ = polynomial_features_arbitrary(Z_tr_k, 2)
    X_te_k_p, _ = polynomial_features_arbitrary(Z_te_k, 2)
    mae_tr_k, mae_te_k, gap_k, cond_k, _ = fit_poly_lstsq(X_tr_k_p, Y_train, X_te_k_p, Y_test)
    mark = " *" if (gap_k <= GAP_LIMIT and mae_te_k < best_m4b_mae) else ""
    if gap_k <= GAP_LIMIT and mae_te_k < best_m4b_mae:
        best_m4b_mae = mae_te_k
        best_m4b_k = k_pca
    print(f"  {k_pca:>3} {mae_tr_k:>10.4f} {mae_te_k:>10.4f} {gap_k:>+7.2f} {cond_k:>12.2e}{mark}")
print()
print(f"  Наилучшая конфигурация M4b: k={best_m4b_k}, d=2, MAE_test={best_m4b_mae:.4f}")
print()

if best_m4a["mae_te"] <= best_m4b_mae:
    best_m4_d = best_m4a["d"]
    best_m4_k = len(CANDIDATES)
    best_m4_label = f"M4a d={best_m4_d}"
    Z_tr_m4 = Z_train_full
    Z_te_m4 = Z_test_full
else:
    best_m4_d = 2
    best_m4_k = best_m4b_k
    best_m4_label = f"M4b k={best_m4_k}"
    Z_tr_m4 = apply_pca(X_train_std, V_pca, k=best_m4_k)
    Z_te_m4 = apply_pca(X_test_std, V_pca, k=best_m4_k)

X_tr_m4_p, _ = polynomial_features_arbitrary(Z_tr_m4, best_m4_d)
X_te_m4_p, _ = polynomial_features_arbitrary(Z_te_m4, best_m4_d)
mae_train_m4, mae_test_m4, gap_m4, _, beta_m4_coef = fit_poly_lstsq(X_tr_m4_p, Y_train, X_te_m4_p, Y_test)
A_te_m4 = np.hstack([np.ones((X_te_m4_p.shape[0], 1)), X_te_m4_p])
Y_hat_test_m4 = A_te_m4 @ beta_m4_coef
mse_test_m4 = mse(Y_test, Y_hat_test_m4)
rmse_test_m4 = float(np.sqrt(mse_test_m4))
r2_test_m4 = r_squared(Y_test, Y_hat_test_m4)

print(f"Итог M4 ({best_m4_label}): d={best_m4_d}, k={best_m4_k} компонент,")
print(f"  MAE_test={mae_test_m4:.4f}, gap={gap_m4:+.2f}%")

mu_raw_test = X_test_raw.mean(axis=0)
std_raw_test = X_test_raw.std(axis=0, ddof=1)
X_sw_m4_tr_raw = (X_test_raw - mu_raw_test) / std_raw_test
X_sw_m4_te_raw = (X_train_raw - mu_raw_test) / std_raw_test
V_sw_m4, _, _ = compute_pca(X_sw_m4_tr_raw)
Z_sw_m4_tr = apply_pca(X_sw_m4_tr_raw, V_sw_m4, k=best_m4_k)
Z_sw_m4_te = apply_pca(X_sw_m4_te_raw, V_sw_m4, k=best_m4_k)
X_tr_m4_sw_p, _ = polynomial_features_arbitrary(Z_sw_m4_tr, best_m4_d)
X_te_m4_sw_p, _ = polynomial_features_arbitrary(Z_sw_m4_te, best_m4_d)
_, mae_te_m4_sw, _, _, _ = fit_poly_lstsq(X_tr_m4_sw_p, Y_test, X_te_m4_sw_p, Y_train)
print(f"  Stability swap M4: ориг={mae_test_m4:.4f}, обр={mae_te_m4_sw:.4f}, "
      f"расх={100 * abs(mae_te_m4_sw - mae_test_m4) / mae_test_m4:.1f}%")
print()

print("СРАВНЕНИЕ МОДЕЛЕЙ")

print()

print(f"{'Модель':<30} {'MAE_train':>12} {'MAE_test':>12} {'RMSE_test':>12} {'R2_test':>12}")
print("-" * 78)
print(f"  {'M0 (линейная)':<28} {mae_train_m0:>12.4f} {mae_test_m0:>12.4f} {rmse_test_m0:>12.4f} {r2_test_m0:>12.4f}")
print(
    f"  {'M1 (бета-преобр.)':<28} {mae_train_m1:>12.4f} {mae_test_m1:>12.4f} {rmse_test_m1:>12.4f} {r2_test_m1:>12.4f}")
m2_label = f"M2 (poly d={best_m2['d']}, {best_m2_scenario[:1]})"
print(f"  {m2_label:<28} {mae_train_m2:>12.4f} {mae_test_m2:>12.4f} {rmse_test_m2:>12.4f} {r2_test_m2:>12.4f}")
m3_label = f"M3 (backward d={best_m3_degree})"
print(f"  {m3_label:<28} {mae_train_m3:>12.4f} {mae_test_m3:>12.4f} {rmse_test_m3:>12.4f} {r2_test_m3:>12.4f}")
m4_label = f"M4 ({best_m4_label})"
print(f"  {m4_label:<28} {mae_train_m4:>12.4f} {mae_test_m4:>12.4f} {rmse_test_m4:>12.4f} {r2_test_m4:>12.4f}")
print()

mae_test_vals = {"M0": mae_test_m0, "M1": mae_test_m1, "M2": mae_test_m2, "M3": mae_test_m3, "M4": mae_test_m4, }
winner = min(mae_test_vals, key=mae_test_vals.get)
print(f"Основная метрика (MAE_test): наилучший результат — {winner}, MAE_test = {mae_test_vals[winner]:.4f}")

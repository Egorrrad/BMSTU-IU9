import numpy as np
import pandas as pd

CSV_PATH = "diabetes.csv"
df = pd.read_csv(CSV_PATH)

print("Количество объектов:", df.shape[0])
print("Количество столбцов:", df.shape[1])
print("\nСтолбцы:", list(df.columns))
print("\nПервые 5 строк:")
print(df.head().to_string(index=False))
print("\nОписательная статистика:")
print(df.describe().round(2).to_string())
print("\nПропущенные значения:")
print(df.isnull().sum().to_string())

TARGET_ORIG = "Outcome"
orig_feature_cols = [c for c in df.columns if c != TARGET_ORIG]
rename_map = {col: f"k{i+1}" for i, col in enumerate(orig_feature_cols)}
rename_map[TARGET_ORIG] = "Outcome"
df = df.rename(columns=rename_map)

print("\nСоответствие исходных столбцов и новых имён:")
for orig, new in rename_map.items():
    print(f"  {new} <- {orig}")

TARGET = "Outcome"
feature_cols = [c for c in df.columns if c != TARGET]
X_df = df[feature_cols]
X = X_df.to_numpy(dtype=float)
n_features = len(feature_cols)


print("\n" + "-"*60)
print("Сравнение бинаризаций: среднее vs медиана")
print("-"*60)
print(f"  {'Признак':<6}  {'Среднее':>10}  {'Медиана':>10}  {'|ср - мед|':>12}  {'% от std':>10}")
for col in feature_cols:
    mean_val   = X_df[col].mean()
    median_val = X_df[col].median()
    std_val    = X_df[col].std()
    diff       = abs(mean_val - median_val)
    pct        = diff / std_val * 100 if std_val > 0 else 0
    print(f"  {col:<6}  {mean_val:>10.3f}  {median_val:>10.3f}  {diff:>12.3f}  {pct:>9.1f}%")


def compute_contingency_table(A, B):
    a = int(np.sum(A & B))
    b = int(np.sum(A & ~B))
    c = int(np.sum(~A & B))
    d = int(np.sum(~A & ~B))
    return a, b, c, d

def yule_colligation(a, b, c, d):
    sqrt_ad = np.sqrt(a * d)
    sqrt_bc = np.sqrt(b * c)
    denom = sqrt_ad + sqrt_bc
    if denom == 0:
        return 0.0
    return (sqrt_ad - sqrt_bc) / denom

def association(a, b, c, d):
    ad, bc = a * d, b * c
    denom = ad + bc
    if denom == 0:
        return 0.0
    return (ad - bc) / denom

def contingency_pearson(a, b, c, d):
    ad, bc = a * d, b * c
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return 0.0
    return (ad - bc) / denom

def bernstein_coefficient(A, B):
    P_A = A.mean()
    P_B = B.mean()
    if P_B == 0:
        return 0.0
    P_A_given_B = (A & B).sum() / B.sum()
    return P_B * (P_A_given_B - P_A)

# Бинаризация по медиане
X_bin = {}
for col in feature_cols:
    med = X_df[col].median()
    X_bin[col] = (X_df[col] < med).to_numpy()

# Бинаризация по среднему (для сравнения)
X_bin_mean = {}
for col in feature_cols:
    X_bin_mean[col] = (X_df[col] < X_df[col].mean()).to_numpy()

def build_matrices(X_bin_dict):
    mat_yule = np.zeros((n_features, n_features))
    mat_ass  = np.zeros((n_features, n_features))
    mat_cont = np.zeros((n_features, n_features))
    mat_bern = np.zeros((n_features, n_features))
    for i, ci in enumerate(feature_cols):
        A = X_bin_dict[ci]
        for j, cj in enumerate(feature_cols):
            B = X_bin_dict[cj]
            a, b, c, d = compute_contingency_table(A, B)
            mat_yule[i, j] = yule_colligation(a, b, c, d)
            mat_ass[i, j]  = association(a, b, c, d)
            mat_cont[i, j] = contingency_pearson(a, b, c, d)
            mat_bern[i, j] = bernstein_coefficient(A, B)
    return mat_yule, mat_ass, mat_cont, mat_bern

mat_yule, mat_ass, mat_cont, mat_bern = build_matrices(X_bin)
mat_yule_m, mat_ass_m, mat_cont_m, mat_bern_m = build_matrices(X_bin_mean)

df_yule = pd.DataFrame(mat_yule, index=feature_cols, columns=feature_cols)
df_ass  = pd.DataFrame(mat_ass,  index=feature_cols, columns=feature_cols)
df_cont = pd.DataFrame(mat_cont, index=feature_cols, columns=feature_cols)
df_bern = pd.DataFrame(mat_bern, index=feature_cols, columns=feature_cols)

df_bern_m = pd.DataFrame(mat_bern_m, index=feature_cols, columns=feature_cols)

print("\n" + "-"*60)
print("Коэффициент коллигации Бернштейна (бинаризация по среднему):")
print("-"*60)
print(df_bern_m.round(4).to_string())

print("\n" + "-"*60)
print("Коэффициент коллигации Бернштейна (бинаризация по медиане):")
print("-"*60)
print(df_bern.round(4).to_string())

print("\n" + "-"*60)
print("Коэффициент контингенции Пирсона (бинаризация по медиане):")
print("-"*60)
print(df_cont.round(4).to_string())

print("\n" + "-"*60)
print("Коэффициент коллигации Юла (бинаризация по медиане):")
print("-"*60)
print(df_yule.round(4).to_string())

print("\n" + "-"*60)
print("Коэффициент ассоциации (бинаризация по медиане):")
print("-"*60)
print(df_ass.round(4).to_string())

df_yule.to_csv("yule_colligation_matrix.csv")
df_ass.to_csv("association_matrix.csv")
df_cont.to_csv("contingency_pearson_matrix.csv")
df_bern.to_csv("bernstein_matrix.csv")
print("\nМатрицы сохранены.")



print("\nПроверки (бинаризация по медиане):")

print("\n  1. Коэффициент коллигации Юла и Бернштейна совпадают по знаку:")
mismatch_count = 0
for i, col_i in enumerate(feature_cols):
    for j, col_j in enumerate(feature_cols):
        if i == j:
            continue
        q  = mat_yule[i, j]
        kb = mat_bern[i, j]
        if not ((np.sign(q) == np.sign(kb)) or (abs(q) < 1e-9 and abs(kb) < 1e-9)):
            print(f"     [!] {col_i} vs {col_j}: Юла={q:.4f}, Бернштейн={kb:.6f}")
            mismatch_count += 1
if mismatch_count == 0:
    print("     ✓ Знаки совпадают для всех пар признаков.")

print("\n  2. Соотношение |ассоциация| >= |коллигация| >= |контингенция|:")
v1 = v2 = 0
for i, col_i in enumerate(feature_cols):
    for j, col_j in enumerate(feature_cols):
        if i == j:
            continue
        av = abs(mat_ass[i, j])
        cv = abs(mat_yule[i, j])
        kv = abs(mat_cont[i, j])
        if av < cv - 1e-9:
            v1 += 1
        if cv < kv - 1e-9:
            v2 += 1
if v1 == 0 and v2 == 0:
    print("     ✓ Соотношение выполняется для всех пар признаков.")


# СОКРАЩЕНИЕ РАЗМЕРНОСТИ
# Критерий: средний |коллигация Юла| по строке (без диагонали).
# Признаки с низким средним исключаются.
print("\n" + "-"*60)
print("Сокращение размерности")
print("-"*60)

mean_abs_yule = []
for i, col in enumerate(feature_cols):
    off_diag = [abs(mat_yule[i, j]) for j in range(n_features) if j != i]
    mean_abs_yule.append((col, np.mean(off_diag)))
mean_abs_yule.sort(key=lambda x: -x[1])

print("\n  Средний |коэффициент коллигации Юла| по строке (без диагонали):")
for col, val in mean_abs_yule:
    print(f"    {col}: {val:.4f}")

THRESHOLD = 0.10
selected_cols  = [col for col, val in mean_abs_yule if val >= THRESHOLD]
excluded_cols  = [col for col, val in mean_abs_yule if val < THRESHOLD]

print(f"\n  Порог отбора: средний |Юла| >= {THRESHOLD}")
print(f"  Отобрано  ({len(selected_cols)}): {selected_cols}")
print(f"  Исключено ({len(excluded_cols)}): {excluded_cols}")


# K-MEANS
def euclidean(a, b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))

def assign_clusters(X, centers):
    labels = np.empty(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        dists = [euclidean(X[i], centers[k]) for k in range(len(centers))]
        labels[i] = int(np.argmin(dists))
    return labels

def compute_centers(X, labels, k):
    centers = np.zeros((k, X.shape[1]))
    for j in range(k):
        mask = labels == j
        if mask.any():
            centers[j] = X[mask].mean(axis=0)
    return centers

def wcss_val(X, labels, centers):
    return sum(euclidean(X[i], centers[labels[i]])**2 for i in range(X.shape[0]))

def kmeans_single_run(X, k, max_iter, rng):
    idx = rng.choice(X.shape[0], size=k, replace=False)
    centers = X[idx].copy()
    labels = np.full(X.shape[0], -1, dtype=int)
    for iteration in range(1, max_iter + 1):
        new_labels = assign_clusters(X, centers)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers = compute_centers(X, labels, k)
    return labels, centers, wcss_val(X, labels, centers), iteration

def kmeans(X, k=2, n_init=10, max_iter=300, seed=42, verbose=False):
    best_labels, best_centers, best_inertia = None, None, np.inf
    rng_master = np.random.default_rng(seed)
    for run in range(1, n_init + 1):
        rng = np.random.default_rng(rng_master.integers(0, 10**9))
        labels, centers, inertia, iters = kmeans_single_run(X, k, max_iter, rng)
        if verbose:
            marker = " <- лучший" if inertia < best_inertia else ""
            print(f"  Запуск {run:>2}/{n_init}: итераций={iters:>3}, WCSS={inertia:>10.2f}{marker}")
        if inertia < best_inertia:
            best_labels, best_centers, best_inertia = labels.copy(), centers.copy(), inertia
    return best_labels, best_centers, best_inertia

X_sel = df[selected_cols].to_numpy(dtype=float)
sel_means = X_sel.mean(axis=0)
sel_stds  = X_sel.std(axis=0)
sel_stds[sel_stds == 0] = 1
X_sel_scaled = (X_sel - sel_means) / sel_stds

print("\n" + "-"*60)
print("Выбор числа кластеров (метод локтя, WCSS):")
print("-"*60)
for k in range(1, 7):
    if k == 1:
        center = X_sel_scaled.mean(axis=0)
        w = sum(euclidean(X_sel_scaled[i], center)**2 for i in range(X_sel_scaled.shape[0]))
    else:
        _, _, w = kmeans(X_sel_scaled, k=k, n_init=10, seed=42)
    print(f"  K={k}: WCSS = {w:.2f}")

for K in [2, 3]:
    print(f"\n" + "-"*60)
    print(f"K-means, K={K}, признаки: {selected_cols}")
    print("-"*60)
    labels, centers, best_w = kmeans(X_sel_scaled, k=K, n_init=10, seed=42, verbose=True)
    print(f"\n  Лучшее WCSS = {best_w:.2f}")

    counts = np.bincount(labels)
    print(f"\n  Распределение по кластерам:")
    for cl in range(K):
        print(f"    Кластер {cl+1}: {counts[cl]} объектов ({counts[cl]/len(labels)*100:.1f}%)")

    print(f"\n  Центры кластеров (исходные единицы):")
    header = f"    {'':12}" + "".join(f"{c:>8}" for c in selected_cols)
    print(header)
    for cl in range(K):
        center_orig = centers[cl] * sel_stds + sel_means
        row = f"    {'Кластер '+str(cl+1):<12}" + "".join(f"{v:>8.2f}" for v in center_orig)
        print(row)

    print(f"\n  Статистика по кластерам (среднее / std):")
    for cl in range(K):
        mask = labels == cl
        print(f"    Кластер {cl+1}:")
        for ci, col in enumerate(selected_cols):
            vals = X_sel[mask, ci]
            print(f"      {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
                  f"min={vals.min():.2f}, max={vals.max():.2f}")
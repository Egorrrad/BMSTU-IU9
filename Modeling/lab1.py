import matplotlib.pyplot as plt
import numpy as np

N_X, N_Y = 20, 20
X_MIN, X_MAX = -3.0, 3.0
Y_MIN, Y_MAX = -3.0, 3.0
GRID_SEED = 42


def TEST_FUNC(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


N_APP = 10
M_DENSE = 100
APP_SEED = 7

Z_DISTRIBUTION = 'uniform'
Z_UNIFORM_RANGE = (-1.0, 1.0)
Z_NORMAL_PARAMS = (0.0, 0.35)
Z_BETA_PARAMS = (2.0, 2.0)
Z_RAYLEIGH_SCALE = 0.5

FIXED_Z_TABLE = None


def prepare_grid(x_labels, y_labels, z_table):
    sy = np.argsort(y_labels)
    sx = np.argsort(x_labels)
    return x_labels[sx], y_labels[sy], z_table[sy, :][:, sx]


def bilinear_model(x, y, x_grid, y_grid, z_grid):
    i = int(np.clip(np.searchsorted(x_grid, x, side='right') - 1, 0, len(x_grid) - 2))
    j = int(np.clip(np.searchsorted(y_grid, y, side='right') - 1, 0, len(y_grid) - 2))
    t = (x - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
    s = (y - y_grid[j]) / (y_grid[j + 1] - y_grid[j])
    z00 = z_grid[j, i];
    z10 = z_grid[j, i + 1]
    z01 = z_grid[j + 1, i];
    z11 = z_grid[j + 1, i + 1]
    return (1 - t) * (1 - s) * z00 + t * (1 - s) * z10 + (1 - t) * s * z01 + t * s * z11


def bilinear_grid(x_grid, y_grid, z_grid, X_out, Y_out):
    return np.vectorize(lambda x, y: bilinear_model(x, y, x_grid, y_grid, z_grid))(X_out, Y_out)


def make_test_grid(n_x, n_y, func, seed):
    rng = np.random.default_rng(seed)
    x_base = np.linspace(X_MIN, X_MAX, n_x)
    y_base = np.linspace(Y_MIN, Y_MAX, n_y)
    dx = (X_MAX - X_MIN) / (n_x - 1) * 0.4
    dy = (Y_MAX - Y_MIN) / (n_y - 1) * 0.4
    x_grid = np.sort(x_base + rng.uniform(-dx, dx, n_x))
    y_grid = np.sort(y_base + rng.uniform(-dy, dy, n_y))
    x_grid[0], x_grid[-1] = X_MIN, X_MAX
    y_grid[0], y_grid[-1] = Y_MIN, Y_MAX
    z_table = np.array([[func(x, y) for x in x_grid] for y in y_grid])
    return x_grid, y_grid, z_table


def test_cell_midpoints(x_grid, y_grid, z_grid, true_func):
    errors, points = [], []
    for j in range(len(y_grid) - 1):
        for i in range(len(x_grid) - 1):
            xm = (x_grid[i] + x_grid[i + 1]) / 2.0
            ym = (y_grid[j] + y_grid[j + 1]) / 2.0
            errors.append(abs(true_func(xm, ym) - bilinear_model(xm, ym, x_grid, y_grid, z_grid)))
            points.append((xm, ym))
    return np.array(errors), points


def check_nodes_error(x_grid, y_grid, z_grid):
    return max(
        abs(z_grid[j, i] - bilinear_model(x, y, x_grid, y_grid, z_grid)) for j, y in enumerate(y_grid) for i, x in
        enumerate(x_grid))


def print_test_results(x_grid, y_grid, z_grid, errors, midpoints):
    W = 68
    idx_min = int(np.argmin(errors))
    idx_max = int(np.argmax(errors))

    print(f"\n{'═' * W}")
    print(f"  ЧАСТЬ 1: ТЕСТИРОВАНИЕ  "
          f"z = sin(√(x²+y²)) ∈ [-1,1]   {len(x_grid)}×{len(y_grid)},  seed={GRID_SEED}")
    print(f"{'═' * W}")
    n_cells = len(errors)
    node_err = check_nodes_error(x_grid, y_grid, z_grid)
    print(f"  Узлов сетки:           {len(x_grid)} × {len(y_grid)} = {len(x_grid) * len(y_grid)}")
    print(f"  Ячеек:                 {len(x_grid) - 1} × {len(y_grid) - 1} = {n_cells}")
    print(f"  Контрольных точек:     {n_cells}  (центры ячеек)")
    print(f"  Ошибка в узлах:        {node_err:.2e}  (должна быть ~0)")
    print(f"{'─' * W}")

    special = {idx_min, idx_max}
    shown = sorted(set(range(min(7, len(errors)))) | set(range(max(0, len(errors) - 3), len(errors))) | special)
    print(f"  {'№':<5}  {'x_цент':>8}  {'y_цент':>8}  "
          f"{'z_ист':>8}  {'z_мод':>8}  {'|ошибка|':>10}")
    print(f"  {'─' * 5}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 10}")
    prev = -1
    for k in shown:
        if k - prev > 1:
            print(f"  {'...'}")
        xm, ym = midpoints[k]
        zt = TEST_FUNC(xm, ym)
        zm = bilinear_model(xm, ym, x_grid, y_grid, z_grid)
        mark = " ← MIN" if k == idx_min else (" ← MAX" if k == idx_max else "")
        print(f"  {k:<5}  {xm:>8.4f}  {ym:>8.4f}  "
              f"{zt:>8.5f}  {zm:>8.5f}  {errors[k]:>10.6f}{mark}")
        prev = k

    print(f"{'─' * W}")
    print(f"  MIN = {errors[idx_min]:.6f}   в ({midpoints[idx_min][0]:.4f}, {midpoints[idx_min][1]:.4f})")
    print(f"  MAX = {errors[idx_max]:.6f}   в ({midpoints[idx_max][0]:.4f}, {midpoints[idx_max][1]:.4f})")
    print(f"  Ср. = {errors.mean():.6f}")

    print(f"{'═' * W}")


def plot_testing(x_grid, y_grid, z_grid, errors, midpoints):
    X_d = np.linspace(x_grid.min(), x_grid.max(), 250)
    Y_d = np.linspace(y_grid.min(), y_grid.max(), 250)
    XX, YY = np.meshgrid(X_d, Y_d)
    Z_true = TEST_FUNC(XX, YY)
    Z_model = bilinear_grid(x_grid, y_grid, z_grid, XX, YY)
    Xn, Yn = np.meshgrid(x_grid, y_grid)
    mx = np.array([p[0] for p in midpoints])
    my = np.array([p[1] for p in midpoints])
    idx_min, idx_max = int(np.argmin(errors)), int(np.argmax(errors))

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Часть 1: Тестирование  z=sin(√(x²+y²)) ∈ [-1,1]  "
                 f"[{len(x_grid)}×{len(y_grid)}]", fontsize=12)

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(XX, YY, Z_true, cmap='viridis', alpha=0.80)
    ax1.scatter(Xn.ravel(), Yn.ravel(), z_grid.ravel(), color='red', s=8, zorder=5)
    ax1.set_title('Истинная поверхность\nz = sin(√(x²+y²))')
    ax1.set_zlim(-1, 1)
    ax1.set_xlabel('x');
    ax1.set_ylabel('y');
    ax1.set_zlabel('z')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(XX, YY, Z_model, cmap='plasma', alpha=0.80)
    ax2.scatter(Xn.ravel(), Yn.ravel(), z_grid.ravel(), color='red', s=8, zorder=5)
    ax2.set_title('Билинейная модель')
    ax2.set_zlim(-1, 1)
    ax2.set_xlabel('x');
    ax2.set_ylabel('y');
    ax2.set_zlabel('z')

    ax3 = fig.add_subplot(133)
    sc = ax3.scatter(mx, my, c=errors, cmap='hot_r', s=40, zorder=3)
    ax3.vlines(x_grid, y_grid.min(), y_grid.max(), colors='gray', lw=0.3, alpha=0.4)
    ax3.hlines(y_grid, x_grid.min(), x_grid.max(), colors='gray', lw=0.3, alpha=0.4)
    plt.colorbar(sc, ax=ax3, label='|ошибка|')
    ax3.scatter(*midpoints[idx_min], color='blue', s=120, zorder=5, label=f'MIN={errors[idx_min]:.4f}')
    ax3.scatter(*midpoints[idx_max], color='lime', s=120, zorder=5, marker='^', label=f'MAX={errors[idx_max]:.4f}')
    ax3.legend(fontsize=8)
    ax3.set_title('Ошибка в центрах ячеек')
    ax3.set_xlabel('x');
    ax3.set_ylabel('y');
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('testing.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: testing.png")
    plt.show()


def sample_z(n_rows, n_cols, distribution, seed):
    rng = np.random.default_rng(seed)
    if distribution == 'uniform':
        lo, hi = Z_UNIFORM_RANGE
        return rng.uniform(lo, hi, (n_rows, n_cols))
    elif distribution == 'normal':
        mu, sigma = Z_NORMAL_PARAMS
        return np.clip(rng.normal(mu, sigma, (n_rows, n_cols)), -1.0, 1.0)
    elif distribution == 'beta':
        a, b = Z_BETA_PARAMS
        return 2.0 * rng.beta(a, b, (n_rows, n_cols)) - 1.0
    elif distribution == 'rayleigh':
        z = rng.rayleigh(Z_RAYLEIGH_SCALE, (n_rows, n_cols))
        return np.clip(z, 0, 2.0) - 1.0
    else:
        raise ValueError(f"Неизвестный закон: {distribution!r}. "
                         f"Доступны: 'uniform', 'normal', 'beta', 'rayleigh'")


def make_app_grid(n, distribution, seed, z_table_fixed=None):
    rng = np.random.default_rng(seed)
    x_grid = np.sort(rng.uniform(X_MIN, X_MAX, n))
    y_grid = np.sort(rng.uniform(Y_MIN, Y_MAX, n))
    if z_table_fixed is not None:
        assert z_table_fixed.shape == (n, n), f"FIXED_Z_TABLE должна иметь форму ({n}, {n})"
        z_table = z_table_fixed.astype(float)
    else:
        z_table = sample_z(n, n, distribution, seed)
    return x_grid, y_grid, z_table


def densify_grid(x_grid, y_grid, z_grid, m):
    x_dense = np.linspace(x_grid.min(), x_grid.max(), m)
    y_dense = np.linspace(y_grid.min(), y_grid.max(), m)
    Xd, Yd = np.meshgrid(x_dense, y_dense)
    Z_dense = bilinear_grid(x_grid, y_grid, z_grid, Xd, Yd)
    return x_dense, y_dense, Z_dense


def print_application_results(x_grid, y_grid, z_grid, x_dense, y_dense, Z_dense, dist_name):
    W = 68
    m = len(x_dense)
    dx = (x_grid.max() - x_grid.min()) / (m - 1)
    dy = (y_grid.max() - y_grid.min()) / (m - 1)

    print(f"\n{'═' * W}")
    print(f"  ЧАСТЬ 2: ПРИМЕНЕНИЕ — СГУЩЕНИЕ СЕТКИ ВЫСОТ")
    print(f"{'═' * W}")
    print(f"  Исходная:   {len(x_grid)}×{len(y_grid)}  (псевдослучайная, z ~ {dist_name})")
    print(f"  Итоговая:   {m}×{m}  (регулярная)")
    print(f"  Диапазон x: [{x_grid.min():.4f}, {x_grid.max():.4f}]")
    print(f"  Диапазон y: [{y_grid.min():.4f}, {y_grid.max():.4f}]")
    print(f"{'─' * W}")
    print(f"  Исходный шаг  Δx ∈ [{np.diff(x_grid).min():.4f}, {np.diff(x_grid).max():.4f}],  "
          f"Δy ∈ [{np.diff(y_grid).min():.4f}, {np.diff(y_grid).max():.4f}]")
    print(f"  Шаг после сгущения:  Δx = {dx:.5f},  Δy = {dy:.5f}")
    print(f"  z исходное:  [{z_grid.min():.4f}, {z_grid.max():.4f}]")
    print(f"  z итоговое:  [{Z_dense.min():.4f}, {Z_dense.max():.4f}]")
    print(f"{'═' * W}")


def plot_application(x_grid, y_grid, z_grid, x_dense, y_dense, Z_dense, dist_name):
    Xn, Yn = np.meshgrid(x_grid, y_grid)
    Xd, Yd = np.meshgrid(x_dense, y_dense)

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(f"Часть 2: Сгущение {len(x_grid)}×{len(y_grid)} → "
                 f"{len(x_dense)}×{len(y_dense)}  (z ~ {dist_name})", fontsize=12)

    kw_nodes = dict(color='red', s=35, zorder=5)
    zlim = (min(z_grid.min(), Z_dense.min()) - 0.05, max(z_grid.max(), Z_dense.max()) + 0.05)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(Xn, Yn, z_grid, cmap='terrain', alpha=0.70)
    ax1.scatter(Xn.ravel(), Yn.ravel(), z_grid.ravel(), **kw_nodes, label='Узлы')
    ax1.set_title(f'Исходная сетка  {len(x_grid)}×{len(y_grid)}\n'
                  f'(неравномерная, z ~ {dist_name})', fontsize=10)
    ax1.set_zlim(*zlim)
    ax1.set_xlabel('x');
    ax1.set_ylabel('y');
    ax1.set_zlabel('z')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(Xd, Yd, Z_dense, cmap='terrain', alpha=0.85)
    ax2.scatter(Xn.ravel(), Yn.ravel(), z_grid.ravel(), **kw_nodes, label='Исходные узлы')
    ax2.set_title(f'Модель: сетка {len(x_dense)}×{len(y_dense)}\n'
                  f'(Δx={x_dense[1] - x_dense[0]:.4f}, Δy={y_dense[1] - y_dense[0]:.4f})', fontsize=10)
    ax2.set_zlim(*zlim)
    ax2.set_xlabel('x');
    ax2.set_ylabel('y');
    ax2.set_zlabel('z')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('application.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: application.png")
    plt.show()


if __name__ == "__main__":
    xg, yg, zt = make_test_grid(N_X, N_Y, TEST_FUNC, GRID_SEED)
    xg, yg, zg = prepare_grid(xg, yg, zt)
    errors, mpts = test_cell_midpoints(xg, yg, zg, TEST_FUNC)
    print_test_results(xg, yg, zg, errors, mpts)
    plot_testing(xg, yg, zg, errors, mpts)

    xg2, yg2, zt2 = make_app_grid(N_APP, Z_DISTRIBUTION, APP_SEED, FIXED_Z_TABLE)
    xg2, yg2, zg2 = prepare_grid(xg2, yg2, zt2)
    xd, yd, Zd = densify_grid(xg2, yg2, zg2, M_DENSE)
    print_application_results(xg2, yg2, zg2, xd, yd, Zd, Z_DISTRIBUTION)
    plot_application(xg2, yg2, zg2, xd, yd, Zd, Z_DISTRIBUTION)

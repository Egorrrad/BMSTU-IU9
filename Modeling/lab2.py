import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

D_CRIT  = 0.05
OUT_DIR = "lab2"
data_path = "diabetes.csv"

TRIM_Q0 = TRIM_Q1 = 0.083

def load_samples(file_path: str):
    df = pd.read_csv(file_path, usecols=["Glucose", "Outcome"])
    df = df[df["Glucose"] > 0]
    ksi0 = df.loc[df["Outcome"] == 0, "Glucose"].to_numpy(float)
    ksi1 = df.loc[df["Outcome"] == 1, "Glucose"].to_numpy(float)
    return ksi0, ksi1

def trim_tails(sample: np.ndarray, q: float) -> np.ndarray:
    lo, hi = np.quantile(sample, q), np.quantile(sample, 1 - q)
    return sample[(sample >= lo) & (sample <= hi)]

def ecdf(sample: np.ndarray):
    xs = np.sort(sample)
    fs = np.arange(1, len(xs) + 1) / len(xs)
    return xs, fs

def ks_stat(a: np.ndarray, b: np.ndarray):
    sa, sb = np.sort(a), np.sort(b)
    grid = np.sort(np.unique(np.concatenate([sa, sb])))
    Fa = np.searchsorted(sa, grid, side="right") / len(sa)
    Fb = np.searchsorted(sb, grid, side="right") / len(sb)
    idx = np.argmax(np.abs(Fa - Fb))
    return np.abs(Fa - Fb)[idx], grid[idx]

def h(b, ksi0: np.ndarray):
    m1 = np.mean(ksi0 ** b)
    m2 = np.mean(ksi0 ** (2 * b))
    return m2 / m1 ** 2

def estimate_parameters(ksi0: np.ndarray, ksi1: np.ndarray):
    mu1 = np.mean(ksi1)
    mu2 = np.mean(ksi1 ** 2)
    R   = mu2 / mu1 ** 2

    beta  = brentq(lambda b: h(b, ksi0) - R, 0.01, 10.0, xtol=1e-10)
    alpha = mu1 / np.mean(ksi0 ** beta)
    return alpha, beta

def phi(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return alpha * x ** beta

def save_plots(ksi0, ksi1, alpha, beta, d_raw, d_model):
    os.makedirs(OUT_DIR, exist_ok=True)

    x0, F0 = ecdf(ksi0)
    x1, F1 = ecdf(ksi1)
    xp, Fp = ecdf(phi(ksi0, alpha, beta))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(x0, F0, where="post", color="steelblue", lw=2,
            label="F0(x) — xi0 (без диабета)")
    ax.step(x1, F1, where="post", color="tomato",    lw=2,
            label="F1(x) — xi1 (с диабетом)")
    ax.set_title(f"Эмпирические функции распределения Glucose\n"
                 f"max|F0 - F1| = {d_raw:.4f}")
    ax.set_xlabel("Глюкоза (мг/дл)")
    ax.set_ylabel("Fn(x)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ecdf_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # после преобразования phi
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(xp, Fp, where="post", color="seagreen", lw=2, ls="--",
            label=f"F_phi(x) — phi(xi0) = {alpha:.4f}*xi0^{beta:.4f}")
    ax.step(x1, F1, where="post", color="tomato", lw=2,
            label="F1(x) — xi1 (с диабетом)")
    verdict_sym = "<=" if d_model <= D_CRIT else ">"
    ax.set_title(f"Критерий согласия: phi(xi0) -> xi1\n"
                 f"max|F_phi - F1| = {d_model:.4f}  "
                 f"{verdict_sym}  D = {D_CRIT}")
    ax.set_xlabel("Глюкоза (мг/дл)")
    ax.set_ylabel("Fn(x)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ecdf_transformed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def print_results(ksi0, ksi1, alpha, beta, d_raw, z_raw, d_model, z_model):
    print(f"Модель:  xi1 = alpha * xi0^beta")
    print(f"  alpha = {alpha:.6f},  beta = {beta:.6f}")
    print()
    print(f"Критерий согласия  (D_crit = {D_CRIT}):")
    print(f"  до преобразования:    max|F0 - F1|    = {d_raw:.6f}"
          f"  (z = {z_raw:.2f})")
    print(f"  после преобразования: max|F_phi - F1| = {d_model:.6f}"
          f"  (z = {z_model:.2f})")

def main():
    ksi0, ksi1 = load_samples(data_path)

    ksi0 = trim_tails(ksi0, TRIM_Q0)
    ksi1 = trim_tails(ksi1, TRIM_Q1)
    alpha, beta = estimate_parameters(ksi0, ksi1)

    d_raw,   z_raw   = ks_stat(ksi0, ksi1)
    d_model, z_model = ks_stat(phi(ksi0, alpha, beta), ksi1)

    save_plots(ksi0, ksi1, alpha, beta, d_raw, d_model)
    print_results(ksi0, ksi1, alpha, beta, d_raw, z_raw, d_model, z_model)


if __name__ == "__main__":
    main()
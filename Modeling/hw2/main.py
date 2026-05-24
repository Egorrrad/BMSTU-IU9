import math
import random


def nCr(n, r):
    return math.comb(n, r)


def theoretical_p_win(N, K, p_stay):
    p_a = 1.0 / N
    p_not_a = (N - 1) / N

    term_a = (1.0 / nCr(N - 1, K)) * p_a

    term_not_a = (1.0 / nCr(N - 2, K)) * p_not_a

    p_a_given_b = term_a / (term_a + term_not_a)

    p_switch = p_not_a / (N - K - 1)
    return p_stay * p_a_given_b + (1 - p_stay) * p_switch


def monty_hall_simulation(n_doors=3, k_open=1, p_stay=0.5, n_trials=5000, seed=42):
    random.seed(seed)
    wins = 0
    for _ in range(n_trials):
        prize = random.randint(0, n_doors - 1)
        choice = random.randint(0, n_doors - 1)

        available = [i for i in range(n_doors) if i != prize and i != choice]
        if len(available) < k_open:
            available = [i for i in range(n_doors) if i != choice]
        opened = random.sample(available, k_open)

        remaining = [i for i in range(n_doors) if i != choice and i not in opened]

        if random.random() < p_stay:
            final_choice = choice
        else:
            final_choice = random.choice(remaining)
        if final_choice == prize:
            wins += 1
    return wins / n_trials


print("Основной эксперимент (N=3, K=1)")
for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    est = monty_hall_simulation(p_stay=p)
    theo = theoretical_p_win(3, 1, p)
    print(f"p_stay = {p:.2f} | Monte-Carlo = {est:.4f} "
          f"| Teoria = {theo:.4f} | diff = {abs(est - theo):.4f}")

print("\nТестирование стратегии switch (p_stay=0)")
for N, K in [(3, 1), (10, 8), (100, 98)]:
    est = monty_hall_simulation(N, K, p_stay=0, n_trials=10000, seed=0)
    theo = theoretical_p_win(N, K, 0)
    print(f"N={N:>3}, K={K:>2} | est = {est:.4f} "
          f"| theory = {theo:.4f} | diff = {abs(est - theo):.4f}")

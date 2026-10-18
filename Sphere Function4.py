
import numpy as np
import math
import matplotlib.pyplot as plt


# ==========================================
# 1. Sphere Objective Function
# ==========================================
def sphere(x):
    return np.sum(x ** 2)


# ==========================================
# 2. Levy Flight
# ==========================================
def levy_flight(dim, beta=1.5):
    sigma = (
        math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)

    return 0.01 * u / (
        np.abs(v) ** (1 / beta) + 1e-12
    )


# ==========================================
# 3. Improved AVOA Algorithm
# ==========================================
def improved_AVOA(
    objective,
    dim=5,
    population_size=20,
    max_iter=500,
    lb=-100,
    ub=100,
    seed=42,
    patience=30
):

    rng = np.random.default_rng(seed)

    lb = np.broadcast_to(
        np.asarray(lb, dtype=float), (dim,)
    )
    ub = np.broadcast_to(
        np.asarray(ub, dtype=float), (dim,)
    )
    span = ub - lb

    # Initial population
    X = rng.uniform(
        lb, ub, (population_size, dim)
    )

    # Opposition-based initialization
    X_opp = lb + ub - X
    X_all = np.vstack((X, X_opp))

    fit_all = np.array([
        objective(x) for x in X_all
    ])

    selected = np.argsort(fit_all)[:population_size]
    X = X_all[selected].copy()
    fitness = fit_all[selected].copy()

    best_idx = np.argmin(fitness)
    best_pos = X[best_idx].copy()
    best_fit = float(fitness[best_idx])

    history = []
    stall = 0

    # Main optimization loop
    for t in range(max_iter):

        progress = t / max_iter
        old_best = best_fit

        # Sort agents by fitness
        order = np.argsort(fitness)
        X = X[order]
        fitness = fitness[order]

        leader1 = X[0].copy()
        leader2 = X[1].copy()

        # Adaptive local search
        local_sigma = (
            0.08 * (1 - progress) ** 2 + 1e-5
        )

        X_new = X.copy()

        for i in range(population_size):

            current = X[i].copy()

            # Select a leader
            if rng.random() < 0.8:
                R = leader1
            else:
                R = leader2

            # Hunger coefficient
            h = rng.uniform(-2, 2)
            z = rng.uniform(-1, 1)
            r = rng.random()

            disturbance = h * (
                np.sin(np.pi * progress / 2) ** 2.5
                + np.cos(np.pi * progress / 2)
                - 1
            )

            F = (
                (2 * r + 1)
                * z
                * (1 - progress)
                + disturbance
            )

            D = np.abs(R - current)

            # --------------------------
            # Exploration
            # --------------------------
            if abs(F) >= 1:

                if rng.random() < 0.6:
                    candidate = R - F * D
                else:
                    random_point = rng.uniform(lb, ub)
                    candidate = (
                        R - F
                        + rng.random(dim) * random_point
                    )

            # --------------------------
            # Exploitation
            # --------------------------
            elif abs(F) >= 0.5:

                if rng.random() < 0.5:
                    candidate = (
                        D * (F + rng.random(dim))
                        - (R - current)
                    )
                else:
                    candidate = (
                        R - (F + rng.random(dim)) * D
                    )

            else:

                if rng.random() < 0.5:
                    candidate = (
                        R
                        - np.abs(D)
                        * F
                        * levy_flight(dim)
                    )
                else:
                    candidate = (
                        R - F
                        + rng.random(dim)
                        * levy_flight(dim)
                        * D
                    )

            # Local search around the global best
            if rng.random() < 0.25:

                local_candidate = (
                    best_pos
                    + rng.normal(0, local_sigma, dim)
                    * span
                )

                local_candidate = np.clip(
                    local_candidate, lb, ub
                )

                candidate = np.clip(
                    candidate, lb, ub
                )

                if objective(local_candidate) < objective(
                    candidate
                ):
                    candidate = local_candidate

            X_new[i] = np.clip(
                candidate, lb, ub
            )

        # Greedy selection
        new_fitness = np.array([
            objective(x) for x in X_new
        ])

        improved = new_fitness < fitness

        X[improved] = X_new[improved]
        fitness[improved] = new_fitness[improved]

        # Update global best
        idx = np.argmin(fitness)

        if fitness[idx] < best_fit:
            best_fit = float(fitness[idx])
            best_pos = X[idx].copy()

        # Stagnation detection
        if best_fit < old_best:
            stall = 0
        else:
            stall += 1

        # Restart the worst agents
        if stall >= patience:

            n_restart = max(
                1, population_size // 5
            )

            worst = np.argsort(fitness)[-n_restart:]

            X[worst] = rng.uniform(
                lb, ub, (n_restart, dim)
            )

            fitness[worst] = np.array([
                objective(x) for x in X[worst]
            ])

            # Preserve the global best
            worst_idx = np.argmax(fitness)
            X[worst_idx] = best_pos.copy()
            fitness[worst_idx] = best_fit

            stall = 0

        history.append(best_fit)

    return best_pos, best_fit, history


# ==========================================
# 4. Run AVOA
# ==========================================
best_pos, best_value, history = improved_AVOA(
    objective=sphere,
    dim=5,
    population_size=20,
    max_iter=5000,
    lb=-100,
    ub=100,
    seed=42,
    patience=30
)


# ==========================================
# 5. Display Results
# ==========================================
print("=" * 40)
print("Improved AVOA Results")
print("=" * 40)

print("Number of agents:", 20)
print("Number of dimensions:", 5)
print("Maximum iterations:", 5000)

print("\nBest position:")
for i, value in enumerate(best_pos):
    print(f"x{i + 1} = {value:.10f}")

print("\nBest fitness:")
print(f"{best_value:.12e}")

print("\nTheoretical optimum: 0")


# ==========================================
# 6. Convergence Plot
# ==========================================
plt.figure(figsize=(9, 5))

plt.semilogy(
    np.maximum(history, 1e-16),
    color="blue",
    linewidth=2
)

plt.xlabel("Iteration")
plt.ylabel("Best Fitness (log scale)")
plt.title(
    "Improved AVOA - Sphere Function\n"
    "20 Agents, 5 Dimensions"
)

plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

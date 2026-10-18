# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:54:28 2026

@author: rostami
"""

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. Sphere objective function
# -----------------------------
def sphere(x):
    return np.sum(x ** 2)


# -----------------------------
# 2. Levy flight
# -----------------------------
def levy_flight(dim, beta=1.5):
    sigma = (
        np.math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (
            np.math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)

    return u / (np.abs(v) ** (1 / beta) + 1e-12)


# -----------------------------
# 3. AVOA algorithm
# -----------------------------
def AVOA(
    objective,
    dim=5,
    population_size=20,
    max_iter=100,
    lb=-10,
    ub=10,
    p1=0.6,
    p2=0.4,
    p3=0.6,
    seed=42
):
    np.random.seed(seed)

    # Initialize vultures
    X = np.random.uniform(
        lb, ub, (population_size, dim)
    )

    fitness = np.array([objective(x) for x in X])

    # Sort initial population
    order = np.argsort(fitness)
    X = X[order]
    fitness = fitness[order]

    best_pos = X[0].copy()
    best_fit = fitness[0]

    second_pos = X[1].copy()
    history = []

    for t in range(max_iter):

        # Update the two leading vultures
        order = np.argsort(fitness)
        X = X[order]
        fitness = fitness[order]

        best_pos = X[0].copy()
        second_pos = X[1].copy()

        progress = (t + 1) / max_iter

        for i in range(population_size):

            # Select one of the two leading vultures
            if np.random.rand() < 0.8:
                R = best_pos.copy()
            else:
                R = second_pos.copy()

            # Hunger rate
            h = np.random.uniform(-2, 2)
            z = np.random.uniform(-1, 1)
            r = np.random.rand()

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

            current = X[i].copy()
            D = np.abs(R - current)

            # -------------------------
            # Exploration: |F| >= 1
            # -------------------------
            if abs(F) >= 1:

                if np.random.rand() < p1:
                    X_new = R - F * D

                else:
                    X_new = (
                        R - F
                        + np.random.rand(dim)
                        * (
                            np.random.rand(dim) * (ub - lb)
                            + lb
                        )
                    )

            # -------------------------
            # Exploitation: |F| < 1
            # -------------------------
            else:

                if abs(F) >= 0.5:

                    # Cooperative siege
                    if np.random.rand() < p2:
                        X_new = (
                            D * (F + np.random.rand(dim))
                            - (R - current)
                        )

                    else:
                        X_new = (
                            R
                            - (F + np.random.rand(dim)) * D
                        )

                else:

                    # Competitive siege
                    if np.random.rand() < p3:
                        X_new = (
                            R
                            - np.abs(D)
                            * F
                            * levy_flight(dim)
                        )

                    else:
                        X_new = (
                            R
                            - F
                            + np.random.rand(dim)
                            * levy_flight(dim)
                            * D
                        )

            # Keep the solution within bounds
            X[i] = np.clip(X_new, lb, ub)

        # Evaluate the updated population
        fitness = np.array([objective(x) for x in X])

        # Update global best
        idx = np.argmin(fitness)

        if fitness[idx] < best_fit:
            best_fit = fitness[idx]
            best_pos = X[idx].copy()

        history.append(best_fit)

    return best_pos, best_fit, history


# -----------------------------
# 4. Run the algorithm
# -----------------------------
if __name__ == "__main__":

    best_position, best_value, history = AVOA(
        objective=sphere,
        dim=30,
        population_size=30,
        max_iter=500,
        lb=-100,
        ub=100
    )

    print("Best position:")
    print(best_position)

    print("\nBest Sphere value:")
    print(best_value)

    # Plot convergence
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title("AVOA Optimization of Sphere Function")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

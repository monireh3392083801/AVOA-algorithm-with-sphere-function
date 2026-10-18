# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:29:56 2026

@author: rostami
"""

import numpy as np

class AVOA_Continuous:
    def __init__(self, n_vultures=20, n_dimensions=5, max_iter=100, lb=-10, ub=10):
        self.n_vultures = n_vultures
        self.n_dimensions = n_dimensions
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        # مقداردهی اولیه
        self.vultures = np.random.uniform(self.lb, self.ub, (self.n_vultures, self.n_dimensions))
        self.best_vulture = None
        self.best_fitness = float('inf')

    def sphere_fitness(self, position):
        return np.sum(np.square(position))

    def optimize(self):
        for t in range(self.max_iter):
            # محاسبه فیتنس برای تمام کرکس‌ها
            fitness_values = np.array([self.sphere_fitness(v) for v in self.vultures])

            # پیدا کردن بهترین کرکس
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < self.best_fitness:
                self.best_fitness = fitness_values[min_idx]
                self.best_vulture = self.vultures[min_idx].copy()

            # ضریب کنترلی برای تغییر فاز از Exploration به Exploitation
            # این ضریب باعث می‌شود در ابتدا جستجو گسترده و در انتها دقیق باشد
            a = 2 * (1 - (t / self.max_iter)) 

            for i in range(self.n_vultures):
                r1 = np.random.rand()
                
                # ایجاد یک بردار تصادفی برای جهت حرکت
                r2 = 2 * np.random.rand() - 1  # مقادیر بین -1 و 1
                
                if r1 < 0.5:
                    # --- مرحله جستجوی گسترده (Exploration) ---
                    # کرکس‌ها در این مرحله با قدرت بیشتر به دنبال نواحی جدید می‌گردند
                    self.vultures[i] = self.best_vulture + r2 * (a * (self.vultures[i] - self.best_vulture))
                else:
                    # --- مرحله جستجوی دقیق (Exploitation) ---
                    # کرکس‌ها به سمت بهترین نقطه "رگلاژ" می‌شوند
                    # استفاده از توان دوم a باعث می‌شود در اواخر تکرارها، حرکت‌ها بسیار ریز و دقیق شوند
                    self.vultures[i] = self.best_vulture - r2 * (a**2 * (self.best_vulture - self.vultures[i]))

                # اعمال محدودیت بازه (Boundary Handling)
                self.vultures[i] = np.clip(self.vultures[i], self.lb, self.ub)

            if (t + 1) % 10 == 0 or t == 0:
                print(f"Iteration {t+1}/{self.max_iter} | Best Fitness: {self.best_fitness:.10e}")

        return self.best_vulture, self.best_fitness

# --- تست نهایی ---
if __name__ == "__main__":
    n_dim = 5
    pop = 30 # افزایش جمعیت برای دقت بهتر
    avoa = AVOA_Continuous(n_vultures=pop, n_dimensions=n_dim, max_iter=100, lb=-10, ub=10)
    best_pos, best_val = avoa.optimize()

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Best Position: {best_pos}")
    print(f"Best Fitness (Target 0): {best_val:.15f}")

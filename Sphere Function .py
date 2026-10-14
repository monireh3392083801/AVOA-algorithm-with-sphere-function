# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:35:55 2026

@author: rostami
"""

import numpy as np

class AVOA_Continuous:
    def __init__(self, n_vultures=20, n_dimensions=5, max_iter=100, lb=-10, ub=10):
        self.n_vultures = n_vultures       # جمعیت (اجنت‌ها)
        self.n_dimensions = n_dimensions   # ابعاد (مثلاً x1 تا x5)
        self.max_iter = max_iter           # حداکثر تکرار
        self.lb = lb                       # حد پایین (مثلاً -10)
        self.ub = ub                       # حد بالا (مثلاً 10)
        
        # ۱. تشکیل ماتریس اولیه اجنت‌ها در بازه مشخص شده
        # هر سطر یک کرکس و هر ستون یک بعد است
        self.vultures = np.random.uniform(self.lb, self.ub, (self.n_vultures, self.n_dimensions))
        
        self.best_vulture = None
        self.best_fitness = float('inf')

    def sphere_fitness(self, position):
        """
        تابع هدف Sphere: مجموع مجذورات (Sum of Squares)
        هدف: رسیدن به مقدار صفر
        """
        return np.sum(np.square(position))

    def optimize(self):
        for t in range(self.max_iter):
            # محاسبه فیتنس برای هر سطر (هر کرکس)
            fitness_values = np.array([self.sphere_fitness(v) for v in self.vultures])

            # پیدا کردن بهترین کرکس در این تکرار
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < self.best_fitness:
                self.best_fitness = fitness_values[min_idx]
                self.best_vulture = self.vultures[min_idx].copy()

            # ضریب کاهش یافتن (برای کنترل جستجو)
            a = 2 * (1 - (t / self.max_iter)) 

            for i in range(self.n_vultures):
                # تولید عدد تصادفی برای تعیین نوع حرکت (شبیه‌سازی مراحل AVOA)
                r1 = np.random.rand()
                
                if r1 < 0.5:
                    # مرحله اول: جستجوی گسترده (Exploration)
                    # حرکت نسبت به بهترین کرکس با استفاده از رندوم بین -1 و 1
                    self.vultures[i] += (np.random.rand() * (self.best_vulture - self.vultures[i])) * a
                else:
                    # مرحله دوم: جستجوی دقیق (Exploitation)
                    # حرکت کوچک‌تر برای یافتن نقطه دقیق صفر
                    self.vultures[i] += (np.random.rand() * (self.best_vulture - self.vultures[i])) * (a**2)

                # رعایت محدودیت بازه (Boundary Handling)
                self.vultures[i] = np.clip(self.vultures[i], self.lb, self.ub)

            print(f"Iteration {t+1}/{self.max_iter} | Best Fitness: {self.best_fitness:.10f}")

        return self.best_vulture, self.best_fitness

# --- اجرای تست ---
if __name__ == "__main__":
    # تنظیمات طبق درخواست شما:
    # ۵ بعد (x1 تا x5)، جمعیت ۲۰، بازه ۱۰ تا منفی ۱۰
    n_dim = 5
    pop = 20
    
    avoa = AVOA_Continuous(n_vultures=pop, n_dimensions=n_dim, max_iter=100000, lb=-10, ub=10)
    best_pos, best_val = avoa.optimize()

    print("\n--- Results ---")
    print(f"Best Position (x1 to x5): {best_pos}")
    print(f"Minimum Value (Target is 0): {best_val}")

import datetime
from datetime import time

import numpy as np

np.random.seed(42)


def get_data(n: int, start: int, stop: int, a: float, b: float):
    """x - (n * 2); y - (n * 1) arrays"""
    x = np.linspace(start, stop, n)  # N точек от start до stop
    y = a * x + b + np.random.normal(0, 0.8, 100)  # добавляем шум
    y = y.reshape(-1, 1)
    x = np.vstack([x, np.ones(n)])
    return x, y


if __name__ == "__main__":
    _start = datetime.datetime.now()
    # 1. Данные
    n = 100
    true_a = 2.5
    true_b = 1.0
    true_w = np.array([[true_a], [true_b]])
    X, Y = get_data(n, 0, 10, true_a, true_b)

    # 2. инициализация
    w = np.zeros((2, 1))  # коэффициенты
    lr = 0.0001  # learning rate (шаг)
    epochs = 10_000

    # 3 обучение
    for epoch in range(epochs):
        w = w - lr * (X @ (X.transpose() @ w - Y))

        # каждые 100 эпох печатаем прогресс:
        if epoch % 100 == 0 or epoch in [epochs - 1, 2, 3]:
            print(f"epoch {epoch}: w = {w}")

    # 4. Результат:
    _finish = datetime.datetime.now()
    print("\n" + "=" * 30)
    print(f"истинные значения: w = {true_w.tolist()}")
    print(f"найденные значения: w = {w.tolist()}")
    print(f"time - {(_finish - _start)}")

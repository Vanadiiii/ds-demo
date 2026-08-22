import numpy as np

np.random.seed(42)


def get_data(n: int, start: int, stop: int, a: float, b: float):
    X = np.linspace(start, stop, n)  # N точек от start до stop
    Y = a * X + b + np.random.normal(0, 0.8, 100)  # добавляем шум
    return X, Y


if __name__ == "__main__":
    # 1. Данные
    n = 100
    true_a = 2.5
    true_b = 1.0
    X, Y = get_data(n, 0, 10, true_a, true_b)

    # 2. инициализация
    a = 0.0
    b = 0.0
    lr = 0.001  # learning rate (шаг)
    epochs = 10000

    # 3 обучение
    for epoch in range(epochs):
        # считаем производные (накапливаются суммы)
        grad_a = 0.0
        grad_b = 0.0

        for i in range(n):
            x = X[i]
            y = Y[i]
            y_pred = a * x + b
            err = y - y_pred

            # получаем ошибки (формула вычислена заранее - производные там, все дела, цепное правило, итд)
            grad_a += -2 * x * err
            grad_b += -2 * err

        # усредняем производные
        grad_a /= n
        grad_b /= n

        # обновляем коэффициенты
        a = a - lr * grad_a
        b = b - lr * grad_b

        # каждые 100 эпох печатаем прогресс:
        if epoch % 100 == 0:
            # считаем текущую ошибку mse (Mean square error) - среднеквадратичная ошибка
            mse = 0.0
            for i in range(n):
                y = a * X[i] + b
                mse += (Y[i] - y) ** 2
            mse /= n
            print(f"epoch {epoch}: a = {a:.4f}, b = {b:.4f}, MSE = {mse:.6f}")

    # 4. Результат:
    print("\n" + "=" * 30)
    print(f"истинные значения: a = {true_a}, b = {true_b}")
    print(f"найденные значения: a = {a:.5f}, b = {b:.5f}")

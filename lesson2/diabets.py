import numpy as np
from numpy._core import float64
from numpy.typing import NDArray
from sklearn.datasets import load_diabetes  # pyright:ignore[reportMissingTypeStubs]
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def source() -> tuple[NDArray[float64], NDArray[float64]]:
    x_y = load_diabetes(return_X_y=True)  # pyright:ignore[reportUnknownVariableType]
    x: NDArray[float64] = x_y[0]  # pyright:ignore[reportAssignmentType]
    y: NDArray[float64] = x_y[1]  # pyright:ignore[reportAssignmentType]

    # 1. добавляем bias - фиктивный столбец спереди для свободного члена (иначе график будет якобы от центра)
    x = np.c_[np.ones(x.shape[0]), x]  # pyright:ignore[reportAny]

    # 2. нормируем Y - слишком большие результаты
    y_mean, y_std = y.mean(), y.std()
    y = (y - y_mean) / y_std
    return x, y.reshape(-1, 1)


def solve_numpy_lgd():
    print("=" * 10 + " lgd by numpy " + "=" * 10)
    X, Y = source()
    w = np.zeros((X.shape[1], 1))

    epochs = 50_000
    lr = 0.05

    for _ in range(epochs):
        grad = X.T @ (X @ w - Y) / len(Y)
        w -= lr * grad

    # SGD ( -> 1 - sum((y_real - y_pred)**2) / sum((y_read - y_mean)**2))
    sgd = 1 - (
        np.sum(np.square(Y - X @ w)) / np.sum(np.square(Y - np.full(Y.shape, Y.mean())))
    )
    # or sklearn.metrics.r2_score(Y, X @ w) # it's easy))
    print(sgd)  # 0.5 - it's good for medical info!!

    print(w.tolist())


def solve_sklearn_lgd():
    print("=" * 10 + " lgd by sklearn " + "=" * 10)
    X, Y = source()

    model = LinearRegression()
    model.fit(X, Y)
    w = model.coef_

    sgd = r2_score(Y, X @ w.T)
    print(sgd)  # 0.5 - it's good for medical info!!

    print(w.tolist())


def solve_numpy_sgd():
    print("=" * 10 + " sgd by numpy " + "=" * 10)
    X, Y = source()
    w = np.zeros((X.shape[1], 1))

    epochs = 50_000
    lr = 0.05

    rnd = np.random.default_rng()
    for _ in range(epochs):
        # выбираем рандомную строку и вычисляем градиент ТОЛЬКО по ней..
        i: int = rnd.integers(0, X.shape[0])
        grad_i = X[i].reshape((-1, 1)) @ (X[i] @ w - Y[i])
        w -= lr * grad_i.reshape((-1, 1))

    sgd = r2_score(Y, X @ w)
    print(sgd)  # 0.5 !!! wow, it's works!!!
    print(w.tolist())


def solve_sklearn_sgd():
    print("=" * 10 + " sgd by sklearn " + "=" * 10)
    X, Y = source()

    model = make_pipeline(
        StandardScaler(),
        SGDRegressor(
            loss="squared_error",  # Квадратичная ошибка (как в обычной регрессии)
            penalty=None,  # Убираем штрафы за переобучение
            max_iter=5000,  # Даем модели достаточно времени дойти до минимума
            tol=1e-4,  # Критерий остановки, если ошибка перестала падать
            random_state=42,  # Фиксируем случайность для воспроизводимости
        ),
    )
    model.fit(X, Y)
    w = model.named_steps["sgdregressor"].coef_
    sgd = r2_score(Y, X @ w.reshape((-1, 1)))
    print(sgd)  # 0.04... something goes wrong))
    print(w)


if __name__ == "__main__":
    # solve_numpy_lgd()
    # print()
    # solve_sklearn_lgd()
    # print()
    solve_sklearn_sgd()

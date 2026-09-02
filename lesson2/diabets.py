from collections.abc import Callable

import numpy as np
from numpy._core import float64
from numpy.typing import NDArray
from pandas.core.generic import FloatFormatType
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


def __check_input(x: NDArray[float64], y: NDArray[float64], w: NDArray[float64]):
    if x.shape[0] != y.shape[0]:
        raise Exception(f"incorrect shape: x({x.shape}), y({y.shape})")
    if x.shape[1] != w.shape[0]:
        raise Exception(f"incorrect shape: x({x.shape}), w({w.shape})")
    if y.shape[1] != 1 or w.shape[1] != 1:
        raise Exception(f"incorrect shape: y({y.shape}), w({w.shape})")


def lgd_solver(
    x: NDArray[float64], y: NDArray[float64], w: NDArray[float64], **kwargs
) -> None:
    __check_input(x, y, w)

    lr = kwargs.get("lr", 0.05)

    grad = x.T @ (x @ w - y) / len(y)
    w -= lr * grad


def sgd_solver(
    x: NDArray[float64], y: NDArray[float64], w: NDArray[float64], **kwargs
) -> None:
    __check_input(x, y, w)
    lr = kwargs.get("lr", 0.05)
    rnd = kwargs.get("rnd", np.random.default_rng())

    # выбираем рандомную строку и вычисляем градиент ТОЛЬКО по ней..
    i: int = rnd.integers(0, x.shape[0])
    grad_i = x[i].reshape((-1, 1)) @ (x[i] @ w - y[i])
    w -= lr * grad_i.reshape((-1, 1))


def mini_batch_solver(
    x: NDArray[float64], y: NDArray[float64], w: NDArray[float64], **kwargs
) -> None:
    __check_input(x, y, w)
    lr = kwargs.get("lr", 0.05)
    rnd = kwargs.get("rnd", np.random.default_rng())
    batch_size = kwargs.get("batch_size", 10)

    def get_batch_indexes() -> list[int]:
        res: set[int] = set()
        while len(res) < batch_size:
            res.add(rnd.integers(0, x.shape[0]))

        return list(res)

    # выбираем рандомные строки (mini-batch) и вычисляем градиент ТОЛЬКО по ним..
    ii = get_batch_indexes()
    x_batch = x[ii]
    grad_ii = x_batch.T @ (x_batch @ w - y[ii])
    w -= lr * grad_ii


def epoch_solver(kind: str | None):
    if kind is None:
        kind = "lgd"
    kind = kind.lower()
    if kind == "lgd":
        return lgd_solver
    elif kind == "sgd":
        return sgd_solver
    elif kind == "mini_batch":
        return mini_batch_solver
    else:
        raise Exception("unknown solver kind")


def solve_by_numpy(kind: str):
    print("=" * 10 + f" solve by {kind} by numpy " + "=" * 10)
    X, Y = source()
    w = np.zeros((X.shape[1], 1))
    solver

    lr = 0.05
    epochs = 50_000
    for _ in range(epochs):
        solver(X, Y, w, lr=lr)

    sgd = 1 - (
        np.sum(np.square(Y - X @ w)) / np.sum(np.square(Y - np.full(Y.shape, Y.mean())))
    )
    print(sgd)

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


# todo - fix!!
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
    solve_by_numpy("mini_batch")

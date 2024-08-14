import jax.numpy as jnp
from jax import random
from src.fleiadex.data_module.thunderstorm_dataloader.classical.satel_loader import (
    SatelDataModule,
)
from src.fleiadex.data_module.thunderstorm_dataloader.classical.radar_loader import (
    RadarDataModule,
)
from src.fleiadex.data_module.thunderstorm_dataloader.classical.ts_loader import (
    TSDataModule,
)


def normalise(x):
    x_max = jnp.abs(x).max()
    return x / x_max


# Rule out sample points that do not meet barrier.
def wash(data, barrier):
    X = data[0]
    y = data[1]
    idx = []
    for i in range(y.shape[0]):
        idx.append((y[i, :, :, 0] >= barrier).all())
    idx = jnp.array(idx)
    return X[idx], y[idx]


def timestack(data, pred_time, sample_arange):
    X = data[0]
    y = data[1]
    max_shift = jnp.max(jnp.array(sample_arange)) * 6
    Xs = []
    for i in sample_arange:
        if i == 0:
            Xs.append(X[max_shift:-pred_time])
        else:
            Xs.append(X[max_shift - 6 * i : -6 * i - pred_time])
    X = jnp.concatenate(Xs, axis=3)
    y = y[max_shift + pred_time :]
    return X, y


def train_test_split(data, ratio, seed):
    key = random.key(seed=seed)
    X = data[0]
    y = data[1]
    Xp = random.permutation(key=key, x=X)
    yp = random.permutation(key=key, x=y)
    l = X.shape[0]
    idx = int(l * (1 - ratio))
    X_train = Xp[:idx]
    y_train = yp[:idx]
    X_val = Xp[idx:]
    y_val = yp[idx:]
    return X_train, X_val, y_train, y_val


def data_setup(num_samples, pred_time, sample_arange):
    print("Start loading data")
    satel_dm = SatelDataModule(data_root="satel_array_")
    radar_dm = RadarDataModule(data_root="radar_array_")
    ts_dm = TSDataModule(data_root="satTS_array_")
    print("End loading data")
    satel_array = satel_dm.load_dataset(
        month_str="202312", num_samples=int(num_samples / 6) * 6
    )
    radar_array = radar_dm.load_dataset(
        month_str="202312", num_samples=int(num_samples / 6) * 6
    )
    TS_array = ts_dm.load_dataset(
        month_str="202312", num_samples=int(num_samples / 6) * 6
    )

    # Set up X and y
    X = jnp.stack((satel_array, radar_array), axis=3)
    y = jnp.stack((TS_array, radar_array), axis=3)

    # Preprocess
    X, y = wash((X, y), barrier=-0.1)
    print("Data washed")
    X, y = timestack((X, y), pred_time=pred_time, sample_arange=sample_arange)
    print("Data timestacked")

    X_train, X_test, y_train, y_test = train_test_split((X, y), ratio=0.2, seed=10)
    print("Done train-test splitting")

    return jnp.array(X_train), jnp.array(X_test), jnp.array(y_train), jnp.array(y_test)

from flax.training import train_state
from jax import jit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
)

import optax
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.pleiades.utils import *

loss_dict = {
    "l2_loss": l2_loss,
    "mixed_loss": mixed_loss,
    "mixed_TS_mask_loss": mixed_TS_mask_loss,
    "mixed_radar_mask_loss": mixed_radar_mask_loss,
}


# Calculate loss and gradient
@jit
def apply_model(state, loss, inputs, targets):
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, inputs, training=True)
        loss_val = loss(pred, targets)
        return loss_val.mean()

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss_val, grads = grad_fn(state.params)
    return loss_val, grads


def model_predict(state, model, inputs):
    params = state.params
    predictions = model.apply({"params": params}, inputs, training=False)
    return predictions


# parameter update
@jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


class UnetTrainer:
    def __init__(self, model, loss, key, max_epoch, batch_size, learning_rate, setup_x):
        self.model = model
        self.loss = loss
        self.key = key
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.state = None
        self.params = self.model.init(key, setup_x, training=False)

    def fit(self, X_train, X_test, y_train, y_test):
        batches = jnp.arange((X_train.shape[0] // self.batch_size) + 1)
        max_iter = len(batches) * self.max_epoch
        print("No. training iterations = %d" % max_iter)

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=self.learning_rate,
            peak_value=self.learning_rate,
            warmup_steps=int(max_iter * 0.1),
            decay_steps=max_iter,
            end_value=1e-6,
        )
        optimizer = optax.adam(learning_rate=lr_scheduler)  # Choose the method

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params["params"], tx=optimizer
        )

        runtime = 0
        train_epoch = 0
        loss_hist = []
        print("Start training")
        while train_epoch < self.max_epoch:
            for batch in batches[:-1]:
                # mini-batch update
                start_time = time.time()
                start, end = batch * self.batch_size, (batch + 1) * self.batch_size
                X_batch, y_batch = jnp.array(X_train[start:end]), jnp.array(
                    y_train[start:end]
                )  # single batch of data

                train_loss, grads = apply_model(self.state, self.loss, X_batch, y_batch)
                self.state = update_model(self.state, grads)

                end_time = time.time()
                runtime += end_time - start_time

            test_loss_set = []
            for i in range(y_test.shape[0]):
                start, end = i, i + 1
                X_batch, y_batch = X_test[start:end], y_test[start:end]
                sample_loss, _ = apply_model(self.state, self.loss, X_batch, y_batch)
                test_loss_set.append(sample_loss)
            test_loss = jnp.mean(jnp.array(test_loss_set))
            if train_epoch % 10 == 0:
                print(
                    "epoch = %04d,  time = %.2fs  |  Train Loss = %.2e  |   Test Loss = %.2e"
                    % (train_epoch, runtime, train_loss, test_loss)
                )
            loss_hist.append(jnp.log(jnp.array([train_loss, test_loss])))

            train_epoch += 1
        print("Training finished")
        loss_hist = jnp.array(loss_hist)
        fig_lh = plt.figure()
        ax_lh = fig_lh.add_subplot()
        ax_lh.plot(range(self.max_epoch), loss_hist[:, 0], label="Training Loss")
        ax_lh.plot(range(self.max_epoch), loss_hist[:, 1], label="Test Loss")
        # plt.show()
        return

    def metrics(self, X_test, y_test):
        predictions = []
        for i in range(y_test.shape[0]):
            start, end = i, i + 1
            inputs = X_test[start:end]
            pred = model_predict(self.state, self.model, inputs)
            predictions.append(pred)

        pred_bin = jnp.array(predictions)[:, :, :, :, 0].flatten()
        pred_num = jnp.array(predictions)[:, :, :, :, 1].flatten()
        pred_int = jnp.round(pred_bin).astype(int)

        y_bin_f = y_test[:, :, :, 0].flatten()
        y_num_f = y_test[:, :, :, 1].flatten()

        acc = accuracy_score(y_true=y_bin_f, y_pred=pred_int)
        auc_score = roc_auc_score(y_true=y_bin_f, y_score=pred_bin)
        precision, recall, barriers = precision_recall_curve(
            y_bin_f, pred_bin, pos_label=1
        )
        tn, fp, fn, tp = confusion_matrix(y_bin_f, pred_int, labels=[0, 1]).ravel()
        mae = (
            mean_absolute_error(y_num_f, pred_num) / jnp.linalg.norm(y_num_f) * 128**2
        )
        mse = (
            mean_squared_error(y_num_f, pred_num) / jnp.linalg.norm(y_num_f) * 128**2
        )
        print("Metrics calculated")
        return [acc, auc_score, tn, fp, fn, tp, mae, mse]

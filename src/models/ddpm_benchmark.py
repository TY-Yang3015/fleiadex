import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple, Union
from DiffusionNN import DiffusionNN, get_timestep_embedding
from IPython.display import clear_output


class DiffusionModel:
    def __init__(self, model: DiffusionNN,
                 dataset: tf.Tensor,
                 loss_fn: callable,
                 optimiser: tf.keras.optimizers.Optimizer):
        self.model = model
        self.dataset = dataset.reshape(-1, dataset.shape[1], dataset.shape[2], 1)
        self.beta_schedule = None
        self.alpha_schedule = None
        self.loss_fn = loss_fn
        self.optimiser = optimiser

    def set_beta_schedule(self, beta_schedule: np.ndarray) -> object:
        if np.max(beta_schedule) > 1 or np.min(beta_schedule) < 0:
            raise ValueError('betas must be between 0 and 1.')

        self.beta_schedule = tf.convert_to_tensor(beta_schedule, dtype=tf.float32)
        self._get_alpha_schedule()

        return self

    def _get_alpha_schedule(self) -> None:
        alpha_schedule = tf.math.cumprod(1 - self.beta_schedule)
        self.alpha_schedule = tf.convert_to_tensor(np.array(alpha_schedule), dtype=tf.float32)

        return None

    def forward_diffusion(self, x0: tf.Tensor, t: tf.int32) -> Tuple[tf.Tensor, tf.Tensor]:
        noise = tf.random.normal(tf.shape(x0))
        z_t = (tf.sqrt(self.alpha_schedule[t]) * x0 +
               tf.sqrt(1 - self.alpha_schedule[t]) * noise)
        return z_t, noise

    # @tf.function
    def _train_step(self) -> tf.float32:

        with tf.GradientTape() as tape:
            # tape.watch(self.model.trainable_variables)
            loss = 0.

            for t in range(1, len(self.beta_schedule) + 1):
                t = tf.convert_to_tensor(t, dtype=tf.int32)
                z_t, noise = self.forward_diffusion(self.dataset, t - 1)
                t = tf.convert_to_tensor([t], dtype=tf.int32)
                t_emb = get_timestep_embedding(tf.cast(t, tf.float32), 32)
                pred_noise = self.model([z_t, t_emb], training=True)
                loss += self.loss_fn(noise, pred_noise)
            loss /= len(self.dataset)

            grad = tape.gradient(loss, self.model.trainable_variables)
            del tape

            self.optimiser.apply_gradients(zip(grad
                                               , self.model.trainable_variables))

        return loss

    def train(self,
              train_step: int,
              disp_freq: int = 1
              ):

        for step in range(train_step):
            train_loss = self._train_step()

            if step % disp_freq == 0:
                print('step: {};  train elbo: {};'.format(step
                                                          , train_loss))
            clear_output(wait=True)

    def reverse_diffusion_step(self, z_t, t):
        predicted_noise = self.model([z_t, t])  # Predict noise
        alpha_t = self.alpha_schedule[t]
        sqrt_one_minus_alpha_t = tf.sqrt(1 - alpha_t)
        sqrt_recip_alpha_t = tf.sqrt(1 / alpha_t)

        z0_pred = sqrt_recip_alpha_t * (z_t - sqrt_one_minus_alpha_t * predicted_noise)
        return z0_pred

    def sample(self, num_of_sample):
        z_t = tf.random.normal((num_of_sample, self.dataset.shape[1], self.dataset.shape[2], 1))
        z_0 = self.reverse_diffusion_step(z_t, len(self.beta_schedule))
        return z_0

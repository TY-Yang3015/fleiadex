import jax.numpy as jnp
from flax.core import FrozenDict
from jax import random
import jax
import optax

import orbax.checkpoint as ocp
import etils.epath as path

from fleiadex.diffuser import DDPMCore
from fleiadex.utils import TrainStateWithDropout

from tqdm.auto import tqdm


def _get_optimiser(config: FrozenDict) -> optax.GradientTransformation:
    if isinstance(config["hyperparams"]["learning_rate"], float):
        return optax.adam(config["hyperparams"]["learning_rate"])
    else:
        try:
            optimiser = optax.adam(
                eval(config["hyperparams"]["learning_rate"], {"optax": optax})
            )
        except Exception as e:
            raise ValueError(
                f"unknown learning rate type: {e} \nplease follow optax syntax."
            )
        return optimiser


class ConditionalDiffuser4D:
    def __init__(self, ckpt_dir: dir, step: None | int = None):
        print(f"loading 4d conditional diffuser from {ckpt_dir}")

        if isinstance(ckpt_dir, str):
            ckpt_dir = path.Path(ckpt_dir)
        else:
            raise ValueError("ckpt dir must be str.")

        mngr = ocp.CheckpointManager(ckpt_dir, item_names=("diffuser_state", "config"))

        if step is not None:
            restored_config = mngr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
            )
        else:
            restored_config = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.Composite(config=ocp.args.JsonRestore()),
            )

        self.config = FrozenDict(restored_config["config"])

        self.diffuser = DDPMCore(
            config=self.config,
            diffusion_time_steps=self.config["hyperparams"]["diffusion_time_steps"],
        )

        init_data = jnp.ones(
            (
                self.config["hyperparams"]["batch_size"],
                self.config["nn_spec"]["sample_input_shape"][0],
                self.config["nn_spec"]["sample_input_shape"][1],
                self.config["nn_spec"]["sample_input_shape"][2] +
                self.config["nn_spec"]["cond_input_shape"][2],
            ),
            jnp.float32,
        )

        rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(42)}
        params = self.diffuser.init(rngs, init_data, random.key(0), False)["params"]

        self.diffuser_optimiser = _get_optimiser(self.config)
        self.diffuser_optimiser = optax.chain(
            optax.clip_by_global_norm(self.config['hyperparams']['gradient_clipping']),
            self.diffuser_optimiser,
            optax.ema(self.config['hyperparams']['ema_decay']),
        )

        diffuser_state = TrainStateWithDropout.create(
            apply_fn=self.diffuser.apply,
            params=params,
            tx=self.diffuser_optimiser,
            key=random.PRNGKey(42),
        )

        sharding = jax.sharding.NamedSharding(
            mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
            spec=jax.sharding.PartitionSpec(),
        )

        create_sharded_array = lambda x: jax.device_put(x, sharding)
        diffuser_state = jax.tree_util.tree_map(create_sharded_array, diffuser_state)
        diffuser_state = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, diffuser_state
        )

        if step is not None:
            restored = mngr.restore(
                step,
                args=ocp.args.Composite(diffuser_state=ocp.args.StandardRestore(diffuser_state)),
            )["diffuser_state"]
        else:
            restored = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.Composite(diffuser_state=ocp.args.StandardRestore(diffuser_state)),
            )["diffuser_state"]

        self.diffuser_state = restored

        print("loading succeeded.")
        del mngr

    def generate_example(self, condition, eval_key, progress_bar=True):
        if progress_bar:
            pbar = tqdm(total=self.config["hyperparams"]["diffusion_time_steps"])
            pbar.set_description('ddpm inferencing')

        z_t = jax.random.normal(
            eval_key,
            shape=(
                condition.shape[0],
                condition.shape[1],
                condition.shape[2],
                self.config["data_spec"]["sample_length"],
            ),
        )

        for t in reversed(range(self.config['hyperparams']['diffusion_time_steps'])):
            ts = jnp.full((condition.shape[0], 1, 1, 1), t, dtype=jnp.int32)
            pred_noise = self.diffuser_state.apply_fn(
                {'params': self.diffuser_state.params},
                z_t, condition, ts,
                method='pred_noise'
            )
            z_t = self.diffuser_state.apply_fn(
                {'params': self.diffuser_state.params},
                z_t, pred_noise, ts, eval_key,
                method='denoise')
            if progress_bar:
                pbar.update(1)

        if progress_bar:
            pbar.close()

        return z_t


diffuser = ConditionalDiffuser4D("/home/arezy/Desktop/fleiadex/outputs/2024-08-18/20-30-00/results/ckpt")

from fleiadex.data_module import FleiadexDataLoader
from fleiadex.utils import save_image

dataloader = FleiadexDataLoader(
    data_dir='../exp_data/satel_array_202312bandopt00_clear.npy',
    batch_size=8,
    auto_normalisation=True,
    output_image_size=64
)

_, data = dataloader.get_train_test_dataset()

cond = next(data)[..., :3]

res = diffuser.generate_example(cond, jax.random.PRNGKey(42))

save_image('./', jnp.array(cond), 'cond.png', 3)
save_image('./', jnp.array(res), 'res.png', 1)

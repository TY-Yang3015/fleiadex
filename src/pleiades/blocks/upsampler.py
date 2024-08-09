import flax.linen as nn
import jax.numpy as jnp
import jax
from einops import rearrange


class UpSampler3D(nn.Module):
    output_channels: int
    target_size: tuple[int, int, int]
    temporal_upsample: bool = False
    kernel_size: int = 3
    layout: str = "t h w c"

    def setup(self):
        self.conv = nn.Conv(
            features=self.output_channels,
            strides=(1, 1),
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.layout not in ["t h w c", "c t h w"]:
            raise ValueError(
                f"unsupported layout: {self.layout}, please choose one of the "
                f'following: "t h w c" or "c t h w".'
            )

    def __call__(self, x):
        if self.layout == "t h w c":
            batch_size, t, h, w, c = x.shape
            if self.temporal_upsample:
                x = jax.image.resize(
                    x, (batch_size,) + self.target_size + (c,), method="nearest"
                )
                x = self.conv(x)
                return x
            else:
                if t != self.target_size[0]:
                    raise ValueError(
                        "if temporal_upsample is False, the target size must "
                        "have the same temporal dimension as the input batch."
                    )
                x = jax.image.resize(
                    x, (batch_size,) + self.target_size + (c,), method="nearest"
                )
                x = self.conv(x)
                return x

        elif self.layout == "c t h w":
            x = rearrange(x, "b c t h w -> b t h w c")

            batch_size, t, h, w, c = x.shape
            if self.temporal_upsample:
                x = jax.image.resize(
                    x, (batch_size,) + self.target_size + (c,), method="nearest"
                )
                x = self.conv(x)
                x = rearrange(x, "b t h w c-> b c t h w")
                return x
            else:
                if t != self.target_size[0]:
                    raise ValueError(
                        "if temporal_upsample is False, the target size must "
                        "have the same temporal dimension as the input batch."
                    )
                x = jax.image.resize(
                    x, (batch_size,) + self.target_size + (c,), method="nearest"
                )
                x = self.conv(x)
                x = rearrange(x, "b t h w c-> b c t h w")
                return x


class UpSampler2D(nn.Module):
    output_channels: int
    target_size: tuple[int, int]
    kernel_size: int = 3
    layout: str = "h w c"

    def setup(self):
        self.conv = nn.Conv(
            features=self.output_channels,
            strides=(1, 1),
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.layout not in ["h w c", "c h w"]:
            raise ValueError(
                f"unsupported layout: {self.layout}, please choose one of the "
                f'following: "h w c" or "c h w".'
            )

    def __call__(self, x):
        if self.layout == "h w c":
            batch_size, h, w, c = x.shape
            x = jax.image.resize(
                x, (batch_size,) + self.target_size + (c,), method="nearest"
            )
            x = self.conv(x)
            return x

        elif self.layout == "c h w":
            x = rearrange(x, "b c h w -> b h w c")

            batch_size, h, w, c = x.shape

            x = jax.image.resize(
                x, (batch_size,) + self.target_size + (c,), method="nearest"
            )
            x = self.conv(x)
            x = rearrange(x, "b h w c-> b c h w")
            return x


# print(UpSampler(253, (5, 16, 16), layout='c t h w').tabulate(jax.random.PRNGKey(1)
#    , jnp.zeros((10, 256, 5, 8, 8)), console_kwargs={'width': 150}))

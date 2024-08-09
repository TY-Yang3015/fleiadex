from jax.tree_util import Partial
import jax.numpy as jnp

from src.pleiades.utils.loss_lib.common_loss import (
    binary_cross_entropy_with_logits,
    mse,
)


def cross_entropy(logits, labels):
    return binary_cross_entropy_with_logits(logits, labels)


def l2_loss(pred, target):
    return mse(target, pred)


@Partial
def mixed_loss(pred, target, weight=0.01):
    bin_pred, reg_pred = pred[:, :, :, 0], pred[:, :, :, 1]
    bin_target, reg_target = target[:, :, :, 0], target[:, :, :, 1]
    return jnp.mean(cross_entropy(bin_pred, bin_target)) + weight * l2_loss(
        reg_pred, reg_target
    )


@Partial
def mixed_TS_mask_loss(pred, target, weight=0.01):
    bin_pred, reg_pred = pred[:, :, :, 0], pred[:, :, :, 1]
    bin_target, reg_target = target[:, :, :, 0], target[:, :, :, 1]
    return (
        cross_entropy(bin_pred, bin_target)
        + weight * jnp.multiply(bin_pred, (reg_pred - reg_target) ** 2).mean()
    )


@Partial
def mixed_radar_mask_loss(pred, target, weight=0.01):
    bin_pred, reg_pred = pred[:, :, :, 0], pred[:, :, :, 1]
    radar_mask = jnp.where(reg_pred < 15, 0, 1).astype(int)
    bin_target, reg_target = target[:, :, :, 0], target[:, :, :, 1]
    return (
        cross_entropy(bin_pred, bin_target)
        + weight * jnp.multiply(radar_mask, (reg_pred - reg_target) ** 2).mean()
    )

import tensorflow as tf


def sse(epsilon_true, epsilon_pred):
    return tf.reduce_mean(tf.square(epsilon_true - epsilon_pred))

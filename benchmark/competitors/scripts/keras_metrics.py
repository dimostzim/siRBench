"""Validation metrics aggregated across the whole dataset rather than per batch."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="siRBench")
class GlobalR2(tf.keras.metrics.Metric):
    def __init__(self, name="r2_metric", dtype="float64", **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=dtype)
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros", dtype=dtype)
        self.sum_y_squared = self.add_weight(name="sum_y_squared", initializer="zeros", dtype=dtype)
        self.squared_error = self.add_weight(name="squared_error", initializer="zeros", dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, self.dtype), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, self.dtype), [-1])
        weights = tf.ones_like(y_true) if sample_weight is None else tf.broadcast_to(
            tf.reshape(tf.cast(sample_weight, self.dtype), [-1]), tf.shape(y_true))
        self.count.assign_add(tf.reduce_sum(weights))
        self.sum_y.assign_add(tf.reduce_sum(weights * y_true))
        self.sum_y_squared.assign_add(tf.reduce_sum(weights * tf.square(y_true)))
        self.squared_error.assign_add(tf.reduce_sum(weights * tf.square(y_true - y_pred)))

    def result(self):
        total = self.sum_y_squared - tf.math.divide_no_nan(tf.square(self.sum_y), self.count)
        score = 1 - tf.math.divide_no_nan(self.squared_error, total)
        constant_score = tf.cast(tf.equal(self.squared_error, 0), self.dtype)
        return tf.where(total > 0, score, constant_score)

    def reset_states(self):
        for variable in self.variables:
            variable.assign(tf.zeros_like(variable))

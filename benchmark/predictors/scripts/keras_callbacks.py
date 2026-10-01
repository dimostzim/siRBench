"""Checkpoint selection at either early stopping or the configured epoch cap."""
from tensorflow.keras.callbacks import EarlyStopping


class RestoreBestAtEnd(EarlyStopping):
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        # TensorFlow 2.4 otherwise restores only when patience is exhausted.
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)

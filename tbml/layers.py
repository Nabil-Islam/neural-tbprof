"""
Custom Keras layers.
"""

import tensorflow as tf
import tf_keras as keras


@keras.utils.register_keras_serializable(package="tbml")
class GatherColumns(keras.layers.Layer):
    """
    Reorder columns of a 2-D tensor by a fixed index list.

    Equivalent to `Lambda(lambda t: tf.gather(t, idx, axis=1))`, but
    implements get_config() so the model round-trips through save/load
    without needing custom_objects or safe_mode=False. Lambda layers
    pickle arbitrary Python bytecode and are a common cause of
    "cannot deserialize" errors when reloading a saved model later.

    Parameters
    ----------
    indices : sequence of int
        Column indices to gather, in output order.
    """

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.indices))

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices})
        return config

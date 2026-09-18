"""
Model-building helpers for the "anchored fine-tune" pattern: start a
Dense layer from a pretrained kernel, and regularize training to stay
close to that starting point.
"""

import tensorflow as tf
import tf_keras as keras


def make_anchor_regularizer(w_init, strength=1e-5):
    """
    Factory for an L2-to-anchor regularizer: penalizes weights for
    drifting away from an initial kernel `w_init`, rather than away
    from zero (as a normal L2 regularizer would).

    Parameters
    ----------
    w_init : array-like or tf.Tensor
        The reference kernel to anchor to (e.g. weights loaded from a
        pretrained model).
    strength : float
        Regularization strength.
    """
    w_init_tf = tf.constant(w_init, dtype=tf.float32)

    def anchor_reg(w):
        return strength * tf.reduce_sum(tf.square(w - w_init_tf))

    return anchor_reg


def build_anchored_dense_model(input_dim, n_outputs, kernel_init, bias_init,
                                layer_name='large_layer', activation='sigmoid',
                                anchor_strength=1e-5):
    """
    Build a single-Dense-layer Keras model whose weights start at
    (kernel_init, bias_init) and are regularized to stay near
    kernel_init during training.

    Returns the compiled-but-not-yet-`.compile()`ed keras.Model
    (weights are set; loss/optimizer are not).
    """
    anchor_reg = make_anchor_regularizer(kernel_init, strength=anchor_strength)

    inputs = keras.Input(shape=(input_dim,))
    outputs = keras.layers.Dense(
        n_outputs, activation=activation, name=layer_name,
        kernel_regularizer=anchor_reg)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.get_layer(layer_name).set_weights([kernel_init, bias_init])
    return model


def build_anchored_hidden_model(input_dim, n_outputs, kernel_init, bias_init,
                                 hidden_units=64, hidden_activation='relu',
                                 output_layer_name='large_layer',
                                 output_activation='sigmoid',
                                 anchor_strength=1e-5, dropout=0.0):
    """
    Like build_anchored_dense_model, but with one hidden Dense layer
    inserted before the anchored output layer. Only the *output* layer
    is anchored/pretrained — the hidden layer trains from a fresh
    (Keras-default) initialization.

    Use this when you're extending a previously-trained linear model
    (kernel_init/bias_init came from that model's output layer) with
    extra representational capacity, while still keeping the output
    layer close to its old, validated behavior.

    Parameters
    ----------
    hidden_units : int
        Width of the inserted hidden layer.
    hidden_activation : str
        Activation for the hidden layer (e.g. 'relu', 'tanh').
    dropout : float
        Optional dropout after the hidden layer (0.0 = disabled).
    """
    anchor_reg = make_anchor_regularizer(kernel_init, strength=anchor_strength)

    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(hidden_units, activation=hidden_activation,
                            name='hidden_layer')(inputs)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(
        n_outputs, activation=output_activation, name=output_layer_name,
        kernel_regularizer=anchor_reg)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.get_layer(output_layer_name).set_weights([kernel_init, bias_init])
    return model


def build_mlp(input_dim, n_outputs, hidden_layers=(128, 64),
              hidden_activation='relu', output_activation='sigmoid',
              dropout=0.0, output_layer_name='output'):
    """
    Generic configurable MLP: any number of hidden Dense layers, all
    freshly initialized (no anchoring/pretrained weights). Use this for
    architecture experiments where you're not warm-starting from an
    existing model at all.

    Parameters
    ----------
    hidden_layers : sequence of int
        Width of each hidden layer, in order. Empty tuple = plain
        logistic regression (input -> output directly).

    Example
    -------
    >>> model = build_mlp(X.shape[1], 15, hidden_layers=(256, 128, 64))
    """
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = keras.layers.Dense(units, activation=hidden_activation,
                                name=f'hidden_{i}')(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(n_outputs, activation=output_activation,
                                  name=output_layer_name)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def build_cross_resistance_model(n_snps, n_non_cr, cr_pairs,
                                  sparse_input=False,
                                  anchor_weights=None, anchor_strength=1e-5):
    """
    Build the cross-resistance architecture:

        snp_vec ──> cr_layer (linear, n_cr units) ──> swap partners
                │                                          │
                ├──────────────── concat ──────────────────┘
                │                    │
                │                 cr_pred (sigmoid, n_cr) ──┐
                │                                           ├─> concat -> output
                └──> r_layer (sigmoid, n_non_cr) ───────────┘

    IMPORTANT: the output column order is
        [non-CR drugs (n_non_cr, in r_layer weight order),
         CR drugs (in cr_pairs key order)]
    which is generally NOT the label matrix order. Use
    tbml.ordering.cross_resistance_output_order() to get the exact list
    and reorder your labels before training.

    Parameters
    ----------
    n_snps : int
        Number of input SNP features.
    n_non_cr : int
        Number of drugs handled by the plain r_layer.
    cr_pairs : dict
        Drug -> partner mapping (symmetric). Key order sets CR column order.
    sparse_input : bool
        Whether the Input accepts a SparseTensor. Sparse is fine for a
        one-shot forward pass but awkward for model.fit(); leave False
        when training and pass dense arrays.
    anchor_weights : dict, optional
        {layer_name: (kernel, bias)} to warm-start from, e.g. loaded from
        the naiive CR model. Layers present here get an anchor
        regularizer keeping them near those values; layers absent train
        from fresh init.
    anchor_strength : float

    Returns
    -------
    Uncompiled keras.Model.
    """
    from tbml.layers import GatherColumns
    from tbml.ordering import partner_permutation

    n_cr = len(cr_pairs)
    p_idx = partner_permutation(cr_pairs)
    anchor_weights = anchor_weights or {}

    def _reg(layer_name):
        if layer_name in anchor_weights:
            return make_anchor_regularizer(anchor_weights[layer_name][0],
                                            strength=anchor_strength)
        return None

    snp_vec = keras.Input(shape=(n_snps,), sparse=sparse_input)

    # Cross-resistance information, then swap each drug with its partner
    cr_layer = keras.layers.Dense(n_cr, activation=None, name='cr_layer',
                                   kernel_regularizer=_reg('cr_layer'))
    cr_info = cr_layer(snp_vec)
    cr_reorder = GatherColumns(p_idx, name='cr_swap')(cr_info)

    # Partner signal is concatenated back onto the SNP vector
    cr_pred = keras.layers.Dense(n_cr, activation='sigmoid', name='cr_pred',
                                  kernel_regularizer=_reg('cr_pred'))
    comb_cr = keras.layers.Concatenate(axis=1)([snp_vec, cr_reorder])
    cr_out = cr_pred(comb_cr)

    # Drugs with no modelled cross-resistance
    r_layer = keras.layers.Dense(n_non_cr, activation='sigmoid', name='r_layer',
                                  kernel_regularizer=_reg('r_layer'))
    r_out = r_layer(snp_vec)

    outputs = keras.layers.Concatenate(axis=1)([r_out, cr_out])
    model = keras.Model(inputs=snp_vec, outputs=outputs)

    for layer_name, (kernel, bias) in anchor_weights.items():
        model.get_layer(layer_name).set_weights([kernel, bias])

    return model

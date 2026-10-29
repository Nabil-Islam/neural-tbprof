"""
tbml: reusable training/eval/plotting utilities for the TB drug-resistance
masked multi-label models.

Prefer importing from submodules directly, e.g.:

    from tbml.losses import make_masked_weighted_bce
    from tbml.metrics import MaskedF1, MaskedAUC, confusion_stats

Common names are also available straight from the package (e.g.
`from tbml import confusion_stats`) via lazy attribute access below, so
that importing a pure-numpy submodule like tbml.thresholds doesn't force
tensorflow/tf_keras to be importable.
"""

__all__ = [
    "masked_binary_crossentropy",
    "make_masked_weighted_bce",
    "compute_pos_weight",
    "MaskedF1",
    "MaskedAUC",
    "confusion_stats",
    "PerDrugHistory",
    "pick_threshold",
    "set_plot_style",
    "plot_learning_curves",
    "plot_confusion_grid",
    "make_anchor_regularizer",
    "build_anchored_dense_model",
    "build_anchored_hidden_model",
    "build_mlp",
    "build_cross_resistance_model",
    "GatherColumns",
    "cross_resistance_output_order",
    "partner_permutation",
    "permutation_to",
    "reorder_columns",
    "assert_output_order",
    "load_layer_weights",
    "inspect_h5",
    "build_rule_weight_matrix",
    "build_cr_plus_signal_matrix",
    "sparse_to_kernel",
    "AnchorRegularizer",
    "strip_anchor_regularizers",
    "scores_to_calls",
    "save_prediction_tables",
]

_LAZY_MAP = {
    "masked_binary_crossentropy": "tbml.losses",
    "make_masked_weighted_bce": "tbml.losses",
    "compute_pos_weight": "tbml.losses",
    "MaskedF1": "tbml.metrics",
    "MaskedAUC": "tbml.metrics",
    "confusion_stats": "tbml.confusion",
    "PerDrugHistory": "tbml.callbacks",
    "pick_threshold": "tbml.thresholds",
    "set_plot_style": "tbml.plotting",
    "plot_learning_curves": "tbml.plotting",
    "plot_confusion_grid": "tbml.plotting",
    "make_anchor_regularizer": "tbml.models",
    "build_anchored_dense_model": "tbml.models",
    "build_anchored_hidden_model": "tbml.models",
    "build_mlp": "tbml.models",
    "build_cross_resistance_model": "tbml.models",
    "GatherColumns": "tbml.layers",
    "cross_resistance_output_order": "tbml.ordering",
    "partner_permutation": "tbml.ordering",
    "permutation_to": "tbml.ordering",
    "reorder_columns": "tbml.ordering",
    "assert_output_order": "tbml.ordering",
    "load_layer_weights": "tbml.weights",
    "inspect_h5": "tbml.weights",
    "build_rule_weight_matrix": "tbml.weight_matrices",
    "build_cr_plus_signal_matrix": "tbml.weight_matrices",
    "sparse_to_kernel": "tbml.weight_matrices",
    "AnchorRegularizer": "tbml.models",
    "strip_anchor_regularizers": "tbml.models",
    "scores_to_calls": "tbml.predictions",
    "save_prediction_tables": "tbml.predictions",
}


def __getattr__(name):
    # PEP 562 module-level lazy attribute access: `tbml.MaskedF1` etc.
    # only imports its owning submodule (and that submodule's deps,
    # e.g. tensorflow) at the point of use, not at `import tbml` time.
    module_path = _LAZY_MAP.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, name)

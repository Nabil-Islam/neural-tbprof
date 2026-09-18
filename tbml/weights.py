"""
Loading pretrained layer weights out of saved .h5 models.

The single-layer model only needed one kernel/bias pair; the
cross-resistance model needs three (cr_layer, cr_pred, r_layer), so this
generalises the lookup and is defensive about the dataset naming, which
differs between Keras versions ('kernel' vs 'kernel:0').
"""

import h5py


def _read_pair(group):
    """Pull (kernel, bias) out of an h5 group, tolerating name variants."""
    keys = list(group.keys())

    def find(stem):
        for candidate in (stem, f'{stem}:0'):
            if candidate in group:
                return group[candidate][:]
        raise KeyError(
            f"no '{stem}' dataset in group; available keys: {keys}"
        )

    return find('kernel'), find('bias')


def load_layer_weights(h5_path, layer_names):
    """
    Load {layer_name: (kernel, bias)} from a saved .h5 model.

    Parameters
    ----------
    h5_path : str or Path
    layer_names : sequence of str
        Layer names as they appear in the saved model, e.g.
        ['cr_layer', 'cr_pred', 'r_layer'] or ['large_layer'].

    Returns
    -------
    dict of layer_name -> (kernel ndarray, bias ndarray), suitable for
    passing straight to build_cross_resistance_model(anchor_weights=...).
    """
    out = {}
    with h5py.File(str(h5_path), 'r') as f:
        if 'model_weights' not in f:
            raise KeyError(
                f"'model_weights' not found in {h5_path}; "
                f"top-level keys are {list(f.keys())}. "
                "Was this saved by a different Keras version?"
            )
        mw = f['model_weights']
        for name in layer_names:
            if name not in mw:
                raise KeyError(
                    f"layer '{name}' not in saved weights; "
                    f"available layers: {list(mw.keys())}"
                )
            group = mw[name]
            # Keras nests the layer name twice: model_weights/foo/foo/kernel
            if name in group:
                group = group[name]
            out[name] = _read_pair(group)
    return out


def inspect_h5(h5_path):
    """
    Print the layer/dataset tree of a saved .h5 model. Useful when
    load_layer_weights raises and you need to see the actual structure.
    """
    with h5py.File(str(h5_path), 'r') as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f'  {name}  shape={obj.shape}')
        print(f'{h5_path}:')
        f.visititems(walk)

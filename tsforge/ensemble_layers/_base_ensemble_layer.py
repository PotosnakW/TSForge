from ..common._modules import IdentityLayer

class BaseEnsembleLayer:
    def __init__(self):
        pass

    def get_ensemble_layer(self, config):
        # Defaults to 'none' when unspecified — no config in configs/ sets this,
        # so requiring it made every model unconstructible. Note this gates on
        # ensemble_layer, NOT input_layer: keying the 'none' branch off
        # input_layer meant a real input_layer always fell through to the
        # ensembler branch (or an unreachable error), regardless of whether any
        # ensembling was actually requested.
        ensemble_key = getattr(config, "ensemble_layer", "none")

        if ensemble_key is None or str(ensemble_key).lower() == "none":
            return IdentityLayer()

        from .ensembler_torch import Ensembler
        return Ensembler(config)
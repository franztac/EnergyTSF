from models.gru import SimpleGRUModel
from models.transformer import TransformerModel
from models.mamba import MambaModel


def build_model_from_configs(configs, verbose: bool = False):
    model_name = configs.model_name

    if model_name == "gru":
        return SimpleGRUModel(configs=configs, verbose=verbose)

    if model_name == "transformer":
        return TransformerModel(configs=configs, verbose=verbose)

    if model_name == "mamba":
        return MambaModel(configs=configs, verbose=verbose)

    raise ValueError(f"Unsupported model_name: {model_name}")
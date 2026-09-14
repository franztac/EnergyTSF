import warnings


def configure_runtime_warnings(model_name: str | None = None) -> None:
    """
    Silence known non-critical warnings used in this project.
    """
    if model_name == "transformer":
        warnings.filterwarnings(
            "ignore",
            message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
            category=UserWarning,
        )
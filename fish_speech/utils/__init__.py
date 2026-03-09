from importlib import import_module

__all__ = [
    "enforce_tags",
    "extras",
    "get_metric_value",
    "RankedLogger",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "print_config_tree",
    "task_wrapper",
    "braceexpand",
    "get_latest_checkpoint",
    "autocast_exclude_mps",
    "set_seed",
]

_EXPORTS = {
    "braceexpand": (".braceexpand", "braceexpand"),
    "autocast_exclude_mps": (".context", "autocast_exclude_mps"),
    "get_latest_checkpoint": (".file", "get_latest_checkpoint"),
    "instantiate_callbacks": (".instantiators", "instantiate_callbacks"),
    "instantiate_loggers": (".instantiators", "instantiate_loggers"),
    "RankedLogger": (".logger", "RankedLogger"),
    "log_hyperparameters": (".logging_utils", "log_hyperparameters"),
    "enforce_tags": (".rich_utils", "enforce_tags"),
    "print_config_tree": (".rich_utils", "print_config_tree"),
    "extras": (".utils", "extras"),
    "get_metric_value": (".utils", "get_metric_value"),
    "task_wrapper": (".utils", "task_wrapper"),
    "set_seed": (".seed", "set_seed"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

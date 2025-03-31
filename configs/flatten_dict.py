from omegaconf import DictConfig


def flatten_dictconfig(cfg: DictConfig) -> DictConfig:
    current = cfg
    while isinstance(current, DictConfig) and len(current) == 1:
        current = list(current.values())[0]
    return current

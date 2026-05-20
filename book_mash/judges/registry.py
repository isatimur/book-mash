from book_mash.judges.base import JudgeDim

DIM_REGISTRY_VERSION = "0.1.0"

dim_registry: dict[str, type[JudgeDim]] = {}


def register_dim(cls: type[JudgeDim]) -> type[JudgeDim]:
    dim_registry[cls.name] = cls
    return cls

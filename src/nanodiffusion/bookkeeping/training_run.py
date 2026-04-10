from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .wandb_utils import maybe_init_wandb


@dataclass
class TrainingRunState:
    checkpoint_dir: Path
    wandb: Any = None


def _config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return vars(config).copy()
    raise TypeError(f"Unsupported config object: {type(config)!r}")


def setup_training_run(
    config: Any,
    model: Any,
    *,
    extra_run_config: Optional[Mapping[str, Any]] = None,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> TrainingRunState:
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_params = _config_to_dict(config)
    run_params["model_parameters"] = sum(p.numel() for p in model.parameters())
    if extra_run_config:
        run_params.update(extra_run_config)

    watch_model = getattr(config, "watch_model", False)
    wandb = maybe_init_wandb(
        getattr(config, "logger", "none"),
        run_params,
        model=model if watch_model else None,
        watch_model=watch_model,
        project_name=project_name,
        run_name=run_name,
    )
    return TrainingRunState(checkpoint_dir=checkpoint_dir, wandb=wandb)

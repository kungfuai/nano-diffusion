import os
import torch


def get_wandb_module():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "logger='wandb' requested but wandb is not installed. Use the repo Dockerfile or set --logger none."
        ) from exc
    return wandb


def maybe_init_wandb(
    logger,
    run_params,
    model=None,
    watch_model: bool = False,
    project_name: str = None,
    run_name: str = None,
):
    if logger != "wandb":
        return None

    wandb = get_wandb_module()
    project_name = project_name or os.getenv("WANDB_PROJECT") or "nano-diffusion"
    print(f"Logging to Weights & Biases project: {project_name}")
    init_kwargs = {"project": project_name, "config": run_params}
    if run_name is not None:
        init_kwargs["name"] = run_name
    wandb.init(**init_kwargs)
    if watch_model and model is not None:
        print("  Watching model gradients (can be slow)")
        wandb.watch(model)
    return wandb


def load_model_from_wandb(model, run_path, file_name):
    wandb = get_wandb_module()

    print(f"Restoring model from {run_path} and {file_name}")
    model_to_resume_from = wandb.restore(file_name, run_path=run_path, replace=True)
    model.load_state_dict(torch.load(model_to_resume_from.name, weights_only=True))
    print(f"Model restored from {model_to_resume_from.name}")
    return model

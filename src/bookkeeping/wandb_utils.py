def load_model_from_wandb(model, run_path, file_name):
    print(f"Restoring model from {run_path} and {file_name}")
    model_to_resume_from = wandb.restore(file_name, run_path=run_path, replace=True)
    model.load_state_dict(torch.load(model_to_resume_from.name, weights_only=True))
    print(f"Model restored from {model_to_resume_from.name}")
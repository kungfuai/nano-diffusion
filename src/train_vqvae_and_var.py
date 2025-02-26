from src.train_var import train_var, var_model_params, var_training_params
from src.train_vqvae import (
    make_log_dir,
    train_vqvae,
    vqvae_model_params,
    vqvae_training_params,
)


if __name__ == "__main__":
    dataset = "zzsi/afhq64_16k"
    log_dir = make_log_dir("logs/train_vqvae_and_var")

    vq_model = train_vqvae(dataset, vqvae_model_params, vqvae_training_params, log_dir)
    train_var(
        vq_model,
        dataset,
        var_model_params,
        var_training_params,
        log_dir,
    )

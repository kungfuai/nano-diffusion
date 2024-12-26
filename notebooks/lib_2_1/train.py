import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .config import TrainingConfig
from .data import load_data
from .model import create_unet_model
from .training_loop import train
from .diffusion import generate_samples_by_denoising, create_noise_schedule
from .bookkeeping import Bookkeeping


class DatasetBuilder:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_dataset, self.val_dataset = load_data(config)

    def training_dataset(self):
        return self.train_dataset

    def validation_dataset(self):
        return self.val_dataset


class TrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def fit(self, dataset_builder: DatasetBuilder, train_steps: int=10000):
        train_dataloader, val_dataloader = self._create_dataloaders(dataset_builder)
        self.noise_schedule = create_noise_schedule(n_T=self.config.num_denoising_steps, device=self.config.device)
        self.denoising_model = self._create_model(device=self.config.device)
        self.optimizer = self._create_optimizer(self.denoising_model)
        bookkeeping = Bookkeeping(config=self.config, denoising_model=self.denoising_model, noise_schedule=self.noise_schedule)
        bookkeeping.set_up_logger()
        train(
            config=self.config,
            model=self.denoising_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            noise_schedule=self.noise_schedule,
            optimizer=self.optimizer,
            steps=train_steps,
            silent=False,
            bookkeeping=bookkeeping,
        )
        return self.denoising_model

    def generate_samples(self, num_samples: int):
        num_channels = 3
        x_T = torch.randn(num_samples, num_channels, self.config.resolution, self.config.resolution)
        return generate_samples_by_denoising(
            denoising_model=self.denoising_model,
            x_T=x_T,
            noise_schedule=self.noise_schedule,
            num_samples=num_samples,
            device=self.config.device
        )

    def _create_model(self, device: str):
        # create the denoising model
        denoising_model = create_unet_model(self.config, device)
        return denoising_model

    def _create_optimizer(self, denoising_model: nn.Module):
        # create the optimizer
        config = self.config
        optimizer = optim.AdamW(denoising_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        return optimizer
    
    def _create_dataloaders(self, dataset_builder: DatasetBuilder):
        train_dataloader = DataLoader(
            dataset_builder.training_dataset(), batch_size=self.config.batch_size, shuffle=True, num_workers=2
        )
        val_dataloader = DataLoader(
            dataset_builder.validation_dataset(), batch_size=self.config.batch_size, shuffle=False, num_workers=2
        )
        return train_dataloader, val_dataloader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--num-denoising-steps', type=int, default=1000)
    parser.add_argument('--random-flip', action='store_true')
    parser.add_argument('--logger', type=str, default=None)
    parser.add_argument('--train-steps', type=int, default=10000)
    args = parser.parse_args()

    config = TrainingConfig(
        device=args.device,
        resolution=args.resolution,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_denoising_steps=args.num_denoising_steps,
        random_flip=args.random_flip,
        logger=args.logger,
    )
    dataset_builder = DatasetBuilder(config)
    training_pipeline = TrainingPipeline(config)
    training_pipeline.fit(dataset_builder, train_steps=args.train_steps)

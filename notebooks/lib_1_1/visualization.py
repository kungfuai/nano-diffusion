from typing import Tuple
import torch
import matplotlib.pyplot as plt
from .diffusion import generate_samples_by_denoising


class TrajectorySet:
    def __init__(self, embeddings):
        """
        Managing a set of trajectories, each of which is a sequence of embeddings.

        Parameters
        ----------
        embeddings: (n_timesteps, n_samples, *embedding_dims). This assumes
            the first dimension is time. And it is ordered from t=0 to t=n_timesteps-1.
            With t=0 representing the clean data and t=n_timesteps-1 representing the noise.

        """
        self.embeddings = embeddings
        self.embeddings_2d = None
    
    def run_tsne(self, n_components: int = 2, seed: int = 0, **kwargs):
        """Run t-SNE on the embeddings.
        """
        print(f"Running t-SNE on {self.embeddings.shape} embeddings...")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components, random_state=seed, **kwargs)
        flattened_embeddings = self.embeddings.reshape(-1, self.embeddings.shape[-1])
        flattened_embeddings_2d = tsne.fit_transform(flattened_embeddings)
        self.embeddings_2d = flattened_embeddings_2d.reshape(self.embeddings.shape[0], self.embeddings.shape[1], -1)
        print(f"t-SNE done. Shape of 2D embeddings: {self.embeddings_2d.shape}")
        return self.embeddings_2d
    
    def plot_trajectories(
            self,
            n: int = 10,
            show_figure: bool = False,
            noise_color: Tuple[float, float, float] = (0, 0, 1),  # blue
            data_color: Tuple[float, float, float] = (1, 0, 0),  # red
            figsize: tuple = (6, 6),
            xlim: Tuple[float, float] = None,
            ylim: Tuple[float, float] = None,
            with_ticks: bool = False,
            with_lines: bool = True,
            title: str = None,
            tsne_seed: int = 0,
            **kwargs):
        """Plot trajectories of some selected samples.

        This assumes the first dimension is time. And it is ordered from t=0 to t=n_timesteps-1.
        With t=0 representing the clean data and t=n_timesteps-1 representing the noise.

        Parameters
        ----------
        n: int
            number of samples to plot
        figsize: tuple
            figure size
        kwargs:
            other keyword arguments for matplotlib.pyplot.scatter
        """
        import numpy as np
        import matplotlib.pyplot as plt

        colors = []
        for t in range(self.embeddings.shape[0]):
            # interpolate between noise_color and data_color
            factor = t / (self.embeddings.shape[0] - 1)
            colors.append(np.array(noise_color) * factor + np.array(data_color) * (1 - factor))
        colors = np.array(colors)
        
        if self.embeddings_2d is None:
            if self.embeddings.shape[2] == 2:
                self.embeddings_2d = self.embeddings
            else:
                self.embeddings_2d = self.run_tsne(seed=tsne_seed)

        traj = self.embeddings_2d[:, :n, :]
        g = plt.figure(figsize=figsize)
        plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="red")  # real
        plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")  # noise
        plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.5, alpha=0.7, c=colors.repeat(n, axis=0))  # "olive"
        if with_lines:  
            plt.plot(traj[:, :n, 0], traj[:, :n, 1], c="olive", alpha=0.3)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if with_lines:
            plt.legend(["Data", "Noise", "Intermediate Samples (color coded)", "Trajectory"], loc="upper right")
        else:
            plt.legend(["Data", "Noise", "Intermediate Samples (color coded)"], loc="upper right")
        if not with_ticks:
            plt.xticks([])
            plt.yticks([])
        elif xlim is not None and ylim is not None:
            plt.xticks(xlim)
            plt.yticks(ylim)
        if title is not None:
            plt.title(title)
        if show_figure:
            plt.show()
        
        plt.tight_layout()
        # save to bytes (png)
        import io

        bytes_io = io.BytesIO()
        g.savefig(bytes_io, format="png")
        return bytes_io.getvalue()
        
        # # return the figure
        # return plt.gcf()

# visualize denoising trajectories
def visualize_denoising_trajectories(traj, step_size=1, num_samples=16):
    traj = traj[::step_size]
    traj_set = TrajectorySet(traj)
    _ = traj_set.plot_trajectories(n=num_samples, show_figure=True, figsize=(8, 8))


def visualize_denoising_model(model, noise_schedule, num_samples=16, timestep_size=50, num_denoising_steps=1000, device=None):
    data_dim = [2]
    x_T = torch.randn(num_samples, *data_dim)
    traj = generate_samples_by_denoising(
        model, x_T, noise_schedule,
        n_T=num_denoising_steps,
        device=device, return_full_trajectory=True
    ).cpu().detach().numpy()
    traj = traj[::timestep_size]
    visualize_denoising_trajectories(traj, num_samples=num_samples)


def visualize_sampled_data(model, noise_schedule, num_samples=128, device=None):
  # print("Loss of the denoising model:", loss.item())
  x_T = torch.randn(num_samples, 2)
  x_sampled = generate_samples_by_denoising(model, x_T, noise_schedule, n_T=1000, device=device).cpu().detach().numpy()

  # plt.scatter(x_sampled[:, 0], x_sampled[:, 1])
  fig, axs = plt.subplots(1, 1, figsize=(5, 5))
  axs.scatter(x_sampled[:,0], x_sampled[:,1], color='white', edgecolor='gray', s=5)
  # axs.set_axis_off()
  # plt.xlim(-3.6, 3.6)
  # plt.ylim(-3.6, 3.6)
import matplotlib.pyplot as plt
import numpy as np
import argparse

from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import IncrementalPCA
import torch

from guided_diffusion.patch_dataset import PatchBag

def take_pca(data: Dataset, n_components: int, batch_size: int) -> torch.Tensor:
    """
    Perform PCA on the data and return the transformed data

    Args:
        data (Dataset): The data to perform PCA on
        n_components (int): The number of components to keep
        batch_size (int): Batch size for DataLoader
    """
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ipca = IncrementalPCA(n_components=n_components)

    for batch in data_loader:
        X, _ = batch
        X = X.view(X.size(0), -1).numpy()
        ipca.partial_fit(X)

    X_pca_list = []
    for batch in data_loader:
        X, _ = batch
        X = X.view(X.size(0), -1).numpy()
        X_pca = ipca.transform(X)
        X_pca_list.append(X_pca)

    X_pca = np.vstack(X_pca_list)

    return torch.tensor(X_pca)

def plot_comps(comps1: torch.Tensor, comps2: torch.Tensor, save_path: str) -> None:
    """
    Plot the first two components of PCA over two datasets.

    Args:
        comps1 (torch.Tensor): PCA components for the first dataset.
        comps2 (torch.Tensor): PCA components for the second dataset.
        save_path (str): Path to save plot
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot first set of components
    ax1.scatter(comps1[:, 0], comps1[:, 1], alpha=0.5)
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_title('PCA Components - Set 1')

    # Plot second set of components
    ax2.scatter(comps2[:, 0], comps2[:, 1], alpha=0.5)
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_title('PCA Components - Set 2')

    plt.savefig(save_path)
    plt.show()

def compare_pcas(pb1: PatchBag, pb2: PatchBag, n_components: int, batch_size: int, save_path: str) -> None:
    """
    Full PCA Comparison pipeline

    Args:
        pb1 (PatchBag): PatchBag dataset of first dataset
        pb2 (PatchBag): PatchBag dataset of second dataset
        n_components (int): Number of components to keep in PCA
        batch_size (int): Batch size for DataLoader
        save_path (str): Path to save plot
    """
    pca_1 = take_pca(pb1, n_components=n_components, batch_size=batch_size)
    pca_2 = take_pca(pb2, n_components=n_components, batch_size=batch_size)
    plot_comps(pca_1, pca_2, save_path)

def main():
    parser = argparse.ArgumentParser()

    #PCA Params
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)

    #Data Params
    parser.add_argument('--wsi_path_1', type=str)
    parser.add_argument('--wsi_path_2', type=str)
    parser.add_argument('--h5_path_1', type=str)
    parser.add_argument('--h5_path_2', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()
    pb1 = PatchBag(args.wsi_path_1, args.h5_path_1)
    pb2 = PatchBag(args.wsi_path_2, args.h5_path_2)
    compare_pcas(pb1, pb2, args.n_components, args.batch_size, args.save_path)

if __name__ == '__main__':
    main()


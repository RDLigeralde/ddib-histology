import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import argparse

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.decomposition import PCA
import torch

from guided_diffusion.patch_dataset import PatchBag

def take_pca(data: Dataset, n_components: int, batch_size: int, subset_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform PCA on the data and return the transformed data

    Args:
        data (Dataset): The data to perform PCA on
        n_components (int): The number of components to keep
        batch_size (int): Batch size for DataLoader
        subset_size (int): Number of samples to take for PCA
    """
    indices = np.random.choice(len(data), subset_size, replace=False)
    subset = Subset(data, indices)
    data_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    pca = PCA(n_components=n_components)

    X, _ = next(iter(data_loader))
    X = X.view(X.size(0), -1).numpy()
    pca.fit_transform(X)
    return torch.tensor(X), torch.tensor(pca.components_)

def plot_comps(comps1: torch.Tensor, vecs1: torch.Tensor, comps2: torch.Tensor, vecs2: torch.Tensor, save_path: str) -> None:
    """
    Plot the first two components of PCA over two datasets.

    Args:
        comps1 (torch.Tensor): PCA components for the first dataset.
        vecs1 (torch.Tensor): PCA vectors for the first dataset.
        comps2 (torch.Tensor): PCA components for the second dataset.
        vecs2 (torch.Tensor): PCA vectors for the second dataset.
        save_path (str): Path to save plot
    """
    plt.scatter(comps1[:, 0], comps1[:, 1], alpha=0.5, label='Bright', color='orange')
    plt.scatter(comps2[:, 0], comps2[:, 1], alpha=0.5, label='Dim', color='blue')

    for i, (comp1, comp2) in enumerate(zip(vecs1, vecs2)):
        plt.arrow(0, 0, comp1[0]*2, comp1[1]*2, color='orange', width=0.01, label=f'Bright PC{i+1}' if i == 0 else "")
        plt.arrow(0, 0, comp2[0]*2, comp2[1]*2, color='blue', width=0.01, label=f'Dim PC{i+1}' if i == 0 else "")
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('PCA Components and Projections Overlay')
    plt.legend()

    plt.savefig(save_path)
    plt.show()

def compare_pcas(pb1: PatchBag, pb2: PatchBag, n_components: int, batch_size: int, subset_size: int, save_path: str) -> None:
    """
    Full PCA Comparison pipeline

    Args:
        pb1 (PatchBag): PatchBag dataset of first dataset
        pb2 (PatchBag): PatchBag dataset of second dataset
        n_components (int): Number of components to keep in PCA
        batch_size (int): Batch size for DataLoader
        subset_size (int): Number of samples to take for PCA
        save_path (str): Path to save plot
    """
    comps1,vecs1 = take_pca(pb1, n_components=n_components, batch_size=batch_size, subset_size=subset_size)
    comps2,vecs2= take_pca(pb2, n_components=n_components, batch_size=batch_size, subset_size=subset_size)
    plot_comps(comps1, vecs1, comps2, vecs2, save_path)

def main():
    parser = argparse.ArgumentParser()

    #PCA Params
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--subset_size', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=10_000)

    #Data Params
    parser.add_argument('--wsi_path_1', type=str)
    parser.add_argument('--wsi_path_2', type=str)
    parser.add_argument('--h5_path_1', type=str)
    parser.add_argument('--h5_path_2', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()
    pb1 = PatchBag(args.wsi_path_1, args.h5_path_1)
    pb2 = PatchBag(args.wsi_path_2, args.h5_path_2)
    compare_pcas(pb1, pb2, args.n_components, args.batch_size, args.subset_size, args.save_path)

if __name__ == '__main__':
    main()


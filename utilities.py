import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import KDTree
from MATE.tools import builder
import torch.nn as nn



def grad_freeze(model):
    for param in model.parameters():
        param.requires_grad = False
        

class PointDataset(Dataset):
    def __init__(self, data_root, label_path, corruption):
        # Construct the filename and load the data
        filename = os.path.join(data_root, f"data_{corruption}_5.npy")
        self.data = np.load(filename)

        # Load labels
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])
    
    
def load_base_model(args, config, load_part_seg=False):
    """
    Loads the base model, handles checkpoint loading, GPU usage, and distributed setup.
    
    Parameters:
    args (Namespace): Arguments containing model configurations and training setup.
    config (Config): Configuration object for the model.
    load_part_seg (bool): Flag to indicate whether to load part segmentation weights.

    Returns:
    torch.nn.Module: The initialized model.
    """
    
    # Build the base model
    base_model = builder.model_builder(config.model)
    
    # Load the model weights from checkpoint
    base_model.load_model_from_ckpt(args.pointmae_ckpt, load_part_seg)
    
    # Move the model to GPU if enabled
    if args.use_gpu:
        base_model = base_model.cuda()
    
    # Setup distributed or data parallel mode
    if args.distributed:
        if args.sync_bn:
            # Convert to synchronized batch normalization if required
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        
        # Wrap the model for distributed data parallel training
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True
        )
    else:
        # Use standard DataParallel if not in distributed mode
        base_model = nn.DataParallel(base_model).cuda()
    
    return base_model

    

def normalize(data):
    """
    Normalize and re-center the point cloud data so that it is confined within a unit cube.

    Args:
    - data (torch.Tensor): Point cloud data of shape (batch_size, num_points, num_dimensions).

    Returns:
    - torch.Tensor: Normalized and re-centered point cloud data.
    """
    # Find the max and min values for each point cloud in the batch
    data_max = data.max(dim=1, keepdim=True).values
    data_min = data.min(dim=1, keepdim=True).values
    
    # Compute the center of the point cloud
    data_center = (data_max + data_min) / 2
    
    # Re-center the point cloud by subtracting the center
    data = data - data_center
    
    # Find the maximum absolute value across all points and dimensions
    max_vals = data.view(data.shape[0], -1).max(dim=1).values.view(data.shape[0], 1, 1)
    
    # Normalize the point cloud to fit within the unit cube
    data = data / max_vals
    
    return data, data_center, max_vals


def unnormalize_data(data, max_vals, data_center):
    """
    Unnormalizes the input data by reversing the normalization process.
    
    Parameters:
    data (array-like): The normalized data.
    max_vals (array-like): The maximum values used during normalization.
    data_center (array-like): The center values (means or shifts) used during normalization.
    
    Returns:
    array-like: The unnormalized data.
    """
    # Rescale the data back to its original scale using max_vals
    data = data * max_vals
    
    # Shift the data back to its original center
    data = data + data_center
    
    return data




def interpolate_point_cloud(points, num_samples=1000, k=3):
    """
    Upsample a point cloud by interpolating new points between each point and its k nearest neighbors.

    Parameters:
    - points: numpy array of shape (N, 3), representing the original point cloud.
    - num_samples: int, number of new points to generate.
    - k: int, number of nearest neighbors to use for interpolation.

    Returns:
    - numpy array of shape (N + num_samples, 3), representing the upsampled point cloud.
    """
    n_points = points.shape[0]
    if n_points < k:
        raise ValueError("The point cloud must have at least k points.")

    # Create a KD-tree for efficient nearest neighbor queries
    tree = KDTree(points)

    # Select base points randomly
    base_indices = np.random.randint(0, n_points, size=num_samples)
    base_points = points[base_indices]

    # Find k nearest neighbors for each selected base point
    distances, indices = tree.query(base_points, k=k+1)
    
    # Remove the first column which represents the distance to themselves (zero)
    neighbor_indices = indices[:, 1:]
    neighbor_points = points[neighbor_indices.flatten()].reshape(num_samples, k, 3)

    # Choose random neighbors
    chosen_neighbors = neighbor_points[np.arange(num_samples), np.random.randint(0, k, size=num_samples)]

    # Generate random interpolation coefficients
    t = np.random.rand(num_samples, 1)

    # Compute new points
    new_points = (1 - t) * base_points + t * chosen_neighbors

    # Combine the original and new points
    upsampled_points = np.vstack((points, new_points))
    
    return upsampled_points


import numpy as np

def upsample(data_sample, num_samples=2048):
    """
    Upsample or downsample a point cloud to have a fixed number of points.

    Args:
    - data_sample (np.ndarray): The input point cloud of shape (num_points, num_dimensions).
    - num_samples (int): The desired number of points in the output point cloud (default: 2048).

    Returns:
    - np.ndarray: The upsampled or downsampled point cloud.
    """
    # If the point cloud has more points than needed, randomly downsample it
    if data_sample.shape[0] > num_samples:
        index = np.random.choice(data_sample.shape[0], num_samples, replace=False)
        data_sample = data_sample[index, :]
    # If the point cloud has fewer points, upsample it using interpolation
    elif data_sample.shape[0] < num_samples:
        data_sample = interpolate_point_cloud(data_sample, num_samples=num_samples - data_sample.shape[0], k=3)
    
    return data_sample


def upsample_all(data_batch, num_samples):
    """
    Upsample or downsample all point clouds in a batch to a fixed number of points.

    Args:
    - data_batch (np.ndarray): The batch of point clouds of shape (batch_size, num_points, num_dimensions).
    - num_samples (int): The desired number of points for each point cloud.

    Returns:
    - np.ndarray: A batch of point clouds where each has `num_samples` points.
    """
    # Initialize an empty array to store the upsampled point clouds
    data_batch_new = np.empty((data_batch.shape[0], num_samples, 3))
    
    # Upsample or downsample each point cloud in the batch
    for i in range(data_batch.shape[0]):
        data_batch_new[i] = upsample(data_batch[i], num_samples)
    
    return data_batch_new




def rotate_pointcloud(data_sample):
    """
    Rotate the point cloud by swapping the axes (x, y, z) to (z, y, x) and negating the x-axis.

    Args:
    - data_sample (np.ndarray): The input point cloud of shape (batch_size, num_points, 3).

    Returns:
    - np.ndarray: The rotated point cloud with axes swapped and the x-axis negated.
    """
    # Swap axes: (x, y, z) -> (z, y, x)
    data_sample = data_sample[:, :, [2, 1, 0]]
    
    # Negate the x-axis (now the first axis after swap)
    data_sample[:, :, 0] = -data_sample[:, :, 0]
    
    return data_sample


def rotateback_pointcloud(data_sample):
    """
    Undo the rotation by negating the x-axis and swapping the axes (z, y, x) back to (x, y, z).

    Args:
    - data_sample (np.ndarray): The input rotated point cloud of shape (batch_size, num_points, 3).

    Returns:
    - np.ndarray: The point cloud with the rotation undone, back to its original orientation.
    """
    # Negate the x-axis to reverse the effect
    data_sample[:, :, 0] = -data_sample[:, :, 0]
    
    # Swap axes back: (z, y, x) -> (x, y, z)
    data_sample = data_sample[:, :, [2, 1, 0]]
    
    return data_sample


from plot_tools import gif_save
import os
import torch
import numpy as np

from datetime import datetime


import argparse
import torch
from torch.utils.data import DataLoader
from utils_mate.config import *
from default_config import cfg as diff_config
from models.lion import LION
from utils_mate import misc
from default_config import cfg as configs
from utilities_3dd_tta import *
from tta import *
from tqdm import tqdm  


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=40, 
                    help='Batch size for processing data')

parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml", 
                    help='Path to the diffusion model config file')
parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt", 
                    help='Path to the diffusion model checkpoint')
parser.add_argument('--denoising_step', type=int, default=30, 
                    help="Number of denoising steps to perform (default: 30)")

# Dataset-related arguments
parser.add_argument('--dataset_name', type=str, default="modelnet-c", 
                    choices=["modelnet-c", "shapenet-c", "scanobjectnn-c"], 
                    help="Dataset name (options: modelnet-c, shapenet-c, scanobjectnn-c)")
parser.add_argument('--dataset_root', type=str, default="./data/modelnet40_c", 
                    help='Root directory of the dataset')
parser.add_argument('--corruption', type=str, default="background", 
                    help="Type of corruption to apply (default: background)")
parser.add_argument('--sample_id', type=int, default=11, 
                    help="sample id (default: 11 for airplane)")


# Shape latent and point updating factors
parser.add_argument('--gamma', type=float, default=0.01, 
                    help='Shape latent updating factor')
parser.add_argument('--eta', type=float, default=0.01, 
                    help='Latent point updating factor')
parser.add_argument('--lambdaa', type=float, default=0.95, 
                    help='SCD distance percentile')

# Device configuration
parser.add_argument('--device', type=str, default="cuda", 
                    help='Device to run the computations on (e.g., cuda, cpu)')

# Get the current time
current_time = datetime.now()

# Format the time as a string (e.g., "2024-11-27_15-30-45")
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

args = parser.parse_args()

points = np.load(f"./data/modelnet40_c/data_{args.corruption}_5.npy")
points = points[args.sample_id]


gif_save(points, os.path.join("./outputs/qualitative", f"{formatted_time}.gif"), multi_color=False)

points_tensor = torch.from_numpy(points)
points_tensor = points_tensor.unsqueeze(0)


        
diff_config.merge_from_file(args.diff_config)
diff_model = LION(configs)
diff_model.load_model(args.diff_ckpt)  

# Load the diffusion configuration and model
diff_config.merge_from_file(args.diff_config)
diff_model = LION(configs)
diff_model.load_model(args.diff_ckpt)



data_sample, data_center, data_max = normalize(points_tensor)
data_sample = upsample_all(data_sample.numpy(), 2048)
data_sample = torch.from_numpy(data_sample).float().to(args.device)

# Scale and rotate the point cloud data
data_sample = 3.3885 * data_sample
data_sample = rotate_pointcloud(data_sample)

# Perform Test-Time Adaptation (TTA) reconstruction
pred_points = tta_reconstruct(data_sample, diff_model, args.denoising_step, args.gamma, args.eta, args.lambdaa, 100)
pred_points = rotateback_pointcloud(pred_points)

pred_points, _, _ = normalize(pred_points)

# Apply farthest point sampling (FPS)
pred_points = misc.fps(pred_points, 1024)

pred_points = pred_points.cpu().squeeze().detach().numpy()

gif_save(pred_points, os.path.join("./outputs/qualitative", f"{formatted_time}_adapted.gif"), multi_color=False)

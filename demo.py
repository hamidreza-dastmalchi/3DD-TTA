import argparse
import torch
from torch.utils.data import DataLoader
from MATE.utils.config import *
from default_config import cfg as diff_config
from models.lion import LION
from MATE.utils import misc
from default_config import cfg as configs
from utilities import *
from tta import *


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size' , type=int, default=16, help='batch size')
parser.add_argument('--pointmae_config', type=str, default="./configs/tta_modelnet.yaml", help='yaml config file')
parser.add_argument('--pointmae_ckpt', type=str, default="./ckpts/modelnet_jt.pth", help='path to the pointmae checkpoint')
parser.add_argument('--diff_config', type=str, default="./configs/cfg.yml", help="Path to the diffusion model config file")
parser.add_argument('--diff_ckpt', type=str, default="./ckpts/epoch_10999_iters_2100999.pt", help="Path to the diffusion model checkpoint")
parser.add_argument('--dataset_name', type=str, default="modelnet-c", help='dataset name')
parser.add_argument('--dataset_root', type=str, default="data/Modelnet40_c", help='dataset root')
parser.add_argument('--label_path', type=str, default="data/Modelnet40_c/label.npy", help='label path')
parser.add_argument('--gamma', type=float, default=0.01, help='shape latent updating factor')
parser.add_argument('--eta', type=float, default=0.01, help='latent point updating factor')
parser.add_argument('--lambdaa', type=float, default=0.95, help='SCD distance percentile')
parser.add_argument('--device', type=str, default="cuda", help='device')





if __name__=="__main__":
    # Load the configuration using parsed arguments        
    args = parser.parse_args()
    # Check if GPU is available and set the appropriate flags
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    args.distributed = False
    
    


    # Load the pointmae configuration using parsed arguments
    config = cfg_from_yaml_file(args.pointmae_config)

    # Set the batch size for the training, extra training, validation, and test datasets
    config.dataset.train.others.bs = config.total_bs

    if config.dataset.get('extra_train'):
        config.dataset.extra_train.others.bs = config.total_bs * 2

    config.dataset.val.others.bs = config.total_bs * 2

    if config.dataset.get('test'):
        config.dataset.test.others.bs = config.total_bs

    # Overwrite model-specific configuration parameters
    config.model.transformer_config.mask_ratio = 0.9   # Update the mask ratio
    config.model.group_norm = False                     # Update group normalization setting

    if args.dataset_name == "modelnet-c":
        config.model.cls_dim = 40
    elif args.dataset_name == "shapenet-c":
        config.model.cls_dim = 55
    elif args.dataset_name == "scanobjectnn-c": 
        config.model.cls_dim = 15
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")     
    
    
    base_model = load_base_model(args, config, None)
    print('Load Source Model...')

    base_model.eval()           
    
    # Define the list of supported corruptions
    corruptions = [
            'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
            'distortion_rbf', 'distortion_rbf_inv', 'density', 'density_inc',
            'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
            ]


    
    
    diff_config.merge_from_file(args.diff_config)
    diff_model = LION(configs)
    diff_model.load_model(args.diff_ckpt)
        
    # Load the diffusion configuration and model
diff_config.merge_from_file(args.diff_config)
diff_model = LION(configs)
diff_model.load_model(args.diff_ckpt)

# Iterate over each corruption type in the dataset
for corruption in corruptions:
    preds = []
    targets = [] 
    if corruption == "background":
        num_steps = 35
    else:
        num_steps = 5
    
    # Define the dataset and dataloader
    dataset = PointDataset(args.dataset_root, args.label_path, corruption)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    for i, (data, label) in enumerate(dataloader):
        
        # Normalize and upsample the point cloud data
        data_sample, data_center, data_max = normalize(data)
        data_sample = upsample_all(data_sample.numpy(), 2048)
        data_sample = torch.from_numpy(data_sample).float().to(args.device)
        
        # Scale and rotate the point cloud data
        data_sample = 3.3885 * data_sample
        data_sample = rotate_pointcloud(data_sample)
        
        # Perform Test-Time Adaptation (TTA) reconstruction
        pred_points = tta_reconstruct(data_sample, diff_model, 5, args.gamma, args.eta, args.lambdaa, 100)
        pred_points = rotateback_pointcloud(pred_points)
        
        # Undo normalization based on the dataset
        if args.dataset_name == "scanobjectnn-c":
            pred_points = pred_points / 3.3885
            pred_points = unnormalize_data(pred_points, data_max, data_center)
        else:
            pred_points, _, _ = normalize(pred_points)
        
        # Apply farthest point sampling (FPS)
        pred_points = misc.fps(pred_points, 1024)
        
        # Perform classification using the base model
        with torch.no_grad():
            logits = base_model.module.classification_only(pred_points, only_unmasked=False)
            probs = torch.softmax(logits, dim=1)
            entropy = (-probs * torch.log2(probs)).sum(dim=1)  # Calculate entropy
            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)
            
            # Store predictions and targets for later accuracy calculation
            preds.append(pred)
            targets.append(target)
    
    # Concatenate predictions and targets for accuracy computation
    targets = torch.cat(targets)
    preds = torch.cat(preds).cpu()
    
    # Compute accuracy
    acc = (preds == targets).float().mean().item()
    
    print(f"Corruption: {corruption}, Accuracy: {acc}")
    with open("results.txt", "a") as result_file:
        print(f"Corruption: {corruption}, Accuracy: {acc}", file=result_file)
   
                
            
            
            
            

            
            
        
    

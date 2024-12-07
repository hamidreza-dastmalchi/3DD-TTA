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


def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()

    # Batch size
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size for processing data')

    # Configuration and checkpoint paths
    parser.add_argument('--pointmae_config', type=str, default="./cfgs/tta_modelnet.yaml", 
                        help='Path to the YAML config file for PointMAE')
    parser.add_argument('--pointmae_ckpt', type=str, default="./pointnet_ckpts/modelnet_jt.pth", 
                        help='Path to the PointMAE checkpoint')
    parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml", 
                        help='Path to the diffusion model config file')
    parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt", 
                        help='Path to the diffusion model checkpoint')

    # Dataset-related arguments
    parser.add_argument('--dataset_name', type=str, default="modelnet-c", 
                        choices=["modelnet-c", "shapenet-c", "scanobjectnn-c"], 
                        help="Dataset name (options: modelnet-c, shapenet-c, scanobjectnn-c)")
    parser.add_argument('--dataset_root', type=str, default="./data/modelnet40_c", help='Root directory of the dataset')
    parser.add_argument('--label_path', type=str, default="./data/modelnet40_c/label.npy", help='Path to the dataset labels')

    # Shape latent and point updating factors
    parser.add_argument('--gamma', type=float, default=0.01, help='Shape latent updating factor')
    parser.add_argument('--eta', type=float, default=0.01, help='Latent point updating factor')
    parser.add_argument('--lambdaa', type=float, default=0.95, help='SCD distance percentile')

    # Device configuration
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the computations on (e.g., cuda, cpu)')

    return parser.parse_args()


def configure_model(args):
    """Load and configure the base and diffusion models."""
    # Load PointMAE configuration
    config = cfg_from_yaml_file(args.pointmae_config)

    # Set classification dimensions based on dataset
    if args.dataset_name == "modelnet-c":
        config.model.cls_dim = 40
    elif args.dataset_name == "shapenet-c":
        config.model.cls_dim = 55
    elif args.dataset_name == "scanobjectnn-c":
        config.model.cls_dim = 15
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    # Load base model
    base_model = load_base_model(args, config, None)
    base_model.eval()
    print('Base model loaded successfully.')

    # Load diffusion model
    diff_config.merge_from_file(args.diff_config)
    diff_model = LION(configs)
    diff_model.load_model(args.diff_ckpt)
    print('Diffusion model loaded successfully.')

    return base_model, diff_model


def process_batches(dataloader, base_model, diff_model, args, num_steps):
    """Process batches of data and compute predictions."""
    preds, targets = [], []
    for data, label in tqdm(dataloader, desc="Processing Batches"):
        # Normalize and upsample the point cloud data
        data_sample, data_center, data_max = normalize(data)
        data_sample = upsample_all(data_sample.numpy(), 2048)
        data_sample = torch.from_numpy(data_sample).float().to(args.device)

        # Scale and rotate the point cloud data
        data_sample *= 3.3885
        data_sample = rotate_pointcloud(data_sample)

        # Perform Test-Time Adaptation (TTA) reconstruction
        pred_points = tta_reconstruct(data_sample, diff_model, num_steps, args.gamma, args.eta, args.lambdaa, 100)
        pred_points = rotateback_pointcloud(pred_points)

        # Undo normalization based on dataset
        if args.dataset_name == "scanobjectnn-c":
            pred_points /= 3.3885
            pred_points = unnormalize_data(pred_points, data_max, data_center)
        else:
            pred_points, _, _ = normalize(pred_points)

        # Apply farthest point sampling (FPS)
        pred_points = misc.fps(pred_points, 1024)

        # Perform classification using the base model
        with torch.no_grad():
            logits = base_model.module.classification_only(pred_points, only_unmasked=False)
            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)

        # Store predictions and targets
        preds.append(pred)
        targets.append(target)

    # Concatenate predictions and targets for accuracy computation
    return torch.cat(targets), torch.cat(preds).cpu()


def main():
    """Main function to execute the pipeline."""
    args = parse_arguments()
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    args.distributed = False

    base_model, diff_model = configure_model(args)

    for corruption in corruptions:
        # Set reconstruction steps based on corruption type
        num_steps = 35 if corruption == "background" else 5

        # Load dataset and dataloader
        dataset = PointDataset(args.dataset_root, args.label_path, corruption)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # Process the batches
        targets, preds = process_batches(dataloader, base_model, diff_model, args, num_steps)

        # Compute accuracy
        acc = (preds == targets).float().mean().item()
        print(f"Corruption: {corruption}, Accuracy: {acc}")

        # Save results to a file
        with open("./outputs/quantitative/results.txt", "a") as result_file:
            result_file.write(f"Corruption: {corruption}, Accuracy: {acc}\n")


if __name__ == "__main__":
    main()

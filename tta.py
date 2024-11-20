import torch
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamfer_grad
from diffusers import DDIMScheduler
from utilities import grad_freeze

def tta_reconstruct(x, lion, steps_back_local, gamma, eta, p, total=100):
    """
    Test-Time Adaptation (TTA) reconstruction using DDIMScheduler and Chamfer Distance.

    Args:
    - x: Input point cloud data.
    - lion: Model instance containing VAE and local prior.
    - steps_back_local: Percentage of total steps to use in reverse scheduling.
    - gamma: Step size for updating noisy_local.
    - eta: Step size for updating style_cond.
    - p: Proportion of points to consider in Chamfer Distance.
    - total: Total number of diffusion steps (default: 100).

    Returns:
    - pred_points: Reconstructed point cloud.
    """
    # Initialize chamfer distance and scheduler
    chamfer_dist = chamfer_grad()
    num_samples, num_points = x.size()[0], x.size()[1]
    
    scheduler = DDIMScheduler(
        beta_end=0.02, beta_schedule="linear", beta_start=0.0001, 
        clip_sample=False, num_train_timesteps=1000, prediction_type="epsilon"
    )
    scheduler.set_timesteps(total, device='cuda')
    
    steps_back_local = (total * steps_back_local) // 100
    timesteps_local = scheduler.timesteps[-steps_back_local:]
    alpha_bar_local = scheduler.alphas_cumprod[timesteps_local[0]]

    # Freeze gradients for VAE and local prior
    vae = lion.vae
    local_prior = lion.priors[1]
    grad_freeze(local_prior)
    grad_freeze(vae)
    
    # Latent encoding
    with torch.no_grad():
        latents = vae.encode(x)
        shape_latent = latents[2][0][0].unsqueeze(2).unsqueeze(3)
        latent_point = latents[2][1][0].unsqueeze(2).unsqueeze(3)
        latent_point_reshaped = latent_point.view(num_samples, 2048, -1)[:, :, :3]
        # latent_point_std = latent_point.std(dim=1).view(num_samples, 1, -1)
    
    # Global style conditioning
    style_cond = vae.global2style(shape_latent)
    
    # Add noise to the local latent vector
    noise = torch.randn_like(latent_point)
    noisy_latent_point = torch.sqrt(alpha_bar_local) * latent_point + noise * torch.sqrt(1 - alpha_bar_local)
 
    # Reverse diffusion process using DDIMScheduler
    for i, t in enumerate(timesteps_local):
        t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t + 1)
        noisy_latent_point = noisy_latent_point.detach()
        noisy_latent_point.requires_grad = True
        style_cond = style_cond.detach()
        style_cond.requires_grad = True

        # Predict noise
        noise_pred = local_prior(x=noisy_latent_point, t=t_tensor.float(), condition_input=style_cond, clip_feat=None)
        scheduler_output = scheduler.step(noise_pred, t, noisy_latent_point)
        pred_latent_point = scheduler_output.pred_original_sample
        
        # Compute Selective Chamfer Distance
        pred_latent_point_reshaped = pred_latent_point.view(num_samples, 2048, -1)[:, :, :3]
        dists1, dists2, _, _ = chamfer_dist(pred_latent_point_reshaped, latent_point_reshaped)
        dists1 = torch.sort(dists1, dim=1).values[:, :int(num_points * p)]
        dists2 = torch.sort(dists2, dim=1).values[:, :int(num_points * p)]
        ch_loss = dists1.sum() + dists2.sum()

        # Zero out gradients and backpropagate Chamfer loss
        if noisy_latent_point.grad is not None:
            noisy_latent_point.grad.zero_()
        ch_loss.backward()
        
        # Update latent variables with gradient step
        noisy_latent_point = scheduler_output.prev_sample - gamma * noisy_latent_point.grad
        style_cond = style_cond - eta * style_cond.grad

    # Decode the predicted points from VAE decoder
    pred_points = vae.decoder(
        None, beta=None, context=noisy_latent_point.squeeze(3).squeeze(2), 
        style=shape_latent.squeeze(3).squeeze(2)
    )
    
    return pred_points

import os
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gif_save(pts, gif_path, multi_color=False):
    """
    Creates a rotating 3D scatter plot of the point cloud and saves it as a GIF.
    
    Args:
    - pts (torch.Tensor): The input point cloud tensor of shape (batch_size, num_points, 3).
    - gif_path (str): Path where the GIF will be saved.
    - multi_color (bool): If True, the point cloud will be colored using two colors. Default is False.
    """
    
    # Detach, move to CPU, and convert the point cloud to numpy
    pts = pts[0].detach().cpu()
    
    # Center and scale the point cloud
    pc_max, _ = pts.max(dim=0, keepdim=True)
    pc_min, _ = pts.min(dim=0, keepdim=True)
    pc_min = pc_min[:, :3]
    pc_max = pc_max[:, :3]
    shift = ((pc_min + pc_max) / 2).view(1, 3)
    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
    pts[:, :3] = (pts[:, :3] - shift) / scale
    pts = pts.numpy()

    # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # Set color scheme based on the multi_color flag
    if multi_color:
        l = pts.shape[0]
        colors = ["blue"] * (l // 2) + ["red"] * (l // 2)
        ax.scatter(z, x, y, c=colors, s=1)
    else:
        ax.scatter(z, x, y, s=1)

    # List to store each frame for the GIF
    frames = []
    file_names = []

    # Ensure the temp folder exists
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    # Rotate the plot and capture each frame
    for angle in range(0, 360, 10):
        ax.view_init(30, angle)
        bound = 1.5
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_zlim(-bound, bound)
        plt.draw()
        plt.pause(0.005)  # Pause to ensure rendering

        # Save the current frame as a temporary image
        filename = os.path.join("tmp", f'frame_{angle}.png')
        file_names.append(filename)
        plt.savefig(filename)
        frames.append(imageio.imread(filename))

    # Create and save the GIF
    imageio.mimsave(gif_path, frames, fps=20)

    # Set axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Clean up the temporary files
    for filename in file_names:
        os.remove(filename)
